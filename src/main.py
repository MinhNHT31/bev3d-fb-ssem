#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MAIN PIPELINE: FB-SSEM BEV VISIBILITY FILTERING
==============================================

Goal:
- For every BEV segmentation frame:
  1) Load BEV seg (RGB)
  2) Segment instance masks in BEV
  3) Load depth -> height map
  4) Build 3D cuboids (annotation.py)
  5) Call utils.visibility.compute_visible_bev_and_flags (SOURCE OF TRUTH)
  6) Save ONLY ONE output:
        seg/bev_visible/{id}.png

Strict rules:
- utils.visibility.py is the ONLY source of truth for visibility.
- NO recompute occlusion / FOV / geometry decision here.
- This script only prepares inputs and saves bev_visible returned by visibility.py.

Output:
imagesX/{train|val|test}/seg/bev_visible/{id}.png
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# ============================================================
# Path setup
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# ============================================================
# Imports (single source of truth)
# ============================================================
from utils.annotation import (
    load_seg,
    load_depth,
    compute_height_map,
    segment_objects,
    get_2d_bounding_boxes,
    build_cuboids_from_2d_boxes,
)
from utils.camera import (
    load_intrinsics,
    load_extrinsics,
    load_camera_bev_height,
)
from utils.visibility import compute_visible_bev_and_flags

# ============================================================
# Logger
# ============================================================
logger = logging.getLogger("fb_ssem_main")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)

# ============================================================
# Utils
# ============================================================
def imwrite_rgb(path: Path, img_rgb: np.ndarray) -> None:
    """Save RGB image using OpenCV (BGR on disk)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def safe_read_bgr(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    img = cv2.imread(str(path))
    return img if img is not None else None


# ============================================================
# Task config
# ============================================================
@dataclass(frozen=True)
class Task:
    bev_path: str
    intrinsics: Tuple[np.ndarray, np.ndarray, float]  # (K, D, xi)
    extrinsics: Dict[str, np.ndarray]

    visible_ratio_thresh: float
    min_pixels: int

    resolution: float
    min_area: int
    offset: float
    yshift: float

    skip_existing: bool


# ============================================================
# Worker
# ============================================================
def process_single_bev(task: Task) -> str:
    """
    Process ONE BEV frame -> seg/bev_visible/{id}.png
    """
    bev_path = Path(task.bev_path)

    # Expected: imagesX/{split}/seg/bev/{id}.png
    # root_split = imagesX/{split}
    root_split = bev_path.parents[2]  # .../imagesX/{split}
    sid = bev_path.stem

    out_dir = bev_path.parent.parent / "bev_visible"  # seg/bev_visible
    out_path = out_dir / f"{sid}.png"

    if task.skip_existing and out_path.exists():
        return f"[SKIP] {sid}"

    # --------------------------------------------------
    # 1) Load BEV segmentation (RGB)
    # --------------------------------------------------
    try:
        bev_seg_rgb = load_seg(str(bev_path))  # RGB
    except Exception as e:
        return f"[ERR] load bev {sid}: {type(e).__name__}"

    # --------------------------------------------------
    # 2) Segment objects in BEV
    # --------------------------------------------------
    obj_masks = segment_objects(bev_seg_rgb, min_area=int(task.min_area))
    if not obj_masks:
        # nothing to filter -> keep original
        imwrite_rgb(out_path, bev_seg_rgb)
        return f"[OK-empty] {sid}"

    # --------------------------------------------------
    # 3) Depth -> height map
    # --------------------------------------------------
    depth_path = root_split / "depth" / f"{sid}.png"
    if not depth_path.exists():
        # If no depth, safest is to keep original (do not delete objects blindly)
        imwrite_rgb(out_path, bev_seg_rgb)
        return f"[WARN-no-depth] {sid}"

    try:
        depth = load_depth(str(depth_path))
    except Exception as e:
        imwrite_rgb(out_path, bev_seg_rgb)
        return f"[WARN-bad-depth] {sid}: {type(e).__name__}"

    cfg_path = root_split / "cameraconfig" / f"{sid}.txt"
    try:
        bev_cam_h = load_camera_bev_height(str(cfg_path)) if cfg_path.exists() else 10.0
    except Exception:
        bev_cam_h = 10.0

    height_map = compute_height_map(depth, bev_cam_h)

    # --------------------------------------------------
    # 4) Build cuboids (annotation.py)
    # --------------------------------------------------
    boxes_2d = get_2d_bounding_boxes(obj_masks)
    cuboids = build_cuboids_from_2d_boxes(
        boxes_2d=boxes_2d,
        height_map=height_map,
        resolution=float(task.resolution),
        offset=float(task.offset),
        yshift=float(task.yshift),
    )

    if not cuboids:
        imwrite_rgb(out_path, bev_seg_rgb)
        return f"[OK-no-cuboids] {sid}"

    # --------------------------------------------------
    # 5) Load camera images (BGR as OpenCV default)
    #    NOTE: visibility.py may use images for shape/mask; we don't alter them here.
    # --------------------------------------------------
    cam_name_map = {
        "front": "Main Camera-front",
        "left":  "Main Camera-left",
        "right": "Main Camera-right",
        "rear":  "Main Camera-rear",
    }

    cam_images: Dict[str, Optional[np.ndarray]] = {}
    for view in cam_name_map.keys():
        p = root_split / "rgb" / view / f"{sid}.png"
        cam_images[view] = safe_read_bgr(p)

    K, D, xi = task.intrinsics

    # --------------------------------------------------
    # 6) Visibility (SOURCE OF TRUTH)
    #    - ego rule, camera masks, MEI projection, occlusion ordering
    #      are handled INSIDE utils.visibility.py.
    # --------------------------------------------------
    try:
        bev_visible_rgb, _, _ = compute_visible_bev_and_flags(
            bev_seg_rgb=bev_seg_rgb,
            obj_masks=obj_masks,
            cuboids=cuboids,
            cam_images=cam_images,
            extrinsics=task.extrinsics,
            K=K,
            D=D,
            xi=xi,
            cam_name_map=cam_name_map,
            visible_ratio_thresh=float(task.visible_ratio_thresh),
            min_pixels=int(task.min_pixels),
        )
    except Exception as e:
        # Fail-safe: keep original rather than producing broken masks
        imwrite_rgb(out_path, bev_seg_rgb)
        return f"[ERR-visibility] {sid}: {type(e).__name__}"

    # --------------------------------------------------
    # 7) Save ONLY bev_visible (RGB -> BGR on disk)
    # --------------------------------------------------
    imwrite_rgb(out_path, bev_visible_rgb)
    return f"[OK] {sid}"


# ============================================================
# Main
# ============================================================
def collect_bev_files(dataset_root: Path) -> List[Path]:
    bev_files: List[Path] = []
    for img_dir in sorted(dataset_root.glob("images*")):
        for split in ("train", "val", "test"):
            bev_dir = img_dir / split / "seg" / "bev"
            if bev_dir.exists():
                bev_files.extend(sorted(bev_dir.glob("*.png")))
    return bev_files


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset-root", required=True, help="FB-SSEM root directory")
    ap.add_argument("--num-workers", type=int, default=10)
    ap.add_argument("--skip-existing", action="store_true", help="Skip frames already in seg/bev_visible")

    # Visibility params (passed straight into compute_visible_bev_and_flags)
    ap.add_argument("--visible-ratio-thresh", type=float, default=0.01)
    ap.add_argument("--min-pixels", type=int, default=20)

    # BEV -> 3D params (annotation.py)
    ap.add_argument("--resolution", type=float, default=100 / (6 * 400))
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--offset", type=float, default=33.0)
    ap.add_argument("--yshift", type=float, default=-0.377)

    args = ap.parse_args()
    dataset_root = Path(args.dataset_root)

    calib_dir = dataset_root 
    intrinsics = load_intrinsics(calib_dir / "camera_intrinsics.yml")  # (K, D, xi)
    extrinsics = load_extrinsics(calib_dir / "camera_positions_for_extrinsics.txt")

    bev_files = collect_bev_files(dataset_root)

    print(f"Found {len(bev_files)} BEV files.")
    if not bev_files:
        return

    tasks = [
        Task(
            bev_path=str(p),
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            visible_ratio_thresh=args.visible_ratio_thresh,
            min_pixels=args.min_pixels,
            resolution=args.resolution,
            min_area=args.min_area,
            offset=args.offset,
            yshift=args.yshift,
            skip_existing=bool(args.skip_existing),
        )
        for p in bev_files
    ]

    # On Linux, fork is OK; on Windows/mac, spawn is safer.
    # Use env override if you want: MP_START_METHOD=spawn
    start_method = os.environ.get("MP_START_METHOD", "").strip() or None
    if start_method:
        try:
            mp.set_start_method(start_method, force=True)
        except RuntimeError:
            pass

    with mp.Pool(processes=int(args.num_workers)) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_single_bev, tasks),
            total=len(tasks),
            desc="Processing BEV",
            ncols=100,
        ):
            pass

    print("=== DONE: BEV VISIBILITY FILTERING COMPLETE ===")


if __name__ == "__main__":
    main()
