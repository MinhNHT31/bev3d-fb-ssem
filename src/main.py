#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MAIN PIPELINE: FB-SSEM BEV VISIBILITY FILTERING (FINAL VERSION)
==============================================================

Goal:
    - For every BEV segmentation in FB-SSEM
    - Build 3D cuboids from BEV + depth (annotation.py)
    - Run NEW visibility logic (visibility.py)
    - Output ONLY ONE BEV:
          seg/bev_visible/{id}.png

Rule:
    - Visible object  -> keep BEV pixels
    - Occluded object -> removed (set to background)

Output structure:
    imagesX/{train|val|test}/seg/bev_visible/{id}.png
"""

import os
import sys
import cv2
import argparse
import logging
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Optional

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
logger.setLevel(logging.ERROR)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)

# ============================================================
# Worker
# ============================================================
def process_single_bev(args):
    """
    Process ONE BEV file → output bev_visible/{id}.png
    """
    (
        bev_path_str,
        intrinsics,
        extrinsics,
        visible_ratio_thresh,
        min_pixels,
        resolution,
        min_area,
        offset,
        yshift,
    ) = args

    bev_path = Path(bev_path_str)

    # Example:
    # images0/train/seg/bev/123.png
    # -> root_split = images0/train
    root_split = bev_path.parents[2]
    sid = bev_path.stem

    out_dir = bev_path.parent.parent / "bev_visible"
    out_dir.mkdir(exist_ok=True)

    # --------------------------------------------------
    # 1) Load BEV segmentation
    # --------------------------------------------------
    try:
        bev_seg = load_seg(str(bev_path))  # RGB
    except Exception:
        return f"[ERR] load bev {sid}"

    H, W = bev_seg.shape[:2]

    # --------------------------------------------------
    # 2) Segment objects in BEV
    # --------------------------------------------------
    obj_masks = segment_objects(bev_seg, min_area=int(min_area))
    if not obj_masks:
        cv2.imwrite(str(out_dir / f"{sid}.png"), bev_seg)
        return f"[OK-empty] {sid}"

    # --------------------------------------------------
    # 3) Depth → height map
    # --------------------------------------------------
    depth_path = root_split / "depth" / f"{sid}.png"
    if not depth_path.exists():
        return f"[WARN] no depth {sid}"

    depth = load_depth(str(depth_path))

    cfg_path = root_split / "cameraconfig" / f"{sid}.txt"
    bev_cam_h = load_camera_bev_height(str(cfg_path)) if cfg_path.exists() else 10.0
    height_map = compute_height_map(depth, bev_cam_h)

    # --------------------------------------------------
    # 4) Build cuboids (annotation.py)
    # --------------------------------------------------
    boxes_2d = get_2d_bounding_boxes(obj_masks)
    cuboids = build_cuboids_from_2d_boxes(
        boxes_2d=boxes_2d,
        height_map=height_map,
        resolution=float(resolution),
        offset=float(offset),
        yshift=float(yshift),
    )

    if not cuboids:
        cv2.imwrite(str(out_dir / f"{sid}.png"), bev_seg)
        return f"[OK-no-cuboids] {sid}"

    # --------------------------------------------------
    # 5) Load RGB camera images
    # --------------------------------------------------
    cam_name_map = {
        "front": "Main Camera-front",
        "left":  "Main Camera-left",
        "right": "Main Camera-right",
        "rear":  "Main Camera-rear",
    }

    cam_images: Dict[str, Optional[np.ndarray]] = {}
    for view in cam_name_map:
        p = root_split / "rgb" / view / f"{sid}.png"
        cam_images[view] = cv2.imread(str(p)) if p.exists() else None

    K, D, xi = intrinsics

    # --------------------------------------------------
    # 6) Visibility (NEW logic)
    # --------------------------------------------------
    bev_visible, _, _ = compute_visible_bev_and_flags(
        bev_seg_rgb=bev_seg,
        obj_masks=obj_masks,
        cuboids=cuboids,
        cam_images=cam_images,
        extrinsics=extrinsics,
        K=K,
        D=D,
        xi=xi,
        cam_name_map=cam_name_map,
        visible_ratio_thresh=float(visible_ratio_thresh),
        min_pixels=int(min_pixels),
    )

    # --------------------------------------------------
    # 7) Save ONLY bev_visible
    # --------------------------------------------------
    cv2.imwrite(str(out_dir / f"{sid}.png"), bev_visible)

    return f"[OK] {sid}"

# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--dataset-root",
        required=True,
        help="FB-SSEM root directory",
    )
    ap.add_argument("--num-workers", type=int, default=8)

    # Visibility params
    ap.add_argument("--visible-ratio-thresh", type=float, default=0.33)
    ap.add_argument("--min-pixels", type=int, default=20)

    # BEV → 3D params
    ap.add_argument("--resolution", type=float, default=100 / (6 * 400))
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--offset", type=float, default=36.0)
    ap.add_argument("--yshift", type=float, default=-0.4)

    args = ap.parse_args()
    dataset_root = Path(args.dataset_root)

    # --------------------------------------------------
    # Load camera parameters
    # --------------------------------------------------
    intrinsics = load_intrinsics(
        dataset_root / "CameraCalibrationParameters" / "camera_intrinsics.yml"
    )
    extrinsics = load_extrinsics(
        dataset_root / "CameraCalibrationParameters" / "camera_positions_for_extrinsics.txt"
    )

    # --------------------------------------------------
    # Collect ALL BEV files
    # --------------------------------------------------
    bev_files: List[Path] = []
    for img_dir in sorted(dataset_root.glob("images*")):
        for split in ["train", "val", "test"]:
            bev_dir = img_dir / split / "seg" / "bev"
            if bev_dir.exists():
                bev_files.extend(sorted(bev_dir.glob("*.png")))

    print(f"Found {len(bev_files)} BEV files.")
    if not bev_files:
        return

    # --------------------------------------------------
    # Build worker arguments
    # --------------------------------------------------
    task_args = [
        (
            str(bev_path),
            intrinsics,
            extrinsics,
            args.visible_ratio_thresh,
            args.min_pixels,
            args.resolution,
            args.min_area,
            args.offset,
            args.yshift,
        )
        for bev_path in bev_files
    ]

    # --------------------------------------------------
    # Multiprocessing
    # --------------------------------------------------
    with mp.Pool(processes=args.num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_single_bev, task_args),
            total=len(task_args),
            desc="Processing BEV",
            ncols=100,
        ):
            pass

    print("=== DONE: BEV VISIBILITY FILTERING COMPLETE ===")


if __name__ == "__main__":
    main()
