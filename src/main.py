#!/usr/bin/env python3
"""
MAIN PROCESSING PIPELINE FOR FB-SSEM OCCLUSION MASK GENERATION
===============================================================

What it does:
    - Scan the full FB-SSEM dataset (images0, images1, ...)
    - For each split: train / val / test
    - Find all BEV segmentation files:  images*/(train|val|test)/seg/bev/*.png
    - For each file:
          + Build a visibility mask (0/1/2)
          + Write results to:
                imagesX/split/seg/bev_raw/{id}.png   (0/1/2)
                imagesX/split/seg/bev_occ/{id}.png   (visualization 0/180/255)
      - If **--keep-occlusion**:
            1 = visible, 2 = occluded
        If **NOT** using --keep-occlusion:
            1 = visible, 0 = occluded (dropped to background)

    - Multiprocessing + tqdm for fast, clean progress.

Run example:
    python main.py \
        --dataset-root /media/.../FB-SSEM \
        --keep-occlusion \
        --num-workers 8
"""
# Quick overview:
# - Scans every split in the FB-SSEM dataset for BEV segmentations.
# - For each BEV file, builds 3D cuboids from the BEV mask + depth, projects them into every camera view,
#   and labels each BEV pixel as visible or occluded based on whether the cuboid is seen from any camera.
# - Writes two outputs beside the source BEV: `seg/bev_raw/<id>.png` (raw labels 0/1/2) and
#   `seg/bev_occ/<id>.png` (visualization 0/180/255). If `--keep-occlusion` is off, occluded pixels become 0.

import os
import sys
import cv2
import argparse
import logging
import multiprocessing as mp
from pathlib import Path
import numpy as np
from tqdm import tqdm

# --------------------------------------------------------------
# Add src/ to path
# --------------------------------------------------------------
project_root = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

# --------------------------------------------------------------
# Utils
# --------------------------------------------------------------
from utils.pipeline import (
    load_seg,
    load_depth,
    compute_height_map,
    segment_objects,
    get_2d_bounding_boxes,
    get_3d_bounding_boxes,
)
from utils.camera import load_intrinsics, load_extrinsics, load_camera_bev_height
from utils.visibility import compute_cuboid_visibility


# --------------------------------------------------------------
# Logger (quiet by default; enable when needed)
# --------------------------------------------------------------
logger = logging.getLogger("fb_ssem_main")
logger.setLevel(logging.ERROR)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)


# =====================================================================
# WORKER FUNCTION
# =====================================================================
def process_single_bev(args):
    """
    Process a single BEV file (runs inside a worker process).

    Args tuple:
        bev_path        : str - path to seg/bev/{id}.png
        intrinsics      : (K, D, xi)
        extrinsics      : dict extrinsic
        keep_occlusion  : bool
        visibility_thresh : float
        color_tol       : float
        resolution      : float
        min_area        : int
        offset          : float
        yshift          : float
    """
    (
        bev_path,
        intrinsics,
        extrinsics,
        keep_occlusion,
        visibility_thresh,
        color_tol,
        resolution,
        min_area,
        offset,
        yshift,
    ) = args

    bev_path = Path(bev_path)
    # example: .../images0/train/seg/bev/123.png  -> root = .../images0/train
    root = bev_path.parents[2]
    sid = bev_path.stem  # "123"

    # Output directories at the same level as bev/
    seg_dir = bev_path.parent                  # .../seg/bev
    bev_raw_dir = seg_dir.parent / "bev_raw"   # .../seg/bev_raw
    bev_occ_dir = seg_dir.parent / "bev_occ"   # .../seg/bev_occ
    bev_raw_dir.mkdir(exist_ok=True)
    bev_occ_dir.mkdir(exist_ok=True)

    # ----------------- 1. Load BEV seg -----------------
    try:
        bev_seg = load_seg(str(bev_path))
    except Exception:
        return f"[ERR] load bev: {bev_path}"

    H, W = bev_seg.shape[:2]
    obj_masks = segment_objects(bev_seg, min_area=min_area)
    if not obj_masks:
        # no object -> save zeros and skip
        bev_mask = np.zeros((H, W), np.uint8)
        cv2.imwrite(str(bev_raw_dir / f"{sid}.png"), bev_mask)
        cv2.imwrite(str(bev_occ_dir / f"{sid}.png"), bev_mask)
        return f"[OK-empty] {sid}"

    # ----------------- 2. Depth → height map -----------------
    depth_path = root / "depth" / f"{sid}.png"
    if not depth_path.exists():
        return f"[WARN] no depth: {depth_path}"

    depth_norm = load_depth(str(depth_path))

    cfg_path = root / "cameraconfig" / f"{sid}.txt"
    bev_cam_height = load_camera_bev_height(str(cfg_path)) if cfg_path.exists() else 10.0
    height_map = compute_height_map(depth_norm, bev_cam_height)

    # ----------------- 3. Cuboids -----------------
    boxes_2d = get_2d_bounding_boxes(obj_masks)
    cuboids = get_3d_bounding_boxes(
        boxes_2d,
        height_map,
        resolution,
        offset=offset,
        yshift=yshift,
    )
    if not cuboids:
        bev_mask = np.zeros((H, W), np.uint8)
        cv2.imwrite(str(bev_raw_dir / f"{sid}.png"), bev_mask)
        cv2.imwrite(str(bev_occ_dir / f"{sid}.png"), bev_mask)
        return f"[OK-no_cuboids] {sid}"

    # ----------------- 4. Camera segs -----------------
    cam_map = {
        "front": "Main Camera-front",
        "left": "Main Camera-left",
        "right": "Main Camera-right",
        "rear": "Main Camera-rear",
    }

    cam_segs = {}
    for view in cam_map:
        p = root / "seg" / view / f"{sid}.png"
        cam_segs[view] = load_seg(str(p)) if p.exists() else None

    K, D, xi = intrinsics

    # ----------------- 5. Visibility mask -----------------
    bev_mask_raw = np.zeros((H, W), np.uint8)

    for obj, cub in zip(obj_masks, cuboids):
        obj_color = np.asarray(obj["color"], np.float32)

        label = compute_cuboid_visibility(
            corners=cub["corners"],
            obj_color=obj_color,
            cam_segs=cam_segs,
            extrinsics=extrinsics,
            K=K,
            D=D,
            xi=xi,
            cam_name_map=cam_map,
            visibility_thresh=visibility_thresh,
            color_tol=color_tol,
            keep_occlusion=keep_occlusion
        )

    
        bev_mask_raw[obj["mask"] > 0] = label

    # ----------------- 6. Save output -----------------
    # raw 0/1/2 (or 0/1 if keep_occlusion=False)
    cv2.imwrite(str(bev_raw_dir / f"{sid}.png"), bev_mask_raw)

    # visualization 0/180/255 for quick viewing
    vis = np.zeros_like(bev_mask_raw)
    vis[bev_mask_raw == 1] = 180
    vis[bev_mask_raw == 2] = 255
    cv2.imwrite(str(bev_occ_dir / f"{sid}.png"), vis)

    return f"[OK] {sid}"


# =====================================================================
# MAIN
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True,
                    help="FB-SSEM root (contains CameraCalibrationParameters, images0, images1, ...)")
    ap.add_argument("--keep-occlusion", action="store_true",
                    help="If set: keep label=2 for occluded objects; "
                         "If not set: occluded objects -> label=0 (background)")
    ap.add_argument("--num-workers", type=int, default=8)

    ap.add_argument("--visibility-thresh", type=float, default=0.33)
    ap.add_argument("--color-tol", type=float, default=25.0)

    ap.add_argument("--resolution", type=float, default=100/(6*400))
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--offset", type=float, default=33.0)
    ap.add_argument("--yshift", type=float, default=-0.3)

    args = ap.parse_args()
    dataset_root = Path(args.dataset_root)

    # ----------------- Load intrinsics / extrinsics (global) -----------------
    intrinsics = load_intrinsics(
        dataset_root / "CameraCalibrationParameters" / "camera_intrinsics.yml"
    )
    extrinsics = load_extrinsics(
        dataset_root / "CameraCalibrationParameters" / "camera_positions_for_extrinsics.txt"
    )

    # ----------------- Collect ALL BEV files: train / val / test -----------------
    bev_files = []
    for img_dir in sorted(dataset_root.glob("images*")):
        for split in ["train", "val", "test"]:
            bev_dir = img_dir / split / "seg" / "bev"
            if bev_dir.exists():
                bev_files.extend(sorted(bev_dir.glob("*.png")))

    print(f"Found {len(bev_files)} BEV files across train/val/test.")

    if not bev_files:
        print("No bev files found. Check dataset-root path.")
        return

    # ----------------- Build worker args -----------------
    task_args = [
        (
            str(bev_path),
            intrinsics,
            extrinsics,
            args.keep_occlusion,
            args.visibility_thresh,
            args.color_tol,
            args.resolution,
            args.min_area,
            args.offset,
            args.yshift,
        )
        for bev_path in bev_files
    ]

    # ----------------- Multiprocessing + tqdm -----------------
    with mp.Pool(processes=args.num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_single_bev, task_args),
            total=len(task_args),
            desc="Processing BEV files",
            ncols=100,
        ):
            # keep loop silent so the progress bar stays clean
            pass

    print("=== DONE PROCESSING ALL BEV FILES (train/val/test) ===")


if __name__ == "__main__":
    main()
