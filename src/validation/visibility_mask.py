#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visibility_mask.py
==================

Visualization script for visibility results (NO DEBUG).

Pipeline:
- BEV segmentation → object masks
- BEV + depth → 3D cuboids
- Visibility filtering (FAR → NEAR logic in visibility.py)
- Visualize results directly (NO file saving)

Display:
  1) Ground Truth BEV
  2) BEV after visibility filtering
  3) Visibility mask (binary)
"""

import argparse
import sys
from pathlib import Path
import logging

import cv2
import numpy as np

# ============================================================
# Path setup
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# ============================================================
# Imports
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
# Logging
# ============================================================
logger = logging.getLogger("visibility_mask")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--id", required=True)

    # BEV → 3D params
    ap.add_argument("--resolution", type=float, default=100 / (6 * 400))
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--offset", type=float, default=36.0)
    ap.add_argument("--yshift", type=float, default=-0.4)

    # Visibility params
    ap.add_argument("--visible-ratio-thresh", type=float, default=1 / 3)
    ap.add_argument("--min-pixels", type=int, default=20)

    # Logging
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = ap.parse_args()
    logger.setLevel(getattr(logging, args.log_level))

    root = Path(args.dataset_root)
    fid = args.id

    # ========================================================
    # Load BEV & depth
    # ========================================================
    bev_path = root / "seg" / "bev" / f"{fid}.png"
    depth_path = root / "depth" / f"{fid}.png"
    cfg_path = root / "cameraconfig" / f"{fid}.txt"

    if not bev_path.exists():
        raise FileNotFoundError(bev_path)

    bev_seg = load_seg(str(bev_path))        # RGB
    depth = load_depth(str(depth_path))      # float32
    bev_cam_h = (
        load_camera_bev_height(str(cfg_path)) if cfg_path.exists() else None
    )

    H, W = bev_seg.shape[:2]

    # ========================================================
    # BEV → 3D cuboids
    # ========================================================
    obj_masks = segment_objects(bev_seg, min_area=args.min_area)
    logger.info(f"Found {len(obj_masks)} BEV objects")

    boxes_2d = get_2d_bounding_boxes(obj_masks)
    height_map = compute_height_map(depth, bev_cam_h)

    cuboids = build_cuboids_from_2d_boxes(
        boxes_2d,
        height_map,
        resolution=args.resolution,
        offset=args.offset,
        yshift=args.yshift,
    )

    # ========================================================
    # Camera params
    # ========================================================
    calib_root = root.parents[1] / "CameraCalibrationParameters"
    K, D, xi = load_intrinsics(calib_root / "camera_intrinsics.yml")
    extrinsics = load_extrinsics(
        calib_root / "camera_positions_for_extrinsics.txt"
    )

    cam_name_map = {
        "front": "Main Camera-front",
        "left":  "Main Camera-left",
        "right": "Main Camera-right",
        "rear":  "Main Camera-rear",
    }

    # Load RGB images (only for shape)
    cam_images = {}
    for view in cam_name_map:
        p = root / "rgb" / view / f"{fid}.png"
        cam_images[view] = cv2.imread(str(p)) if p.exists() else None

    # ========================================================
    # Visibility (CLEAN)
    # ========================================================
    bev_visible, visible_by_id, best_ratio = compute_visible_bev_and_flags(
        bev_seg_rgb=bev_seg,
        obj_masks=obj_masks,
        cuboids=cuboids,
        cam_images=cam_images,
        extrinsics=extrinsics,
        K=K,
        D=D,
        xi=xi,
        cam_name_map=cam_name_map,
        visible_ratio_thresh=args.visible_ratio_thresh,
        min_pixels=args.min_pixels,
    )

    # ========================================================
    # Visualization
    # ========================================================
    bev_gt = cv2.cvtColor(bev_seg, cv2.COLOR_RGB2BGR)
    bev_vis = cv2.cvtColor(bev_visible, cv2.COLOR_RGB2BGR)

    vis_mask = np.zeros((H, W), dtype=np.uint8)
    for i, o in enumerate(obj_masks):
        if visible_by_id.get(i, False):
            vis_mask[o["mask"] > 0] = 255
    vis_mask = cv2.cvtColor(vis_mask, cv2.COLOR_GRAY2BGR)

    # Resize
    disp_size = (512, 512)
    bev_gt = cv2.resize(bev_gt, disp_size)
    bev_vis = cv2.resize(bev_vis, disp_size)
    vis_mask = cv2.resize(vis_mask, disp_size)

    # Titles
    cv2.putText(bev_gt, "GT BEV", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(bev_vis, "BEV Visible", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(vis_mask, "Visibility Mask", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show
    concat = cv2.hconcat([bev_gt, bev_vis, vis_mask])
    cv2.imshow("GT | Visible | Mask", concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
