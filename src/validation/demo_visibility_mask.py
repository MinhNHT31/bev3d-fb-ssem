#!/usr/bin/env python3
"""
Visibility Mask Generator (v3)
==============================

Build an occlusion-aware BEV visibility map:
    0 = background
    1 = visible object  (seen by at least one camera)
    2 = occluded object (mostly hidden); set to 0 instead if --keep-occlusion is False

Pipeline overview:
1. BEV segmentation (RGB):
       - Colors represent semantic classes (ground, car, bus, ...).
       - segment_objects splits instances -> each object: {mask, color_RGB, label_id}
2. BEV depth + camera height:
       - Convert normalized depth into a height map for cuboid height estimation.
3. 3D cuboids:
       - BEV mask -> 2D OBB -> 3D cuboid with height from the height map (world space).
4. Visibility check:
       - Sample points on all 6 cuboid faces.
       - Project each point into the four camera segmentation maps.
       - If a camera pixel color matches the BEV object color within tolerance → visible ray.
       - Otherwise that ray is considered occluded.
       - visible_ratio = num_visible_rays / total_rays_projected
         ≥ threshold (33% default) → object visible (1)
         < threshold → object occluded (2 or 0 depending on flag)

Notes:
- Dataset provides semantic segmentation, so each object has a single class color.
  Occlusion is approximated by comparing BEV object color vs. camera segmentation pixel color.
- This color agreement is the core signal to infer which objects hide others.

Outputs:
- Raw visibility mask (0/1/2)
- Visualization mask (0=black, 1=gray, 2=white)

Built on:
- visibility.py (compute_cuboid_visibility + visibility_label)
- pipeline utils (segment_objects, cuboids, depth → height map)
"""

import os
import sys
import cv2
import numpy as np
import argparse
import logging
from pathlib import Path

# ------------------------------------------------------------
# Add src/ to sys.path
# ------------------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Local utilities
from utils.camera import load_intrinsics, load_extrinsics, load_camera_bev_height
from utils.pipeline import (
    load_depth,
    load_seg,
    compute_height_map,
    segment_objects,
    get_2d_bounding_boxes,
    get_3d_bounding_boxes,
)
from utils.visibility import compute_cuboid_visibility

# Script wrapper to convert cuboid visibility checks into BEV occlusion labels.

# ------------------------------------------------------------
# Setup logging
# ------------------------------------------------------------
logger = logging.getLogger("visibility_demo")
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--id", required=True)

    ap.add_argument("--resolution", type=float, default=100/(6*400))
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--offset", type=float, default=33.0)
    ap.add_argument("--yshift", type=float, default=-0.3)

    ap.add_argument("--visibility-thresh", type=float, default=0.33)
    ap.add_argument("--color-tol", type=float, default=25.0)
    
    # NEW: control keeping occlusion objects (2) or drop them (0)
    ap.add_argument("--keep-occlusion", action="store_true", help="keep occluded objects as label=2")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = ap.parse_args()

    logger.setLevel(getattr(logging, args.log_level))

    root = Path(args.dataset_root)
    sid = args.id

    # ---------------------------------------------------------
    # Load BEV segmentation
    # ---------------------------------------------------------
    bev_seg_path = root / "seg" / "bev" / f"{sid}.png"
    bev_seg = load_seg(str(bev_seg_path))

    H, W = bev_seg.shape[:2]

    obj_masks = segment_objects(bev_seg, min_area=args.min_area)
    logger.info(f"Found {len(obj_masks)} BEV objects")

    # ---------------------------------------------------------
    # Depth → height map
    # ---------------------------------------------------------
    depth_path = root / "depth" / f"{sid}.png"
    depth_norm = load_depth(str(depth_path))

    cfg_path = root / "cameraconfig" / f"{sid}.txt"
    bev_cam_height = load_camera_bev_height(str(cfg_path)) if cfg_path.exists() else 10.0

    height_map = compute_height_map(depth_norm, bev_cam_height)

    # ---------------------------------------------------------
    # Build cuboids
    # ---------------------------------------------------------
    boxes_2d = get_2d_bounding_boxes(obj_masks)
    cuboids = get_3d_bounding_boxes(
        boxes_2d, height_map, args.resolution,
        offset=args.offset, yshift=args.yshift
    )
    logger.info(f"Generated {len(cuboids)} cuboids")

    # ---------------------------------------------------------
    # Load camera intrinsics + extrinsics
    # ---------------------------------------------------------
    K, D, xi = load_intrinsics(Path(args.intrinsics))
    extrinsics = load_extrinsics(
        root.parent.parent / "CameraCalibrationParameters" / "camera_positions_for_extrinsics.txt"
    )

    cam_map = {
        "front": "Main Camera-front",
        "left": "Main Camera-left",
        "right": "Main Camera-right",
        "rear": "Main Camera-rear",
    }

    # ---------------------------------------------------------
    # Load camera segmentation images
    # ---------------------------------------------------------
    cam_segs = {}
    for view in cam_map:
        fp = root / "seg" / view / f"{sid}.png"
        cam_segs[view] = load_seg(str(fp)) if fp.exists() else None

    # ---------------------------------------------------------
    # Compute Visibility Mask
    # ---------------------------------------------------------
    bev_visibility = np.zeros((H, W), np.uint8)

    for i, (obj, cub) in enumerate(zip(obj_masks, cuboids)):
        obj_color = np.asarray(obj["color"], np.float32)

        label = compute_cuboid_visibility(
            cub["corners"],
            obj_color,
            cam_segs,
            extrinsics,
            K, D, xi,
            cam_map,
            args.visibility_thresh,
            args.color_tol,
            keep_occlusion=args.keep_occlusion
        )

        # Apply keep-occlusion flag:

        logger.debug(f"Object {i}: label={label}")

        bev_visibility[obj["mask"] > 0] = label

    # ---------------------------------------------------------
    # Save automatic output based on ID
    # ---------------------------------------------------------
    out_raw = f"visibility_{sid}.raw.png"
    out_vis = f"visibility_{sid}.png"

    cv2.imwrite(out_raw, bev_visibility)

    # Visualization mapping
    vis = np.zeros_like(bev_visibility)
    vis[bev_visibility == 1] = 180    # visible
    vis[bev_visibility == 2] = 255    # occluded

    cv2.imwrite(out_vis, vis)
    logger.info(f"[OK] Saved visibility maps: {out_vis}, {out_raw}")

if __name__ == "__main__":
    main()
