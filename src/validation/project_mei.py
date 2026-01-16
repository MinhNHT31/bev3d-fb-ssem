#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
project_mei.py

Visual debugging utility for FB-SSEM BEV → 3D annotation.

This script is a PURE CONSUMER of annotation.py.
It does NOT implement geometry logic on its own.

Pipeline:
    BEV seg + depth
        ↓
    annotation.py (segment → 2D OBB → 3D cuboids)
        ↓
    project_mei.py
        → project cuboids onto camera images
        → visualize for sanity check

IMPORTANT CONVENTIONS:
--------------------------------------------------
1. Geometry (projection):
   - Uses WORLD → CAMERA extrinsics directly
   - No axis flipping
   - No extrinsic inversion

2. Image flipping:
   - FB-SSEM published images may be flipped for readability
   - Flip is applied ONLY FOR DISPLAY after projection

3. This script is for DEBUG ONLY
   - No occlusion
   - No visibility logic
"""

import argparse
import sys
import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Path setup
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# ============================================================
# Imports (annotation is the single source of truth)
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
from utils.bbox3d import draw_cuboids_curved


# ============================================================
# Display helper
# ============================================================
def flip_for_display(img, view):
    """
    Flip images ONLY for visualization.

    Geometry is always computed in the original camera coordinate system.
    """
    if view in ["front", "rear"]:
        img = cv2.flip(img, 1)
        img = cv2.flip(img, 0)
    return img


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True,
                    help="Path to imagesX/{train|val|test}")
    ap.add_argument("--id", required=True,
                    help="Frame id (filename stem)")
    ap.add_argument("--resolution", type=float, default=100 / (6 * 400))
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--offset", type=float, default=33)
    ap.add_argument("--yshift", type=float, default=-0.337)
    args = ap.parse_args()

    root = Path(args.dataset_root)
    fid = args.id

    # --------------------------------------------------
    # Load BEV + depth
    # --------------------------------------------------
    bev_path = root / "seg" / "bev" / f"{fid}.png"
    depth_path = root / "depth" / f"{fid}.png"
    camcfg_path = root / "cameraconfig" / f"{fid}.txt"

    if not bev_path.exists():
        raise FileNotFoundError(bev_path)

    bev_seg = load_seg(bev_path)
    depth = load_depth(depth_path)
    bev_cam_h = load_camera_bev_height(str(camcfg_path))

    # --------------------------------------------------
    # Build annotation (via annotation.py)
    # --------------------------------------------------
    objects = segment_objects(bev_seg, min_area=args.min_area)
    boxes_2d = get_2d_bounding_boxes(objects)
    height_map = compute_height_map(depth, bev_cam_h)

    cuboids = build_cuboids_from_2d_boxes(
        boxes_2d,
        height_map,
        resolution=args.resolution,
        offset=args.offset,
        yshift=args.yshift,
    )

    # --------------------------------------------------
    # Load camera parameters
    # --------------------------------------------------
    calib_root = root.parents[1] / "CameraCalibrationParameters"
    K, D, xi = load_intrinsics(calib_root / "camera_intrinsics.yml")
    extrinsics = load_extrinsics(calib_root / "camera_positions_for_extrinsics.txt")

    cam_map = {
        "front": "Main Camera-front",
        "left":  "Main Camera-left",
        "right": "Main Camera-right",
        "rear":  "Main Camera-rear",
    }

    # --------------------------------------------------
    # Project cuboids to camera images
    # --------------------------------------------------
    images = {}

    for view, cam_name in cam_map.items():
        img_path = root / "rgb" / view / f"{fid}.png"
        if not img_path.exists():
            images[view] = np.zeros((600, 800, 3), dtype=np.uint8)
            continue

        img = cv2.imread(str(img_path))
        img = flip_for_display(img, view)
        ext = extrinsics.get(cam_name, None)

        if ext is not None:
            img = draw_cuboids_curved(img, cuboids, ext, K, D, xi)

        img = flip_for_display(img, view)
        images[view] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --------------------------------------------------
    # Depth + BEV visualization
    # --------------------------------------------------
    depth_vis = cv2.normalize(
        cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH),
        None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2RGB)

    bev_vis = cv2.cvtColor(
        cv2.imread(str(bev_path)), cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"FB-SSEM BEV → 3D → Camera | offset={args.offset}, yshift={args.yshift}",
        fontsize=16,
        color="red",
    )

    panels = [
        ("Left", images["left"]),
        ("Front", images["front"]),
        ("Right", images["right"]),
        ("Depth", depth_vis),
        ("Rear", images["rear"]),
        ("BEV", bev_vis),
    ]

    for ax, (title, img) in zip(axs.flat, panels):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
