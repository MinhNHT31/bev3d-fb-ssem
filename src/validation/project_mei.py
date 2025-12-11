#!/usr/bin/env python3
"""
project_mei.py

Visual debugging utility to project estimated 3D cuboids and height/grids
onto camera images and BEV/depth visualizations.

Purpose:
- Load BEV (bird's-eye-view) segmentation, depth, and camera intrinsics/extrinsics.
- Run the pipeline that extracts objects in BEV and computes 3D cuboids.
- Project the cuboids back onto multiple camera views (front, left, right, rear)
  for visual inspection. Also optionally draw ground / cuboid bottom Y-planes.

Usage (example):
    python src/validation/project_mei.py \
        --dataset-root /path/to/images0/train \
        --intrinsics /path/to/camera_intrinsics.yml \
        --id 0 \
        --offset 33 \
        --yshift -0.3

Notes & assumptions:
- This script expects the repository layout:
    <project_root>/src/validation/project_mei.py
    <project_root>/src/utils/...
  So we add `<project_root>/src` to `sys.path` so `from utils...` imports work.
- The script is primarily for visualization; it uses matplotlib to show a 2x3 grid.
- Keep `sys.path` modification minimal; for more robust usage consider installing
  the package in editable mode or setting PYTHONPATH externally.
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys
import os

# Resolve project root relative to this file. The script is assumed to live in:
#   <project_root>/src/validation/project_mei.py
# So walking up three directories gives the repository root.
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure the repository `src/` directory is on sys.path so imports like
# `from utils.*` resolve when running the script directly.
# If you want the local `src` to take precedence over installed packages,
# consider `sys.path.insert(0, src_dir)`.
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Local utility modules (under src/utils/)
from utils.bbox3d import draw_cuboids_curved
from utils.camera import load_intrinsics, load_extrinsics, load_camera_bev_height
from utils.projects import cam2image
from utils.pipeline import (
    load_depth,
    compute_height_map,
    segment_objects,
    get_2d_bounding_boxes,
    get_3d_bounding_boxes,
)

# Keep logging quiet by default; this script is noisy visually so we only show errors.
logging.basicConfig(
    level=logging.ERROR,   # <-- only show real errors
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("CLEAN")


# ==========================================================
# Drawing helpers
# ==========================================================
def draw_y_planes_on_front(img, extrinsic, K, D, xi, cuboids):
    """
    Draw two Y-plane projections onto the front camera image:
      - the global ground plane at Y=0 (red)
      - the mean bottom plane of the first cuboid (blue)
    These are drawn as dense sampled points projected to image coordinates.

    Parameters:
    - img: OpenCV image (BGR) to draw on (modified in-place).
    - extrinsic: 4x4 camera-to-world (or world-to-camera?) transform used by cam2image.
    - K, D, xi: camera intrinsic params (passed to cam2image).
    - cuboids: list of cuboid dicts expected to have "corners" in camera coords.

    Returns:
    - img after drawing (same object returned for convenience).
    """
    if img is None or img.size == 0 or not cuboids:
        return img

    # Invert extrinsic to go camera->world or world->camera depending on stored format.
    # If inversion fails, abort drawing.
    try:
        cam2world = np.linalg.inv(extrinsic)
    except:
        return img

    # Camera center in world coordinates
    cam_center_world = (cam2world @ np.array([0,0,0,1], dtype=float).reshape(4,1)).flatten()
    cam_x, cam_y, cam_z = cam_center_world[:3]

    # Find the bottom (first 4 corners) of the first cuboid and compute its mean world Y.
    c = cuboids[0]["corners"]
    bottom_pts_cam = c[:4]
    pts_world = (cam2world @ np.hstack([bottom_pts_cam, np.ones((4,1))]).T).T
    y_world_mean = float(pts_world[:, 1].mean())

    # Build sampling grid in front of the camera (in world X-Z plane)
    xs = np.linspace(cam_x - 20, cam_x + 20, 41)
    zs = np.linspace(cam_z + 2, cam_z + 40, 40)
    X, Z = np.meshgrid(xs, zs)

    # Plane 1: Global ground Y=0 (red)
    Y0 = np.zeros_like(X)
    P0 = np.stack([X.ravel(), Y0.ravel(), Z.ravel()], axis=1)
    uv0, m0 = cam2image(P0, extrinsic, K, D, xi)
    uv0 = uv0[m0].astype(int)
    for (u, v) in uv0:
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.circle(img, (u, v), 1, (0, 0, 255), -1)

    # Plane 2: Cuboid bottom mean Y (blue)
    Y1 = np.full_like(X, y_world_mean)
    P1 = np.stack([X.ravel(), Y1.ravel(), Z.ravel()], axis=1)
    uv1, m1 = cam2image(P1, extrinsic, K, D, xi)
    uv1 = uv1[m1].astype(int)
    for (u, v) in uv1:
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.circle(img, (u, v), 1, (255, 0, 0), -1)

    return img


# ==========================================================
# Main visualization routine
# ==========================================================
def main():
    """
    Parse arguments, run 3D pipeline (BEV -> 3D cuboids), and project results
    onto multiple camera views for visual inspection.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--resolution", type=float, default=100/(6*400))
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--offset", type=float, default=33)
    ap.add_argument("--yshift", type=float, default=-0.3)
    args = ap.parse_args()

    # Root folder for a dataset sample (expects subfolders like rgb/, depth/, seg/bev/)
    root = Path(args.dataset_root)
    sample_id = args.id

    bev_path = root / "seg" / "bev" / f"{sample_id}.png"
    depth_path = root / "depth" / f"{sample_id}.png"
    cfg_path = root / "cameraconfig" / f"{sample_id}.txt"

    if not bev_path.exists():
        logger.error(f"Missing BEV mask: {bev_path}")
        return

    # Load BEV mask and compute object masks (pipeline utilities handle formats)
    bev_mask = (load_depth(str(bev_path)) > 0).astype("uint8") * 255
    obj_masks = segment_objects(bev_mask, min_area=args.min_area)
    depth_norm = load_depth(str(depth_path))

    # Optionally read camera-specific BEV height offset if available
    cam_h = None
    if cfg_path.exists():
        try:
            cam_h = load_camera_bev_height(str(cfg_path))
        except:
            cam_h = None

    # Compute per-pixel height map (from normalized depth / camera height info)
    height_map = compute_height_map(depth_norm, cam_h)

    # 2D object boxes in BEV and 3D cuboid estimation
    boxes_2d = get_2d_bounding_boxes(obj_masks)
    cuboids = get_3d_bounding_boxes(
        boxes_2d, height_map, args.resolution,
        args.offset, args.yshift
    )

    # Load intrinsics and extrinsics (extrinsics file maps camera names to poses)
    K, D, xi = load_intrinsics(Path(args.intrinsics))
    config_path = root.parent.parent / "CameraCalibrationParameters" / "camera_positions_for_extrinsics.txt"
    extrinsics_dict = load_extrinsics(config_path)

    # Mapping from simple view names to extrinsic keys in extrinsics_dict
    cam_name_map = {
        "front": "Main Camera-front",
        "left":  "Main Camera-left",
        "right": "Main Camera-right",
        "rear":  "Main Camera-rear",
    }

    processed_images = {}

    # For each view: read image, optionally flip to match coordinate convention,
    # draw cuboids, draw Y-planes on front, then convert to RGB for plotting.
    for view in ["front", "left", "right", "rear"]:
        img_path = root / "rgb" / view / f"{sample_id}.png"
        if not img_path.exists():
            # fallback black image for missing views
            processed_images[view] = np.zeros((600,800,3), dtype=np.uint8)
            continue

        img = cv2.imread(str(img_path))

        # Some datasets may require flipping the front/rear images to align axes.
        if view in ["front", "rear"]:
            img = cv2.flip(img, 1)
            img = cv2.flip(img, 0)
    
        ext_key = cam_name_map[view]
        if ext_key in extrinsics_dict:
            Extrinsic = extrinsics_dict[ext_key]

            # Draw curved cuboids (function expects image, cuboids, extrinsic, intrinsics)
            img = draw_cuboids_curved(img, cuboids, Extrinsic, K, D, xi)

            # Additional visualization: overlay sampled Y-planes for front camera
            if view == "front":
                img = draw_y_planes_on_front(img, Extrinsic, K, D, xi, cuboids)

        # Re-flip to original orientation for display consistency
        if view in ["front", "rear"]:
            img = cv2.flip(img, 1)
            img = cv2.flip(img, 0)

        # Convert BGR (OpenCV) to RGB for matplotlib
        processed_images[view] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Depth visualization: normalize and convert to RGB for display
    depth_viz = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    if depth_viz is None:
        depth_viz = np.zeros((100,100), dtype=np.uint8)

    depth_viz = cv2.normalize(depth_viz, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_viz = cv2.cvtColor(depth_viz, cv2.COLOR_GRAY2RGB)

    # BEV visualization: read and convert to RGB for display
    bev_viz = cv2.imread(str(bev_path))
    if bev_viz is not None:
        bev_viz = cv2.cvtColor(bev_viz, cv2.COLOR_BGR2RGB)
    else:
        bev_viz = np.zeros((100,100,3), dtype=np.uint8)

    # Build a 2x3 matplotlib figure (Left, Front, Right on top row;
    # Depth, Rear, BEV on bottom row) for inspection
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Offset={args.offset}, YShift={args.yshift}", fontsize=16, color='red')

    axes = [
        ("Left", processed_images["left"]),
        ("Front", processed_images["front"]),
        ("Right", processed_images["right"]),
        ("Depth", depth_viz),
        ("Rear", processed_images["rear"]),
        ("BEV", bev_viz),
    ]

    grid = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
    for (title, img), (r,c) in zip(axes, grid):
        axs[r,c].imshow(img)
        axs[r,c].set_title(title)
        axs[r,c].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
