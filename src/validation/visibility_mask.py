#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visibility_mask.py
==================

Visualization script for visibility results (NO DEBUG logic inside visibility).

Display layout (2 rows):
--------------------------------------------------
Row 1:  Front RGB | Front Object Mask (color per object) | Front + BBox
Row 2:  GT BEV    | BEV Visible                          | Visibility Mask

Purpose:
- Debug fisheye valid region (radius_ratio = 1.0)
- Verify cuboid projection on FRONT camera
- Each object has its own color in front mask
"""

import argparse
import sys
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple

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
from utils.bbox3d import draw_cuboids_curved
from utils.visibility import (
    compute_visible_bev_and_flags,
    fisheye_visibility_mask,
    project_cuboid_to_mask,
)

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
# Color palette (BGR for OpenCV)
# ============================================================
PALETTE = [
    (255, 0, 0),    # blue
    (0, 255, 0),    # green
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
    (128, 0, 255),
    (255, 128, 0),
    (0, 128, 255),
]

def color_for_id(i: int):
    return PALETTE[i % len(PALETTE)]

# ============================================================
# Display helpers
# ============================================================
def prep(img: Optional[np.ndarray], title: str, size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    if img is None:
        out = np.zeros((size[1], size[0], 3), np.uint8)
    else:
        out = cv2.resize(img, size)

    cv2.putText(
        out,
        title,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return out

def flip_front(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if img is None:
        return None
    img = cv2.flip(img, 1)
    img = cv2.flip(img, 0)
    return img

# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--id", required=True)

    ap.add_argument("--resolution", type=float, default=100 / (6 * 400))
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--offset", type=float, default=36.0)
    ap.add_argument("--yshift", type=float, default=-0.4)

    ap.add_argument("--visible-ratio-thresh", type=float, default=0.15)
    ap.add_argument("--min-pixels", type=int, default=20)
    ap.add_argument("--r", default=0.85, type=float)

    args = ap.parse_args()
    root = Path(args.dataset_root)
    fid = args.id

    # --------------------------------------------------------
    # Load BEV & depth
    # --------------------------------------------------------
    bev_seg = load_seg(str(root / "seg" / "bev" / f"{fid}.png"))
    depth = load_depth(str(root / "depth" / f"{fid}.png"))
    cfg = root / "cameraconfig" / f"{fid}.txt"
    bev_cam_h = load_camera_bev_height(str(cfg)) if cfg.exists() else None

    H, W = bev_seg.shape[:2]

    # --------------------------------------------------------
    # BEV → cuboids
    # --------------------------------------------------------
    obj_masks = segment_objects(bev_seg, min_area=args.min_area)
    boxes_2d = get_2d_bounding_boxes(obj_masks)
    height_map = compute_height_map(depth, bev_cam_h)

    cuboids = build_cuboids_from_2d_boxes(
        boxes_2d, height_map,
        resolution=args.resolution,
        offset=args.offset,
        yshift=args.yshift,
    )

    # --------------------------------------------------------
    # Camera params
    # --------------------------------------------------------
    calib_root = root.parents[1] / "CameraCalibrationParameters"
    K, D, xi = load_intrinsics(calib_root / "camera_intrinsics.yml")
    extrinsics = load_extrinsics(calib_root / "camera_positions_for_extrinsics.txt")

    cam_name_map = {
        "front": "Main Camera-front",
        "left":  "Main Camera-left",
        "right": "Main Camera-right",
        "rear":  "Main Camera-rear",
    }

    cam_images = {
        v: cv2.imread(str(root / "rgb" / v / f"{fid}.png"))
        for v in cam_name_map
    }

    # --------------------------------------------------------
    # Visibility
    # --------------------------------------------------------
    bev_visible, visible_by_id, _ = compute_visible_bev_and_flags(
        bev_seg_rgb=bev_seg,
        obj_masks=obj_masks,
        cuboids=cuboids,
        cam_images=cam_images,
        extrinsics=extrinsics,
        K=K, D=D, xi=xi,
        cam_name_map=cam_name_map,
        visible_ratio_thresh=args.visible_ratio_thresh,
        min_pixels=args.min_pixels,
    )

    # --------------------------------------------------------
    # FRONT camera masks
    # --------------------------------------------------------
    front_img = cam_images["front"]

    Hf, Wf = front_img.shape[:2]
    fish_mask = fisheye_visibility_mask((Hf, Wf), radius_ratio=args.r)
    front_mask_color = np.zeros((Hf, Wf, 3), np.uint8)

    ext = extrinsics[cam_name_map["front"]]

    for i, cub in enumerate(cuboids):
        obj_mask = project_cuboid_to_mask(
            cub["corners"], ext, K, D, xi, (Hf, Wf)
        )
        obj_mask &= fish_mask

        color = color_for_id(i)
        front_mask_color[obj_mask] = color

    front_mask_color = flip_front(front_mask_color)

    # --------------------------------------------------------
    # Front + BBox
    # --------------------------------------------------------
    colored = []
    for i, cub in enumerate(cuboids):
        if i == 0:
            continue
        colored.append({
            "corners": cub["corners"],
            "color": tuple(c / 255.0 for c in color_for_id(i)),
        })

    front_bbox = draw_cuboids_curved(flip_front(front_img.copy()), colored, ext, K, D, xi)
    # front_bbox = flip_front(front_bbox)

    # --------------------------------------------------------
    # BEV views
    # --------------------------------------------------------
    bev_gt = cv2.cvtColor(bev_seg, cv2.COLOR_RGB2BGR)
    bev_vis = cv2.cvtColor(bev_visible, cv2.COLOR_RGB2BGR)

    vis_mask = np.zeros((H, W), np.uint8)
    for i, o in enumerate(obj_masks):
        if visible_by_id.get(i, False):
            vis_mask[o["mask"] > 0] = 255
    vis_mask = cv2.cvtColor(vis_mask, cv2.COLOR_GRAY2BGR)

    # --------------------------------------------------------
    # Compose
    # --------------------------------------------------------
    size = (512, 512)
    row1 = cv2.hconcat([
        prep(front_img, "Front RGB", size),
        prep(front_mask_color, f"Mask r={args.r}", size),
        prep(flip_front(front_bbox), "Front + BBox", size),
    ])
    row2 = cv2.hconcat([
        prep(bev_gt, "GT BEV", size),
        prep(bev_vis, "BEV Visible", size),
        prep(vis_mask, "Visibility Mask", size),
    ])

    grid = cv2.vconcat([row1, row2])

    cv2.imshow("Visibility Debug", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
