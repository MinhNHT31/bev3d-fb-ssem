#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visibility_debug.py
===================

PURE DEBUG script for utils.visibility.py (VISIBILITY BY CAMERA)

STRICT RULES
------------
- utils.visibility.py is the ONLY source of truth
- NO recompute occlusion
- NO FOV filtering
- NO geometry decisions
- ONLY visualize results returned by visibility.py

PURPOSE
-------
- Verify per-camera visibility correctness
- Overlay projected object masks onto each camera
- Show object IDs ONLY where that camera sees the object
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# ============================================================
# Path setup
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# ============================================================
# Imports (NO LOGIC duplication)
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
from utils.visibility import (
    compute_visible_bev_and_flags,
    project_cuboid_to_mask,
    load_camera_visibility_mask,
)

# ============================================================
# Deterministic color palette
# ============================================================
PALETTE = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
]

def color_for_id(oid: int) -> tuple[int, int, int]:
    return PALETTE[int(oid) % len(PALETTE)]


# ============================================================
# Flip helpers (MUST match training pipeline)
# ============================================================
def flip_for_projection(img: np.ndarray, view: str) -> np.ndarray:
    if view in ("front", "rear"):
        img = cv2.flip(img, 1)
        img = cv2.flip(img, 0)
    return img


def unflip_for_display(img: np.ndarray, view: str) -> np.ndarray:
    if view in ("front", "rear"):
        img = cv2.flip(img, 1)
        img = cv2.flip(img, 0)
    return img


# ============================================================
# Drawing helpers
# ============================================================
def draw_id(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    color=(255, 255, 255),
):
    x, y = pos
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x - 2, y - h - 4), (x + w + 2, y + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ============================================================
# PURE visualization (VISIBILITY BY CAMERA)
# ============================================================
def overlay_visible_masks_by_cam(
    img_bgr: np.ndarray,
    view: str,
    cuboids: List[Dict],
    visible_by_cam: Dict[str, Dict[int, bool]],
    extrinsic_w2c: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    xi: float,
    *,
    segments: int,
    min_valid_points: int,
    alpha: float,
) -> np.ndarray:
    """
    Draw ONLY objects that visibility.py says are visible
    FROM THIS CAMERA.
    """
    H, W = img_bgr.shape[:2]
    work = flip_for_projection(img_bgr.copy(), view)

    cam_mask = load_camera_visibility_mask(view)
    if cam_mask is None:
        cam_mask = np.ones((H, W), dtype=bool)

    overlay = np.zeros_like(work)

    cam_visibility = visible_by_cam.get(view, {})

    for cub in cuboids:
        oid = int(cub.get("local_id", -1))
        if not cam_visibility.get(oid, False):
            continue

        mask = project_cuboid_to_mask(
            corners_world=np.asarray(cub["corners"], dtype=np.float64),
            extrinsic_w2c=extrinsic_w2c,
            K=K,
            D=D,
            xi=xi,
            image_shape=(H, W),
            segments=segments,
            min_valid_points=min_valid_points,
            cam_mask=cam_mask,
        )

        if not mask.any():
            continue

        overlay[mask] = color_for_id(oid)

        ys, xs = np.where(mask)
        cx, cy = int(xs.mean()), int(ys.mean())
        draw_id(overlay, f"ID:{oid}", (cx, cy))

    blended = cv2.addWeighted(work, 1.0, overlay, alpha, 0.0)
    return unflip_for_display(blended, view)


# ============================================================
# Panel helpers
# ============================================================
def panel(img: np.ndarray, title: str, size: int) -> np.ndarray:
    out = cv2.resize(img, (size, size))
    cv2.putText(out, title, (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return out


# ============================================================
# Main
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--panel-size", type=int, default=512)

    ap.add_argument("--resolution", type=float, default=100 / (6 * 400))
    ap.add_argument("--min-area", type=int, default=20)
    ap.add_argument("--offset", type=float, default=33.0)
    ap.add_argument("--yshift", type=float, default=-0.377)

    ap.add_argument("--visible-ratio-thresh", type=float, default=0.01)
    ap.add_argument("--min-pixels", type=int, default=20)

    ap.add_argument("--segments", type=int, default=200)
    ap.add_argument("--min-valid-points", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.6)

    args = ap.parse_args()
    root = Path(args.dataset_root)
    fid = args.id
    S = args.panel_size

    # --------------------------------------------------------
    # BEV → cuboids
    # --------------------------------------------------------
    bev_rgb = load_seg(root / "seg/bev" / f"{fid}.png")
    depth = load_depth(root / "depth" / f"{fid}.png")
    camcfg = root / "cameraconfig" / f"{fid}.txt"
    cam_h = load_camera_bev_height(str(camcfg))

    obj_masks = segment_objects(bev_rgb, min_area=args.min_area)
    boxes = get_2d_bounding_boxes(obj_masks)
    height_map = compute_height_map(depth, cam_h)

    cuboids = build_cuboids_from_2d_boxes(
        boxes,
        height_map,
        resolution=args.resolution,
        offset=args.offset,
        yshift=args.yshift,
    )

    # --------------------------------------------------------
    # Camera params
    # --------------------------------------------------------
    calib_root = root.parents[1]
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
    # VISIBILITY (SOURCE OF TRUTH)
    # --------------------------------------------------------
    bev_visible, visible_by_cam, _visible_any = compute_visible_bev_and_flags(
        bev_seg_rgb=bev_rgb,
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

    # --------------------------------------------------------
    # Render per-camera
    # --------------------------------------------------------
    rendered = {}
    for view, cam_key in cam_name_map.items():
        img = cam_images.get(view)
        if img is None:
            rendered[view] = np.zeros((S, S, 3), np.uint8)
            continue

        rendered[view] = overlay_visible_masks_by_cam(
            img_bgr=img,
            view=view,
            cuboids=cuboids,
            visible_by_cam=visible_by_cam,
            extrinsic_w2c=extrinsics[cam_key],
            K=K, D=D, xi=xi,
            segments=args.segments,
            min_valid_points=args.min_valid_points,
            alpha=args.alpha,
        )

    # --------------------------------------------------------
    # BEV panels
    # --------------------------------------------------------
    vis_bin = (np.any(bev_visible != 0, axis=2)).astype(np.uint8) * 255
    vis_bin = cv2.cvtColor(vis_bin, cv2.COLOR_GRAY2BGR)
    bev_bgr = cv2.cvtColor(bev_rgb, cv2.COLOR_RGB2BGR)

    grid = cv2.vconcat([
        cv2.hconcat([
            panel(rendered["left"], "Left", S),
            panel(rendered["front"], "Front", S),
            panel(rendered["right"], "Right", S),
        ]),
        cv2.hconcat([
            panel(vis_bin, "VISI (BEV)", S),
            panel(rendered["rear"], "Rear", S),
            panel(bev_bgr, "GT BEV", S),
        ]),
    ])

    cv2.imshow("Visibility DEBUG (per-camera)", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
