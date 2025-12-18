#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
demo_sam_raw_vs_label.py

Figure 1: RAW RGB + 3D Cuboids (geometry debug)
Figure 2: SAM2 Pseudo Segmentation + Cuboids (label debug)

Layout: SAME as project_mei.py (2x3)
Calibration-safe:
- Front & Rear rotated BEFORE SAM
- Flip back ONLY for visualization
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from ultralytics import SAM

# =============================================================================
# Path setup
# =============================================================================
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils.bbox3d import draw_cuboids_curved
from utils.camera import load_intrinsics, load_extrinsics, load_camera_bev_height
from utils.projects import cam2image
from utils.pipeline import (
    load_depth,
    load_seg,
    compute_height_map,
    segment_objects,
    get_2d_bounding_boxes,
    get_3d_bounding_boxes,
)

# =============================================================================
# Image loading (CALIBRATION FRAME)
# =============================================================================
def load_rgb_for_model(path: Path, view: str) -> Optional[np.ndarray]:
    if not path.exists():
        return None

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None

    # Rotate front & rear into calibration frame
    if view in ["front", "rear"]:
        bgr = cv2.flip(bgr, 1)
        bgr = cv2.flip(bgr, 0)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def depth_to_rgb(depth_path: Path) -> np.ndarray:
    d = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    if d is None:
        return np.zeros((240, 320, 3), dtype=np.uint8)
    d8 = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.cvtColor(d8, cv2.COLOR_GRAY2RGB)

# =============================================================================
# Geometry helpers
# =============================================================================
def project_cuboid_polygon(corners, Extr, K, D, xi, H, W):
    uv, mask = cam2image(corners, Extr, K, D, xi)
    uv = uv[mask]
    if uv.shape[0] < 3:
        return None

    uv[:, 0] = np.clip(uv[:, 0], 0, W - 1)
    uv[:, 1] = np.clip(uv[:, 1], 0, H - 1)
    return uv.astype(np.int32)


def polygon_to_mask(uv, H, W):
    hull = cv2.convexHull(uv.reshape(-1, 1, 2))
    if hull is None or len(hull) < 3:
        return None
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 255)
    return mask


def iou_mask(a, b):
    A = a > 0
    B = b > 0
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    return float(inter) / float(union) if union > 0 else 0.0

# =============================================================================
# SAM helpers
# =============================================================================
def run_sam(model: SAM, rgb: np.ndarray) -> List[np.ndarray]:
    with torch.no_grad():
        result = model(rgb)[0]

    if result.masks is None:
        return []

    data = result.masks.data.cpu().numpy()
    return [(m > 0.5).astype(np.uint8) * 255 for m in data]

# =============================================================================
# Build pseudo semantic segmentation
# =============================================================================
def build_pseudo_seg(
    rgb, sam_masks, cuboids, bev_objs,
    Extr, K, D, xi, iou_thresh
):
    H, W = rgb.shape[:2]
    out = np.zeros((H, W, 3), dtype=np.uint8)

    for i, cub in enumerate(cuboids):
        uv = project_cuboid_polygon(
            np.asarray(cub["corners"], np.float32),
            Extr, K, D, xi, H, W
        )
        if uv is None:
            continue

        poly = polygon_to_mask(uv, H, W)
        if poly is None:
            continue

        color01 = bev_objs[i].get("color", [1, 1, 1])
        color = tuple(int(c * 255) for c in color01)

        best_iou, best_mask = 0.0, None
        for sm in sam_masks:
            score = iou_mask(poly, sm)
            if score > best_iou:
                best_iou, best_mask = score, sm

        if best_mask is not None and best_iou >= iou_thresh:
            out[best_mask > 0] = color
        else:
            out[poly > 0] = color

    return out

# =============================================================================
# Cuboid overlay
# =============================================================================
def overlay_cuboids(rgb, cuboids, Extr, K, D, xi, flip_back):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    img = draw_cuboids_curved(img, cuboids, Extr, K, D, xi)

    if flip_back:
        img = cv2.flip(img, 1)
        img = cv2.flip(img, 0)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--sam-weights", default="sam2.1_b.pt")
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--resolution", type=float, default=100/(6*400))
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--offset", type=float, default=33)
    ap.add_argument("--yshift", type=float, default=-0.3)
    args = ap.parse_args()

    root = Path(args.dataset_root)
    sid = str(args.id)

    # -------------------------------------------------------------------------
    # BEV → Cuboids
    # -------------------------------------------------------------------------
    bev = load_seg(str(root / "seg" / "bev" / f"{sid}.png"))
    bev_objs = segment_objects(bev, min_area=args.min_area)
    depth = load_depth(str(root / "depth" / f"{sid}.png"))

    cam_h = load_camera_bev_height(str(root / "cameraconfig" / f"{sid}.txt"))
    height_map = compute_height_map(depth, cam_h)
    boxes2d = get_2d_bounding_boxes(bev_objs)
    cuboids = get_3d_bounding_boxes(
        boxes2d, height_map, args.resolution,
        offset=args.offset, yshift=args.yshift
    )

    # -------------------------------------------------------------------------
    # Camera params
    # -------------------------------------------------------------------------
    K, D, xi = load_intrinsics(Path(args.intrinsics))
    extr = load_extrinsics(
        root.parent.parent / "CameraCalibrationParameters" /
        "camera_positions_for_extrinsics.txt"
    )

    cam_map = {
        "left": "Main Camera-left",
        "front": "Main Camera-front",
        "right": "Main Camera-right",
        "rear": "Main Camera-rear",
    }

    # -------------------------------------------------------------------------
    # Load SAM
    # -------------------------------------------------------------------------
    model = SAM(args.sam_weights)
    model.model.to("cuda")

    views = ["left", "front", "right", "rear"]
    raw_fig = {}
    label_fig = {}

    for v in views:
        rgb = load_rgb_for_model(root / "rgb" / v / f"{sid}.png", v)
        Extr = extr[cam_map[v]]

        # ---------- FIGURE 1: RAW + CUBOID ----------
        raw_fig[v] = overlay_cuboids(
            rgb, cuboids, Extr, K, D, xi,
            flip_back=(v in ["front", "rear"])
        )

        # ---------- FIGURE 2: LABEL + CUBOID ----------
        sam_masks = run_sam(model, rgb)
        pseudo = build_pseudo_seg(
            rgb, sam_masks, cuboids, bev_objs,
            Extr, K, D, xi, args.iou_thresh
        )

        label_fig[v] = overlay_cuboids(
            pseudo, cuboids, Extr, K, D, xi,
            flip_back=(v in ["front", "rear"])
        )

    depth_rgb = depth_to_rgb(root / "depth" / f"{sid}.png")

    tiles_raw = [
        ("Left", raw_fig["left"]),
        ("Front", raw_fig["front"]),
        ("Right", raw_fig["right"]),
        ("Depth", depth_rgb),
        ("Rear", raw_fig["rear"]),
        ("BEV", bev),
    ]

    tiles_label = [
        ("Left", label_fig["left"]),
        ("Front", label_fig["front"]),
        ("Right", label_fig["right"]),
        ("Depth", depth_rgb),
        ("Rear", label_fig["rear"]),
        ("BEV", bev),
    ]

    # -------------------------------------------------------------------------
    # FIGURE 1: RAW + CUBOID
    # -------------------------------------------------------------------------
    fig1, axs1 = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle("FIGURE 1 — RAW RGB + 3D CUBOIDS", fontsize=16, color="red")

    for ax, (t, img) in zip(axs1.flatten(), tiles_raw):
        ax.imshow(img)
        ax.set_title(t)
        ax.axis("off")

    plt.tight_layout()

    # -------------------------------------------------------------------------
    # FIGURE 2: LABEL + CUBOID
    # -------------------------------------------------------------------------
    fig2, axs2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle("FIGURE 2 — SAM2 PSEUDO SEGMENTATION + CUBOIDS", fontsize=16, color="red")

    for ax, (t, img) in zip(axs2.flatten(), tiles_label):
        ax.imshow(img)
        ax.set_title(t)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
