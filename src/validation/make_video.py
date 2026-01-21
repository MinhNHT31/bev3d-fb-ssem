#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_video.py

Create a composite visualization video from FB-SSEM dataset frames.

This script is a BATCH version of project_mei.py.
All geometry and projection logic MUST stay identical.

The only difference:
    - Loop over multiple frames
    - Write results to a video instead of matplotlib

IMPORTANT:
- Geometry logic is NOT modified
- Flip logic is kept EXACTLY as in project_mei.py
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import sys
import os
from tqdm import tqdm

# ============================================================
# Path setup
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
from utils.bbox3d import draw_cuboids_curved


# ============================================================
# Display helper (IDENTICAL to project_mei.py)
# ============================================================
def flip_for_display(img, view):
    if view in ["front", "rear"]:
        img = cv2.flip(img, 1)
        img = cv2.flip(img, 0)
    return img


def draw_text(img, text):
    cv2.putText(
        img, text, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1,
        (0, 255, 0), 2, cv2.LINE_AA
    )


# ============================================================
# Process single frame (COPY from project_mei.py)
# ============================================================
def process_frame(fid, root, args, K, D, xi, extrinsics, target_size):
    bev_path = root / "seg" / "bev" / f"{fid}.png"
    depth_path = root / "depth" / f"{fid}.png"
    camcfg_path = root / "cameraconfig" / f"{fid}.txt"

    if not bev_path.exists():
        return None

    # ---------- BEV → 3D ----------
    bev_seg = load_seg(bev_path)
    depth = load_depth(depth_path)
    bev_cam_h = load_camera_bev_height(str(camcfg_path))

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

    cam_map = {
        "left":  "Main Camera-left",
        "front": "Main Camera-front",
        "right": "Main Camera-right",
        "rear":  "Main Camera-rear",
    }

    imgs = {}

    for view, cam_name in cam_map.items():
        img_path = root / "rgb" / view / f"{fid}.png"
        if not img_path.exists():
            img = np.zeros((target_size[1], target_size[0], 3), np.uint8)
        else:
            img = cv2.imread(str(img_path))

            # ===== IMPORTANT: identical flip logic =====
            img = flip_for_display(img, view)

            ext = extrinsics.get(cam_name, None)
            if ext is not None:
                img = draw_cuboids_curved(img, cuboids, ext, K, D, xi)

            img = flip_for_display(img, view)
            img = cv2.resize(img, target_size)

        draw_text(img, view.capitalize())
        imgs[view] = img

    # ---------- Depth ----------
    depth_vis = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
    depth_vis = cv2.resize(depth_vis, target_size)
    draw_text(depth_vis, "Depth")

    # ---------- BEV ----------
    bev_vis = cv2.imread(str(bev_path))
    bev_vis = cv2.resize(bev_vis, target_size)
    draw_text(bev_vis, f"BEV (ID: {fid})")

    # ---------- Compose ----------
    row1 = cv2.hconcat([imgs["left"], imgs["front"], imgs["right"]])
    row2 = cv2.hconcat([depth_vis, imgs["rear"], bev_vis])
    grid = cv2.vconcat([row1, row2])

    return grid


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output", default="output_viz.mp4")
    ap.add_argument("--fps", type=int, default=10)

    ap.add_argument("--resolution", type=float, default=100/(6*400))
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--offset", type=float, default=33)
    ap.add_argument("--yshift", type=float, default=-0.377)
    args = ap.parse_args()

    root = Path(args.dataset_root)

    # ---------- Frame list ----------
    bev_dir = root / "seg" / "bev"
    frame_ids = sorted([p.stem for p in bev_dir.glob("*.png")], key=lambda x: int(x))

    if not frame_ids:
        print("❌ No frames found")
        return

    # ---------- Calibration ----------
    calib_root = root.parents[1]
    K, D, xi = load_intrinsics(calib_root / "camera_intrinsics.yml")
    extrinsics = load_extrinsics(calib_root / "camera_positions_for_extrinsics.txt")

    # ---------- Video ----------
    target_size = (640, 480)
    output_path = Path("videos") / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None

    for fid in tqdm(frame_ids, desc="Rendering video"):
        frame = process_frame(fid, root, args, K, D, xi, extrinsics, target_size)
        if frame is None:
            continue

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.fps,
                (w, h)
            )
            print(f"Video resolution: {w}x{h}")

        writer.write(frame)

    if writer:
        writer.release()
        print(f"\n✅ Video saved to {output_path}")


if __name__ == "__main__":
    main()
