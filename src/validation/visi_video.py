#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visi_video.py

Create a composite visualization video with BEV visibility overlay
and REAL-TIME preview.

Layout:
--------------------------------------------------
Row 1:  Left | Front | Right
Row 2:  VISI | Rear  | BEV

Pipeline:
--------------------------------------------------
BEV seg + depth
    → annotation.py (cuboids)
    → visibility.py (FAR → NEAR, curved projection)
    → visualization video

Controls (when --show is enabled):
--------------------------------------------------
- ESC or 'q' : stop rendering early
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

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
from utils.visibility import compute_visible_bev_and_flags


# ============================================================
# Display helpers
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
# Process one frame
# ============================================================
def process_frame(fid, root, args, K, D, xi, extrinsics, target_size):
    bev_path = root / "seg" / "bev" / f"{fid}.png"
    depth_path = root / "depth" / f"{fid}.png"
    camcfg_path = root / "cameraconfig" / f"{fid}.txt"

    if not bev_path.exists():
        return None

    # --------------------------------------------------
    # BEV → 3D cuboids
    # --------------------------------------------------
    bev_seg = load_seg(str(bev_path))
    depth = load_depth(str(depth_path))
    bev_cam_h = load_camera_bev_height(str(camcfg_path)) if camcfg_path.exists() else None

    obj_masks = segment_objects(bev_seg, min_area=args.min_area)
    boxes_2d = get_2d_bounding_boxes(obj_masks)
    height_map = compute_height_map(depth, bev_cam_h)

    cuboids = build_cuboids_from_2d_boxes(
        boxes_2d,
        height_map,
        resolution=args.resolution,
        offset=args.offset,
        yshift=args.yshift,
    )

    # --------------------------------------------------
    # Visibility (CLEAN API)
    # --------------------------------------------------
    cam_name_map = {
        "front": "Main Camera-front",
        "left":  "Main Camera-left",
        "right": "Main Camera-right",
        "rear":  "Main Camera-rear",
    }

    cam_images = {}
    for view in cam_name_map:
        p = root / "rgb" / view / f"{fid}.png"
        cam_images[view] = cv2.imread(str(p)) if p.exists() else None

    _, visible_by_id, _ = compute_visible_bev_and_flags(
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

    # --------------------------------------------------
    # Camera views (with cuboids)
    # --------------------------------------------------
    imgs = {}
    for view, cam_key in cam_name_map.items():
        img_path = root / "rgb" / view / f"{fid}.png"

        if not img_path.exists():
            img = np.zeros((target_size[1], target_size[0], 3), np.uint8)
        else:
            img = cv2.imread(str(img_path))
            img = flip_for_display(img, view)

            ext = extrinsics.get(cam_key)
            if ext is not None:
                img = draw_cuboids_curved(img, cuboids, ext, K, D, xi)

            img = flip_for_display(img, view)
            img = cv2.resize(img, target_size)

        draw_text(img, view.capitalize())
        imgs[view] = img

    # --------------------------------------------------
    # Visibility mask (BEV space)
    # --------------------------------------------------
    vis_mask = np.zeros(bev_seg.shape[:2], np.uint8)
    for i, o in enumerate(obj_masks):
        if visible_by_id.get(i, False):
            vis_mask[o["mask"] > 0] = 255

    vis_img = cv2.cvtColor(vis_mask, cv2.COLOR_GRAY2BGR)
    vis_img = cv2.resize(vis_img, target_size)
    draw_text(vis_img, "Visibility")

    # --------------------------------------------------
    # BEV panel
    # --------------------------------------------------
    bev_viz = cv2.imread(str(bev_path))
    if bev_viz is None:
        bev_viz = np.zeros((target_size[1], target_size[0], 3), np.uint8)
    else:
        bev_viz = cv2.resize(bev_viz, target_size)

    draw_text(bev_viz, f"BEV (ID: {fid})")

    # --------------------------------------------------
    # Compose grid
    # --------------------------------------------------
    row1 = cv2.hconcat([imgs["left"], imgs["front"], imgs["right"]])
    row2 = cv2.hconcat([vis_img, imgs["rear"], bev_viz])
    grid = cv2.vconcat([row1, row2])

    return grid


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output", default="visibility_video.mp4")
    ap.add_argument("--fps", type=int, default=10)

    ap.add_argument("--resolution", type=float, default=100 / (6 * 400))
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--offset", type=float, default=36)
    ap.add_argument("--yshift", type=float, default=-0.4)

    ap.add_argument("--visible-ratio-thresh", type=float, default=0.33)
    ap.add_argument("--min-pixels", type=int, default=20)

    # LIVE PREVIEW
    ap.add_argument(
        "--show",
        action="store_true",
        help="Show video frames in real-time while rendering"
    )

    args = ap.parse_args()
    root = Path(args.dataset_root)

    # --------------------------------------------------
    # Frame list
    # --------------------------------------------------
    bev_dir = root / "seg" / "bev"
    bev_files = list(bev_dir.glob("*.png"))
    if not bev_files:
        print("❌ No BEV frames found")
        return

    try:
        frame_ids = sorted([p.stem for p in bev_files], key=lambda x: int(x))
    except ValueError:
        frame_ids = sorted([p.stem for p in bev_files])

    # --------------------------------------------------
    # Camera parameters
    # --------------------------------------------------
    calib_root = root.parents[1] / "CameraCalibrationParameters"
    K, D, xi = load_intrinsics(calib_root / "camera_intrinsics.yml")
    extrinsics = load_extrinsics(calib_root / "camera_positions_for_extrinsics.txt")

    # --------------------------------------------------
    # Video writer
    # --------------------------------------------------
    target_size = (640, 480)
    out_path = Path("videos") / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None

    for fid in tqdm(frame_ids, desc="Rendering visibility video"):
        frame = process_frame(fid, root, args, K, D, xi, extrinsics, target_size)
        if frame is None:
            continue

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.fps,
                (w, h),
            )
            print(f"Video resolution: {w}x{h}")

        writer.write(frame)

        # ----------------------------------------------
        # LIVE PREVIEW
        # ----------------------------------------------
        if args.show:
            cv2.imshow("Visibility Video (Live)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord("q")]:
                print("\n⏹️ Stopped by user")
                break

    if writer:
        writer.release()
        print(f"\n✅ Visibility video saved to {out_path}")
    else:
        print("❌ Video not created")

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
