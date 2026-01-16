#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visi_video.py

Composite visualization video for FB-SSEM visibility (NEW visibility.py).

Layout:
--------------------------------------------------
Row 1:  Left | Front | Right
Row 2:  VISI | Rear  | BEV

Pipeline (per frame):
--------------------------------------------------
BEV seg + depth
  -> annotation.py (build cuboids)
  -> visibility.py (NEW occlusion: NEAR->FAR + union prev, multi-cam any)
  -> visualize:
        - draw cuboids on 4 RGB camera images
            * GREEN = visible (per visibility.py)
            * RED   = occluded (per visibility.py)
        - VISI panel: BEV-space mask of visible objects (white)
        - BEV panel: original BEV seg (correct colors)

Controls (when --show enabled):
--------------------------------------------------
- ESC or 'q' : stop rendering early
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
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
def flip_for_display(img_bgr: np.ndarray, view: str) -> np.ndarray:
    """
    Keep consistent with your project's display convention.
    (Same as your previous scripts.)
    """
    if view in ["front", "rear"]:
        img_bgr = cv2.flip(img_bgr, 1)
        img_bgr = cv2.flip(img_bgr, 0)
    return img_bgr


def draw_text(img_bgr: np.ndarray, text: str) -> None:
    cv2.putText(
        img_bgr,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


# ============================================================
# Build cuboids list with colors based on visibility
# ============================================================
def colorize_cuboids(cuboids, visible_by_id: Dict[int, bool]):
    """
    Return a NEW list of cuboids dicts where each cuboid has a color:
      GREEN (0,1,0) visible
      RED   (1,0,0) occluded
    draw_cuboids_curved expects dict keys: "corners", optional "color" (RGB float [0..1])
    """
    colored = []
    for i, cub in enumerate(cuboids):
        is_vis = bool(visible_by_id.get(i, False))
        colored.append(
            {
                "corners": cub["corners"],
                "color": (0.0, 1.0, 0.0) if is_vis else (1.0, 0.0, 0.0),
            }
        )
    return colored


# ============================================================
# Process one frame
# ============================================================
def process_frame(fid: str, root: Path, args, K, D, xi, extrinsics, target_size):
    bev_path = root / "seg" / "bev" / f"{fid}.png"
    depth_path = root / "depth" / f"{fid}.png"
    camcfg_path = root / "cameraconfig" / f"{fid}.txt"

    if not bev_path.exists():
        return None

    # --------------------------------------------------
    # 1) BEV seg + depth
    # --------------------------------------------------
    bev_seg_rgb = load_seg(str(bev_path))          # RGB
    depth = load_depth(str(depth_path))           # float32 [0..1]
    bev_cam_h = load_camera_bev_height(str(camcfg_path)) if camcfg_path.exists() else None

    # --------------------------------------------------
    # 2) BEV -> objects -> cuboids
    # --------------------------------------------------
    obj_masks = segment_objects(bev_seg_rgb, min_area=args.min_area)
    boxes_2d = get_2d_bounding_boxes(obj_masks)
    height_map = compute_height_map(depth, bev_cam_h)

    cuboids = build_cuboids_from_2d_boxes(
        boxes_2d=boxes_2d,
        height_map=height_map,
        resolution=args.resolution,
        offset=args.offset,
        yshift=args.yshift,
    )

    # --------------------------------------------------
    # 3) Visibility (NEW visibility.py) on 4 RGB camera images
    # --------------------------------------------------
    cam_name_map = {
        "front": "Main Camera-front",
        "left":  "Main Camera-left",
        "right": "Main Camera-right",
        "rear":  "Main Camera-rear",
    }

    cam_images: Dict[str, Optional[np.ndarray]] = {}
    for view in cam_name_map:
        p = root / "rgb" / view / f"{fid}.png"
        cam_images[view] = cv2.imread(str(p)) if p.exists() else None

    # visibility.py returns: (bev_visible_rgb, visible_by_id, best_ratio)
    bev_visible_rgb, visible_by_id, best_ratio = compute_visible_bev_and_flags(
        bev_seg_rgb=bev_seg_rgb,
        obj_masks=obj_masks,
        cuboids=cuboids,
        cam_images=cam_images,
        extrinsics=extrinsics,
        K=K, D=D, xi=xi,
        cam_name_map=cam_name_map,
        visible_ratio_thresh=args.visible_ratio_thresh,
        min_pixels=args.min_pixels,
    )

    # Prepare colored cuboids for drawing
    cuboids_colored = colorize_cuboids(cuboids, visible_by_id)

    # --------------------------------------------------
    # 4) Camera views (draw colored cuboids)
    # --------------------------------------------------
    imgs = {}
    for view, cam_key in cam_name_map.items():
        img_path = root / "rgb" / view / f"{fid}.png"

        if not img_path.exists():
            img = np.zeros((target_size[1], target_size[0], 3), np.uint8)
        else:
            img = cv2.imread(str(img_path))  # BGR
            img = flip_for_display(img, view)

            ext = extrinsics.get(cam_key)
            if ext is not None and len(cuboids_colored) > 0:
                img = draw_cuboids_curved(img, cuboids_colored, ext, K, D, xi)

            img = flip_for_display(img, view)  # keep your convention
            img = cv2.resize(img, target_size)

        draw_text(img, view.capitalize())
        imgs[view] = img

    # --------------------------------------------------
    # 5) VISI panel (BEV-space visibility mask)
    # --------------------------------------------------
    vis_mask = np.zeros(bev_seg_rgb.shape[:2], np.uint8)
    n = min(len(obj_masks), len(cuboids))
    for i in range(n):
        if visible_by_id.get(i, False):
            vis_mask[obj_masks[i]["mask"] > 0] = 255

    vis_img = cv2.cvtColor(vis_mask, cv2.COLOR_GRAY2BGR)
    vis_img = cv2.resize(vis_img, target_size)
    draw_text(vis_img, "Visibility (BEV)")

    # --------------------------------------------------
    # 6) BEV panel (keep correct colors)
    #    BEV is RGB, OpenCV expects BGR for display
    # --------------------------------------------------
    if args.show_bev_visible:
        bev_panel_rgb = bev_visible_rgb
        title = f"BEV Visible (ID: {fid})"
    else:
        bev_panel_rgb = bev_seg_rgb
        title = f"BEV (ID: {fid})"

    bev_panel_bgr = cv2.cvtColor(bev_panel_rgb, cv2.COLOR_RGB2BGR)
    bev_panel_bgr = cv2.resize(bev_panel_bgr, target_size)
    draw_text(bev_panel_bgr, title)

    # --------------------------------------------------
    # 7) Compose grid
    # --------------------------------------------------
    row1 = cv2.hconcat([imgs["left"], imgs["front"], imgs["right"]])
    row2 = cv2.hconcat([vis_img, imgs["rear"], bev_panel_bgr])
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

    # annotation.py params
    ap.add_argument("--resolution", type=float, default=100 / (6 * 400))
    ap.add_argument("--min-area", type=int, default=20)
    ap.add_argument("--offset", type=float, default=33.0)
    ap.add_argument("--yshift", type=float, default=-0.337)

    # visibility.py params
    ap.add_argument("--visible-ratio-thresh", type=float, default=0.05)
    ap.add_argument("--min-pixels", type=int, default=5)

    # visualization
    ap.add_argument("--show", action="store_true", help="Live preview while rendering (ESC/q to stop)")
    ap.add_argument("--show-bev-visible", action="store_true", help="Show bev_visible instead of original BEV in BEV panel")
    ap.add_argument("--target-w", type=int, default=640)
    ap.add_argument("--target-h", type=int, default=480)

    args = ap.parse_args()
    root = Path(args.dataset_root)

    # --------------------------------------------------
    # Frame list
    # --------------------------------------------------
    bev_dir = root / "seg" / "bev"
    bev_files = list(bev_dir.glob("*.png"))
    if not bev_files:
        print(f"❌ No BEV frames found in {bev_dir}")
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
    target_size = (args.target_w, args.target_h)
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

        if args.show:
            cv2.imshow("Visibility Video (Live)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord("q")]:
                print("\n⏹️ Stopped by user")
                break

    if writer is not None:
        writer.release()
        print(f"\n✅ Visibility video saved to {out_path}")
    else:
        print("❌ Video not created")

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
