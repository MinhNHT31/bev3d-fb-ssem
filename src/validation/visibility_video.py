#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visibility_video.py
===================

VIDEO DEBUG for utils.visibility.py (VISIBILITY BY CAMERA)

Fixes:
- Correct frame ordering (numeric sort by frame id)
- Realtime preview with --show (ESC/q to stop)

STRICT RULES
------------
- utils.visibility.py is the ONLY source of truth
- NO recompute occlusion
- NO FOV filtering
- NO geometry decisions
- ONLY visualize results returned by visibility.py

CUSTOM VIS (THIS VERSION)
-------------------------
- Draw 3D cuboids on each camera using utils.bbox3d.draw_cuboids_curved
- Draw ALL objects (that can be projected) in each camera:
    GREEN = visible_by_cam[view][oid] == True
    RED   = visible_by_cam[view][oid] == False
- Ego (oid == 100) is NOT drawn on any camera.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
)
from utils.bbox3d import draw_cuboids_curved  # uses cam2image internally

# ============================================================
# Color palette / constants
# ============================================================
COLOR_VISIBLE = [0.0, 1.0, 0.0]     # green in RGB floats
COLOR_INVISIBLE = [1.0, 0.0, 0.0]   # red in RGB floats

EGO_ID = 100

# ============================================================
# Flip helpers
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
# Overlay per camera (PURE VISUALIZATION)
# ============================================================
def overlay_visible_masks_by_cam(
    img_bgr: np.ndarray,
    view: str,
    cuboids: List[Dict],
    visible_by_cam: Dict[str, Dict[int, bool]],
    extrinsic_w2c: np.ndarray,
    K, D, xi,
    *,
    segments: int,
    min_valid_points: int,  # kept for outline compatibility (unused)
    alpha: float,           # kept for outline compatibility (unused)
) -> np.ndarray:
    """
    PURE VISUALIZATION ONLY.

    - Draw ALL cuboids (projectable edges will appear; non-projectable edges are skipped by draw_cuboids_curved)
    - Color rule:
        GREEN = visible_by_cam[view][oid] == True
        RED   = visible_by_cam[view][oid] == False
    - Ego (oid == 100) is NEVER drawn.
    """

    H, W = img_bgr.shape[:2]
    work = flip_for_projection(img_bgr.copy(), view)

    cam_vis = visible_by_cam.get(view, {})

    # Build list for bbox3d.draw_cuboids_curved
    cuboids_to_draw: List[Dict] = []

    for cub in cuboids:
        oid = int(cub.get("local_id", -1))
        if oid < 0:
            continue

        # 🚫 Never draw ego on any camera
        if oid == EGO_ID:
            continue

        corners = np.asarray(cub.get("corners", None), dtype=np.float64)
        if corners is None or corners.shape != (8, 3):
            continue

        color = COLOR_VISIBLE if cam_vis.get(oid, False) else COLOR_INVISIBLE

        cuboids_to_draw.append(
            {
                "corners": corners,
                "color": color,  # RGB floats [0,1]
            }
        )

    # Draw curved cuboid edges using MEI projection
    # Note: draw_cuboids_curved already skips edges with insufficient valid projections.
    work = draw_cuboids_curved(
        img=work,
        cuboids_list=cuboids_to_draw,
        Matrix=extrinsic_w2c,
        K=K,
        D=D,
        xi=xi,
        segments=int(segments),
    )

    return unflip_for_display(work, view)


# ============================================================
# Frame processing
# ============================================================
def process_frame(
    fid: str,
    root: Path,
    args,
    K, D, xi,
    extrinsics: Dict[str, np.ndarray],
    panel_size: int,
) -> Optional[np.ndarray]:
    bev_path = root / "seg/bev" / f"{fid}.png"
    depth_path = root / "depth" / f"{fid}.png"
    camcfg_path = root / "cameraconfig" / f"{fid}.txt"

    if not bev_path.exists():
        return None

    bev_rgb = load_seg(bev_path)
    depth = load_depth(depth_path)
    cam_h = load_camera_bev_height(str(camcfg_path))

    obj_masks = segment_objects(bev_rgb, min_area=int(args.min_area))
    boxes = get_2d_bounding_boxes(obj_masks)
    height_map = compute_height_map(depth, cam_h)

    cuboids = build_cuboids_from_2d_boxes(
        boxes,
        height_map,
        resolution=float(args.resolution),
        offset=float(args.offset),
        yshift=float(args.yshift),
    )

    cam_name_map = {
        "front": "Main Camera-front",
        "left":  "Main Camera-left",
        "right": "Main Camera-right",
        "rear":  "Main Camera-rear",
    }

    cam_images: Dict[str, Optional[np.ndarray]] = {
        v: cv2.imread(str(root / "rgb" / v / f"{fid}.png"))
        for v in cam_name_map
    }

    # ========================================================
    # SOURCE OF TRUTH (NO duplicated logic)
    # ========================================================
    bev_visible, visible_by_cam, _ = compute_visible_bev_and_flags(
        bev_seg_rgb=bev_rgb,
        obj_masks=obj_masks,
        cuboids=cuboids,
        cam_images=cam_images,
        extrinsics=extrinsics,
        K=K, D=D, xi=xi,
        cam_name_map=cam_name_map,
        visible_ratio_thresh=float(args.visible_ratio_thresh),
        min_pixels=int(args.min_pixels),
    )

    rendered: Dict[str, np.ndarray] = {}
    for view, cam_key in cam_name_map.items():
        img = cam_images.get(view)
        if img is None:
            rendered[view] = np.zeros((panel_size, panel_size, 3), np.uint8)
            continue

        rendered[view] = overlay_visible_masks_by_cam(
            img_bgr=img,
            view=view,
            cuboids=cuboids,
            visible_by_cam=visible_by_cam,
            extrinsic_w2c=extrinsics[cam_key],
            K=K, D=D, xi=xi,
            segments=int(args.segments),
            min_valid_points=int(args.min_valid_points),
            alpha=float(args.alpha),
        )

    # VISI (BEV) panel
    vis_bin = (np.any(bev_visible != 0, axis=2)).astype(np.uint8) * 255
    vis_bin = cv2.cvtColor(vis_bin, cv2.COLOR_GRAY2BGR)

    # BEV panel
    bev_bgr = cv2.cvtColor(bev_rgb, cv2.COLOR_RGB2BGR)

    def panel(img, title):
        out = cv2.resize(img, (panel_size, panel_size))
        cv2.putText(out, title, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return out

    # KEEP ORIGINAL LAYOUT (2x3)
    grid = cv2.vconcat([
        cv2.hconcat([
            panel(rendered["left"], "Left"),
            panel(rendered["front"], "Front"),
            panel(rendered["right"], "Right"),
        ]),
        cv2.hconcat([
            panel(vis_bin, "VISI (BEV)"),
            panel(rendered["rear"], "Rear"),
            panel(bev_bgr, f"GT BEV (id={fid})"),
        ]),
    ])

    return grid


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output", default="visibility_debug.mp4")
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--panel-size", type=int, default=512)

    ap.add_argument("--resolution", type=float, default=100 / (6 * 400))
    ap.add_argument("--min-area", type=int, default=15)
    ap.add_argument("--offset", type=float, default=33.0)
    ap.add_argument("--yshift", type=float, default=-0.377)

    ap.add_argument("--visible-ratio-thresh", type=float, default=0.01)
    ap.add_argument("--min-pixels", type=int, default=20)

    ap.add_argument("--segments", type=int, default=200)
    ap.add_argument("--min-valid-points", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.6)

    ap.add_argument("--show", action="store_true",
                    help="Realtime preview (ESC or q to stop early)")

    args = ap.parse_args()
    root = Path(args.dataset_root)

    # -------- FIX: numeric frame ordering --------
    bev_files = list((root / "seg/bev").glob("*.png"))
    if not bev_files:
        raise FileNotFoundError(f"No frames found in {root/'seg/bev'}")

    try:
        frame_ids = sorted([p.stem for p in bev_files], key=lambda s: int(s))
    except ValueError:
        frame_ids = sorted([p.stem for p in bev_files])

    # Camera parameters
    calib_root = root.parents[1]
    K, D, xi = load_intrinsics(calib_root / "camera_intrinsics.yml")
    extrinsics = load_extrinsics(calib_root / "camera_positions_for_extrinsics.txt")

    out = Path("videos") / args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = None

    for fid in tqdm(frame_ids, desc="Rendering visibility video"):
        frame = process_frame(fid, root, args, K, D, xi, extrinsics, args.panel_size)
        if frame is None:
            continue

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(out),
                cv2.VideoWriter_fourcc(*"mp4v"),
                int(args.fps),
                (w, h),
            )

        writer.write(frame)

        if args.show:
            cv2.imshow("Visibility DEBUG (per-camera)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("\n⏹️ Stopped by user")
                break

    if writer is not None:
        writer.release()
        print(f"\n✅ Video saved to {out}")

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
