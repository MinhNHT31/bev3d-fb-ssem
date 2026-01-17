#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visibility_mask.py
==================

Validate visibility.py by visualizing *VISIBLE OBJECT MASKS* (NOT BBOX).

Layout (2 rows x 3 cols):
--------------------------------------------------
Row 1:  Left (visible masks overlay) | Front (visible masks overlay) | Right (visible masks overlay)
Row 2:  VISI (BEV visible mask)      | Rear (visible masks overlay)  | GT BEV

Key requirements:
- Use visibility.py outputs (visible_by_id) as source of truth.
- For each visible object:
    + project cuboid -> image mask (pixel-level)
    + colorize by local_id (fixed palette by ID)
- NO fisheye_visibility_mask
- ONLY camera mask:
    BLACK (0) = visible
    WHITE (255) = not visible
- Respect dataset orientation:
    front & rear images are flipped (flip 1 then flip 0) in visi_video.py
    -> We flip before projecting/overlay, then flip back for display.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

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
from utils.visibility import (
    compute_visible_bev_and_flags,
    project_cuboid_to_mask,
    load_camera_visibility_mask
)

# ============================================================
# Palette (BGR for overlay, but we keep deterministic by local_id)
# ============================================================
PALETTE_BGR = [
    (255,   0,   0),  # blue
    (  0, 255,   0),  # green
    (  0,   0, 255),  # red
    (255, 255,   0),  # cyan
    (255,   0, 255),  # magenta
    (  0, 255, 255),  # yellow
    (128,   0, 255),
    (255, 128,   0),
    (  0, 128, 255),
]

def color_for_local_id(lid: int) -> Tuple[int, int, int]:
    return PALETTE_BGR[int(lid) % len(PALETTE_BGR)]

# ============================================================
# Helpers
# ============================================================
def flip_if_needed(img: np.ndarray, view: str) -> np.ndarray:
    """Match visi_video.py convention: front & rear are flipped."""
    if view in ("front", "rear"):
        img = cv2.flip(img, 1)
        img = cv2.flip(img, 0)
    return img

def unflip_if_needed(img: np.ndarray, view: str) -> np.ndarray:
    """Inverse is same operation for 180° flip."""
    if view in ("front", "rear"):
        img = cv2.flip(img, 1)
        img = cv2.flip(img, 0)
    return img

def prep(img: Optional[np.ndarray], title: str, size: Tuple[int, int]) -> np.ndarray:
    """Resize + title for panel."""
    W, H = size
    if img is None:
        out = np.zeros((H, W, 3), np.uint8)
    else:
        out = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

    cv2.putText(out, title, (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return out

def overlay_visible_object_masks(
    img_bgr: np.ndarray,
    view: str,
    cuboids: List[Dict],
    visible_by_id: Dict[int, bool],
    ext_w2c: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    xi: float,
    alpha: float = 0.55,
) -> np.ndarray:
    """
    Overlay ONLY visible objects as colored pixel masks on the given camera image.

    Steps:
    - Flip image to match projection coordinate convention for front/rear (as in visi_video.py)
    - Load & flip camera mask accordingly
    - For each cuboid:
        if visible_by_id[local_id] is True:
            project_cuboid_to_mask -> obj_mask (H,W) bool
            apply cam_vis_mask
            paint overlay color (by local_id)
    - Blend overlay onto image
    - Unflip back for display
    """
    H, W = img_bgr.shape[:2]

    # Flip working image (for front/rear only)
    work = flip_if_needed(img_bgr.copy(), view)

    mask = load_camera_visibility_mask(view=view,image_shape=(H,W))

    overlay = np.zeros_like(work, dtype=np.uint8)

    for cub in cuboids:
        lid = int(cub.get("local_id", -1))
        if lid < 0:
            continue

        # Only draw visible objects
        if not visible_by_id.get(lid, False):
            continue

        obj_mask = project_cuboid_to_mask(
            np.asarray(cub["corners"], dtype=np.float64),
            ext_w2c,
            K, D, xi,
            (H, W),
        ).astype(bool)
        obj_mask = obj_mask & mask
        # Apply camera visibility region
        # obj_mask &= cam_vis

        if not np.any(obj_mask):
            continue

        overlay[obj_mask] = color_for_local_id(lid)

    # Blend
    blended = cv2.addWeighted(work, 1.0, overlay, alpha, 0.0)

    # Unflip for display
    blended = unflip_if_needed(blended, view)
    return blended

def build_bev_visibility_mask_panel(obj_masks: List[Dict], visible_by_id: Dict[int, bool], shape_hw: Tuple[int, int]) -> np.ndarray:
    """
    BEV-space visibility mask:
        white where object's BEV mask is visible, else black.
    """
    H, W = shape_hw
    m = np.zeros((H, W), np.uint8)
    for o in obj_masks:
        lid = int(o.get("local_id")) if "local_id" in o else None
        # NOTE: obj_masks from segment_objects might not include local_id;
        # In your pipeline, visibility_by_id uses RuntimeObject.local_id (from cuboids).
        # So we should build VISI from cuboids/RuntimeObjects ideally.
        # However, in your existing validation scripts you used enumerate index.
        # We'll handle both robustly:
        pass
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)

# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--id", required=True)

    ap.add_argument("--resolution", type=float, default=100 / (6 * 400))
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--offset", type=float, default=33.0)
    ap.add_argument("--yshift", type=float, default=-0.377)

    ap.add_argument("--visible-ratio-thresh", type=float, default=0.01)
    ap.add_argument("--min-pixels", type=int, default=20)

    ap.add_argument("--panel-size", type=int, default=512)
    ap.add_argument("--alpha", type=float, default=0.55)

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

    H_bev, W_bev = bev_seg.shape[:2]

    # --------------------------------------------------------
    # BEV -> objects -> cuboids (same as your pipeline)
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

    cam_images: Dict[str, Optional[np.ndarray]] = {}
    for view in cam_name_map:
        p = root / "rgb" / view / f"{fid}.png"
        cam_images[view] = cv2.imread(str(p)) if p.exists() else None

    # --------------------------------------------------------
    # Visibility (source of truth)
    # --------------------------------------------------------
    bev_visible, visible_by_id, _best_ratio = compute_visible_bev_and_flags(
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
    # Build camera overlays (VISIBLE MASKS per ID)
    # --------------------------------------------------------
    rendered = {}
    for view, cam_key in cam_name_map.items():
        img = cam_images.get(view)
        if img is None:
            rendered[view] = None
            continue

        ext = extrinsics[cam_key]
        rendered[view] = overlay_visible_object_masks(
            img_bgr=img,
            view=view,
            cuboids=cuboids,
            visible_by_id=visible_by_id,
            ext_w2c=ext,
            K=K, D=D, xi=xi,
            alpha=float(args.alpha),
        )

    # --------------------------------------------------------
    # Build VISI BEV-space mask (from bev_visible vs background)
    # Here we show bev_visible as-is; plus a binary VISI mask panel for clarity.
    # --------------------------------------------------------
    bev_gt_bgr = cv2.cvtColor(bev_seg, cv2.COLOR_RGB2BGR)
    bev_vis_bgr = cv2.cvtColor(bev_visible, cv2.COLOR_RGB2BGR)

    # Binary mask: where bev_visible differs from background(0,0,0)
    vis_bin = (np.any(bev_vis_bgr != 0, axis=2)).astype(np.uint8) * 255
    vis_bin_bgr = cv2.cvtColor(vis_bin, cv2.COLOR_GRAY2BGR)

    # --------------------------------------------------------
    # Compose 2x3
    # Row1: Left | Front | Right
    # Row2: VISI | Rear | GT
    # --------------------------------------------------------
    S = int(args.panel_size)
    size = (S, S)

    row1 = cv2.hconcat([
        prep(rendered.get("left"),  "Left (visible masks)",  size),
        prep(rendered.get("front"), "Front (visible masks)", size),
        prep(rendered.get("right"), "Right (visible masks)", size),
    ])

    row2 = cv2.hconcat([
        prep(vis_bin_bgr,           "VISI (binary)",         size),
        prep(rendered.get("rear"),  "Rear (visible masks)",  size),
        prep(bev_gt_bgr,            "GT BEV",                size),
    ])

    grid = cv2.vconcat([row1, row2])

    cv2.imshow("Visibility Mask Validate (4 cams)", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
