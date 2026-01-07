# -*- coding: utf-8 -*-
"""
visibility.py
=================================================

NEAR → FAR occlusion reasoning in image space (STACK/UNION VERSION).

Core idea (per camera):
1) Sort objects by camera distance: NEAR → FAR
2) Project each cuboid into an image mask (fisheye/MEI curved edges)
3) Maintain union_prev mask = union of ALL objects already projected before
4) For current object:
      total_px   = |mask|
      visible_px = |mask \ union_prev|
      ratio      = visible_px / total_px
   Then update:
      union_prev |= mask

Multi-camera rule:
- An object is visible if visible in ANY camera (best ratio = max across cameras)

Outputs/Interfaces:
- Keep the same function names and return types as your clean version
  so you can run infer/debug without touching other scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2

try:
    from .projects import cam2image
except Exception:
    from utils.projects import cam2image


# ============================================================
# Data structure
# ============================================================

@dataclass
class RuntimeObject:
    local_id: int
    corners: np.ndarray      # (8,3)
    bev_mask: np.ndarray     # (H,W)
    center_world: np.ndarray # (3,)


# ============================================================
# Utilities
# ============================================================

def _as_hom(xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(xyz, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    return np.hstack([pts, ones])


def camera_distance(center_world: np.ndarray, extrinsic_w2c: np.ndarray) -> float:
    p = _as_hom(center_world)[0]
    cam = extrinsic_w2c @ p
    return float(np.linalg.norm(cam[:3]))


# ============================================================
# Projection (CURVED edges for fisheye / MEI)
# ============================================================

def project_cuboid_to_mask(
    corners_world: np.ndarray,
    extrinsic_w2c: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    xi: float,
    image_shape: Tuple[int, int],
    segments: int = 20,
    min_valid_points: int = 10,
    debug: bool = False,
    debug_dir: Optional[str] = None,
    debug_name: str = "cuboid",
) -> np.ndarray:
    """
    Project a 3D cuboid into an image-space binary mask using CURVED edges.

    Strategy:
      - Sample points along all 12 cuboid edges in 3D
      - Project sampled points with cam2image (fisheye/MEI)
      - Collect all valid projected 2D points
      - Build silhouette via convex hull
      - Fill hull to obtain binary mask

    Debug (optional, SAVE ONLY — no imshow):
      - Blue  : projected points
      - Green : hull
      - Red   : filled mask overlay
    """
    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.uint8)

    EDGES = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    corners = np.asarray(corners_world, dtype=np.float64).reshape(8, 3)
    all_pts_2d: List[np.ndarray] = []

    for s, e in EDGES:
        p1, p2 = corners[s], corners[e]
        t = np.linspace(0.0, 1.0, segments).reshape(-1, 1)
        pts3d = p1 + (p2 - p1) * t  # (segments,3)

        uv, valid = cam2image(
            pts3d,
            extrinsic_w2c,
            K, D, xi,
            image_size=(W, H),
        )

        if not np.any(valid):
            continue

        pts = uv[valid]

        inside = (
            (pts[:, 0] >= 0) & (pts[:, 0] < W) &
            (pts[:, 1] >= 0) & (pts[:, 1] < H)
        )
        pts = pts[inside]

        if pts.shape[0] > 0:
            all_pts_2d.append(pts)

    if not all_pts_2d:
        return mask.astype(bool)

    pts2d = np.vstack(all_pts_2d)
    if pts2d.shape[0] < min_valid_points:
        return mask.astype(bool)

    pts2d_i = np.round(pts2d).astype(np.int32)

    hull = cv2.convexHull(pts2d_i.reshape(-1, 1, 2))
    cv2.fillConvexPoly(mask, hull, 1)

    if debug and debug_dir is not None:
        import os
        os.makedirs(debug_dir, exist_ok=True)

        dbg = np.zeros((H, W, 3), dtype=np.uint8)

        # Blue points
        for p in pts2d_i:
            cv2.circle(dbg, (int(p[0]), int(p[1])), 1, (255, 0, 0), -1)

        # Green hull
        cv2.polylines(dbg, [hull], True, (0, 255, 0), 2)

        # Red overlay
        red = np.zeros_like(dbg)
        red[:, :, 2] = (mask * 255).astype(np.uint8)
        dbg = cv2.addWeighted(dbg, 1.0, red, 0.35, 0)

        cv2.imwrite(os.path.join(debug_dir, f"{debug_name}.png"), dbg)

    return mask.astype(bool)


# ============================================================
# Core NEAR → FAR visibility (UNION of all previous objects)
# ============================================================

def render_near_to_far_visibility(
    objects: List[RuntimeObject],
    extrinsic_w2c: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    xi: float,
    image_shape: Tuple[int, int],
) -> Tuple[
    Dict[int, int],  # total_px
    Dict[int, int],  # visible_px
]:
    """
    NOTE: Name kept for compatibility, but logic is now NEAR → FAR.

    Algorithm (as you requested):
      - Sort NEAR → FAR
      - union_prev = union of ALL objects already projected
      - For each object in order:
          total_px   = |mask|
          visible_px = |mask & ~union_prev|
          union_prev |= mask
    """
    # Sort NEAR → FAR (ascending distance)
    pairs = sorted(
        [(camera_distance(o.center_world, extrinsic_w2c), o.local_id) for o in objects],
        key=lambda x: x[0],
    )
    obj_map = {o.local_id: o for o in objects}

    H, W = image_shape
    union_prev = np.zeros((H, W), dtype=bool)

    total_px: Dict[int, int] = {o.local_id: 0 for o in objects}
    visible_px: Dict[int, int] = {o.local_id: 0 for o in objects}

    for _, lid in pairs:
        obj = obj_map[lid]

        mask = project_cuboid_to_mask(
            obj.corners,
            extrinsic_w2c,
            K, D, xi,
            image_shape,
        )

        if not np.any(mask):
            continue

        total = int(mask.sum())
        total_px[lid] = total

        vis_mask = mask & (~union_prev)
        visible_px[lid] = int(vis_mask.sum())

        union_prev |= mask

    return total_px, visible_px


# ============================================================
# Per-camera visibility
# ============================================================

def visibility_one_camera(
    objects: List[RuntimeObject],
    extrinsic_w2c: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    xi: float,
    image_shape: Tuple[int, int],
    visible_ratio_thresh: float,
    min_pixels: int,
) -> Tuple[Dict[int, bool], Dict[int, float]]:
    total_px, visible_px = render_near_to_far_visibility(
        objects, extrinsic_w2c, K, D, xi, image_shape
    )

    visible_by_id: Dict[int, bool] = {}
    ratio_by_id: Dict[int, float] = {}

    for o in objects:
        lid = o.local_id
        t = int(total_px.get(lid, 0))
        v = int(visible_px.get(lid, 0))
        v = max(v, 0)

        r = float(v / max(t, 1))
        ratio_by_id[lid] = r
        visible_by_id[lid] = (t >= int(min_pixels)) and (r >= float(visible_ratio_thresh))

    return visible_by_id, ratio_by_id


# ============================================================
# Multi-camera aggregation
# ============================================================

def visibility_multi_camera_any(
    objects: List[RuntimeObject],
    cam_images: Dict[str, Optional[np.ndarray]],
    extrinsics: Dict[str, np.ndarray],
    K: np.ndarray,
    D: np.ndarray,
    xi: float,
    cam_name_map: Dict[str, str],
    visible_ratio_thresh: float,
    min_pixels: int,
) -> Tuple[Dict[int, bool], Dict[int, float]]:
    final_visible = {o.local_id: False for o in objects}
    best_ratio = {o.local_id: 0.0 for o in objects}

    for view, cam_key in cam_name_map.items():
        img = cam_images.get(view)
        if img is None or cam_key not in extrinsics:
            continue

        H, W = img.shape[:2]
        vis_cam, ratio_cam = visibility_one_camera(
            objects,
            extrinsics[cam_key],
            K, D, xi,
            (H, W),
            visible_ratio_thresh,
            min_pixels,
        )

        for lid in vis_cam:
            best_ratio[lid] = max(best_ratio[lid], float(ratio_cam.get(lid, 0.0)))
            if vis_cam[lid]:
                final_visible[lid] = True

    return final_visible, best_ratio


# ============================================================
# BEV helpers
# ============================================================

def build_runtime_objects(
    obj_masks: List[Dict[str, Any]],
    cuboids: List[Dict[str, Any]],
) -> List[RuntimeObject]:
    objects: List[RuntimeObject] = []
    n = min(len(obj_masks), len(cuboids))

    for i in range(n):
        corners = np.asarray(cuboids[i]["corners"], dtype=np.float64).reshape(8, 3)
        objects.append(
            RuntimeObject(
                local_id=i,
                corners=corners,
                bev_mask=obj_masks[i]["mask"],
                center_world=corners.mean(axis=0),
            )
        )
    return objects


def filter_bev_by_visibility(
    bev_seg_rgb: np.ndarray,
    objects: List[RuntimeObject],
    visible_by_id: Dict[int, bool],
) -> np.ndarray:
    out = bev_seg_rgb.copy()
    for o in objects:
        if not visible_by_id.get(o.local_id, False):
            out[o.bev_mask > 0] = 0
    return out


# ============================================================
# Public API
# ============================================================

def compute_visible_bev_and_flags(
    bev_seg_rgb: np.ndarray,
    obj_masks: List[Dict[str, Any]],
    cuboids: List[Dict[str, Any]],
    cam_images: Dict[str, Optional[np.ndarray]],
    extrinsics: Dict[str, np.ndarray],
    K: np.ndarray,
    D: np.ndarray,
    xi: float,
    cam_name_map: Dict[str, str],
    visible_ratio_thresh: float = 0.33,
    min_pixels: int = 20,
) -> Tuple[np.ndarray, Dict[int, bool], Dict[int, float], None]:
    objects = build_runtime_objects(obj_masks, cuboids)

    visible_by_id, best_ratio = visibility_multi_camera_any(
        objects,
        cam_images,
        extrinsics,
        K, D, xi,
        cam_name_map,
        visible_ratio_thresh,
        min_pixels,
    )

    bev_visible = filter_bev_by_visibility(bev_seg_rgb, objects, visible_by_id)
    return bev_visible, visible_by_id, best_ratio
