# -*- coding: utf-8 -*-
"""
visibility.py
=================================================

NEAR → FAR occlusion reasoning in image space (STACK/UNION VERSION).

Key rules (camera mask only):
- camera mask convention:
    BLACK (0)   = visible region
    WHITE (255) = NOT visible region
- We do NOT use fisheye_visibility_mask anymore.

Ego rule:
- The closest object is treated as ego-car.
- Ego ID is reassigned to ego_id=100.
- Ego is NOT projected into camera (no occlusion contribution),
  but ego is still considered visible in BEV (visible_by_id[100] = True).

Multi-camera rule:
- An object is visible if visible in ANY camera (best ratio = max across cameras).

This file keeps the same public API:
- compute_visible_bev_and_flags(...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path
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
    corners: np.ndarray      # (8, 3)
    bev_mask: np.ndarray     # (H, W) bool/uint8
    center_world: np.ndarray # (3,)


# ============================================================
# Small helpers
# ============================================================

def ego_distance(center_world: np.ndarray) -> float:
    """Euclidean distance from object center to ego origin (0,0,0) in world/ego frame."""
    return float(np.linalg.norm(center_world))


def _project_root() -> Path:
    """
    Repo layout assumption:
        <root>/src/utils/visibility.py  -> parents[2] is <root>
    """
    return Path(__file__).resolve().parents[2]


def load_camera_visibility_mask(
    view: str,
    image_shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    """
    Load camera mask from <project_root>/masks/{view}.png.

    Mask convention:
        BLACK (0)   = visible region  -> True
        WHITE (255) = NOT visible     -> False

    Returns:
        cam_vis_mask: (H, W) bool  OR  None if missing/unreadable
    """
    H, W = image_shape
    root = _project_root()
    p = root / "masks" / f"{view}.png"
    if not p.exists():
        return None

    gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None

    if gray.shape != (H, W):
        gray = cv2.resize(gray, (W, H), interpolation=cv2.INTER_NEAREST)

    # BLACK = visible
    cam_vis = (gray == 255)
    return cam_vis.astype(bool)


def _ensure_unique_ego_id(objects: List[RuntimeObject], ego_old_id: int, ego_id: int = 100) -> None:
    """
    Reassign closest object's local_id to ego_id safely.

    If some other object already has ego_id, we move that object's id to a free one.
    This prevents KeyError / id collisions in dicts.
    """
    # Find ego object by old id
    ego_obj = None
    for o in objects:
        if o.local_id == ego_old_id:
            ego_obj = o
            break
    if ego_obj is None:
        raise RuntimeError("Ego object not found for id reassignment.")

    # If ego already has ego_id, nothing to do
    if ego_old_id == ego_id:
        return

    # If another object has ego_id, move it away to a free id
    occupied = {o.local_id for o in objects}
    if ego_id in occupied:
        # Find the conflicting object (not ego)
        conflict = None
        for o in objects:
            if o.local_id == ego_id and o is not ego_obj:
                conflict = o
                break
        if conflict is not None:
            # pick a free id
            new_free = 1
            while new_free in occupied:
                new_free += 1
            conflict.local_id = new_free

    # Finally assign ego id
    ego_obj.local_id = ego_id


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
    segments: int = 200,
    min_valid_points: int = 10,
    vis_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Project a 3D cuboid into image-space binary mask using curved edges.

    Returns:
        mask: (H, W) bool
    """
    H, W = image_shape
    mask_u8 = np.zeros((H, W), dtype=np.uint8)
    # print(vis_mask)

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
        pts3d = p1 + (p2 - p1) * t

        uv, valid = cam2image(
            pts3d,
            extrinsic_w2c,
            K, D, xi
        )

        if valid is None or not np.any(valid):
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
        return mask_u8.astype(bool)

    pts2d = np.vstack(all_pts_2d)
    if pts2d.shape[0] < min_valid_points:
        return mask_u8.astype(bool)

    pts2d_i = np.round(pts2d).astype(np.int32)

    hull = cv2.convexHull(pts2d_i.reshape(-1, 1, 2))
    cv2.fillConvexPoly(mask_u8, hull, 1)

    obj_mask = mask_u8.astype(bool)

    if vis_mask is not None:
        obj_mask = obj_mask & vis_mask

    return obj_mask

# ============================================================
# Core NEAR → FAR visibility (one camera)
# ============================================================

def render_near_to_far_visibility(
    objects: List[RuntimeObject],
    extrinsic_w2c: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    xi: float,
    image_shape: Tuple[int, int],
    cam_vis_mask: Optional[np.ndarray],   # True=visible pixels
    ego_id: int = 100,
) -> Tuple[Dict[int, int], Dict[int, int], List[RuntimeObject]]:
    """
    Compute occlusion-aware visibility for ONE camera.

    Inputs:
        cam_vis_mask:
            - bool (H,W) where True = visible pixels
            - if None -> all pixels visible

    Ego behavior:
        - closest object becomes ego (id -> ego_id)
        - ego is NOT projected into camera, and does not occlude anything

    Returns:
        total_px[lid], visible_px[lid], objects (ego id already updated)
    """
    H, W = image_shape

    # Sort NEAR -> FAR
    pairs = sorted(
        [(ego_distance(o.center_world), o.local_id) for o in objects],
        key=lambda x: x[0],
    )
    if not pairs:
        return {}, {}, objects

    # Identify ego by closest distance
    _, ego_old_id = pairs[0]
    _ensure_unique_ego_id(objects, ego_old_id, ego_id=ego_id)

    # Rebuild pairs after potential id changes
    pairs = sorted(
        [(ego_distance(o.center_world), o.local_id) for o in objects],
        key=lambda x: x[0],
    )

    # Build object map
    obj_map = {o.local_id: o for o in objects}

    # Visibility region from camera mask (None -> all visible)
    if cam_vis_mask is None:
        cam_vis_mask = np.ones((H, W), dtype=bool)
    elif cam_vis_mask.shape != (H, W):
        raise ValueError(f"cam_vis_mask shape {cam_vis_mask.shape} != {(H, W)}")
    
    union_prev = np.zeros((H, W), dtype=bool)
    total_px: Dict[int, int] = {}
    visible_px: Dict[int, int] = {}

    for _, lid in pairs:
        # Skip ego projection completely (but we still want ego visible in BEV)
        if lid == ego_id:
            total_px[lid] = 0
            visible_px[lid] = 0
            continue

        obj = obj_map[lid]

        obj_mask = project_cuboid_to_mask(
            obj.corners,
            extrinsic_w2c,
            K, D, xi,
            image_shape,
            vis_mask= cam_vis_mask
        )
        # print(obj_mask)
        # Restrict to camera-visible pixels
        if not np.any(obj_mask):
            total_px[lid] = 0
            visible_px[lid] = 0
            continue

        total = int(obj_mask.sum())
        vis = int((obj_mask & (~union_prev)).sum())

        total_px[lid] = total
        visible_px[lid] = vis

        union_prev |= obj_mask

    return total_px, visible_px, objects


def visibility_one_camera(
    objects: List[RuntimeObject],
    extrinsic_w2c: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    xi: float,
    image_shape: Tuple[int, int],
    visible_ratio_thresh: float,
    min_pixels: int,
    cam_vis_mask: bool,
    ego_id: int = 100,
) -> Tuple[Dict[int, bool], Dict[int, float], List[RuntimeObject]]:
    """
    Per-camera visibility flags + ratios.

    NOTE: ego_id is always set visible=True (even though it is not projected).
    """
    total_px, visible_px, new_objects = render_near_to_far_visibility(
        objects,
        extrinsic_w2c, K, D, xi,
        image_shape,
        cam_vis_mask=cam_vis_mask,
        ego_id=ego_id,
    )

    visible_by_id: Dict[int, bool] = {}
    ratio_by_id: Dict[int, float] = {}

    for o in new_objects:
        lid = o.local_id

        # Ego: force visible in BEV logic
        if lid == ego_id:
            visible_by_id[lid] = True
            ratio_by_id[lid] = 1.0
            continue

        t = int(total_px.get(lid, 0))
        v = int(visible_px.get(lid, 0))
        r = float(v / max(t, 1))

        ratio_by_id[lid] = r
        visible_by_id[lid] = (t >= int(min_pixels)) and (r >= float(visible_ratio_thresh))

    return visible_by_id, ratio_by_id, new_objects


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
    ego_id: int = 100,
) -> Tuple[Dict[int, bool], Dict[int, float], List[RuntimeObject]]:
    """
    Aggregate multi-camera visibility:
        visible if visible in ANY camera
        best_ratio = max ratio across cameras
    """
    # Make ego id consistent ONCE before aggregation (avoid KeyError later)
    if objects:
        # Determine closest id first
        pairs = sorted([(ego_distance(o.center_world), o.local_id) for o in objects], key=lambda x: x[0])
        _, ego_old_id = pairs[0]
        _ensure_unique_ego_id(objects, ego_old_id, ego_id=ego_id)

    # Initialize dicts with current ids (including ego_id)
    final_visible: Dict[int, bool] = {o.local_id: False for o in objects}
    best_ratio: Dict[int, float] = {o.local_id: 0.0 for o in objects}

    # Ego always visible in BEV
    if ego_id in final_visible:
        final_visible[ego_id] = True
        best_ratio[ego_id] = 1.0

    for view, cam_key in cam_name_map.items():
        img = cam_images.get(view)
        if img is None:
            continue
        if cam_key not in extrinsics:
            continue

        H, W = img.shape[:2]
        cam_vis = load_camera_visibility_mask(view, (H, W))
        # print(cam_vis)
        # if cam_vis is None:
        #     cam_vis = np.ones((H, W), dtype=bool)

        vis_cam, ratio_cam, new_objects = visibility_one_camera(
            objects,
            extrinsics[cam_key],
            K, D, xi,
            (H, W),
            visible_ratio_thresh,
            min_pixels,
            cam_vis_mask=cam_vis,
            ego_id=ego_id,
        )

        # Update dicts (must include ego_id & all ids)
        for lid, vflag in vis_cam.items():
            if lid not in best_ratio:
                best_ratio[lid] = 0.0
            if lid not in final_visible:
                final_visible[lid] = False

            best_ratio[lid] = max(best_ratio[lid], float(ratio_cam.get(lid, 0.0)))
            if vflag:
                final_visible[lid] = True

        objects = new_objects  # keep updated ids

    return final_visible, best_ratio, objects


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
        local_id = int(cuboids[i].get("local_id", i))

        objects.append(
            RuntimeObject(
                local_id=local_id,
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
    min_pixels: int = 50,
    ego_id: int = 100,
) -> Tuple[np.ndarray, Dict[int, bool], Dict[int, float]]:
    """
    Main entry:
    - build RuntimeObject list
    - compute multi-camera visibility
    - filter BEV by visibility
    """
    objects = build_runtime_objects(obj_masks, cuboids)

    visible_by_id, best_ratio, objects = visibility_multi_camera_any(
        objects,
        cam_images,
        extrinsics,
        K, D, xi,
        cam_name_map,
        visible_ratio_thresh,
        min_pixels,
        ego_id=ego_id,
    )

    bev_visible = filter_bev_by_visibility(bev_seg_rgb, objects, visible_by_id)
    return bev_visible, visible_by_id, best_ratio
