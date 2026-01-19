# -*- coding: utf-8 -*-
"""
visibility.py
=================================================

Refactor: VisibilityEngine (stateful) to avoid passing repeated params everywhere.

Behavior kept (same public API):
- compute_visible_bev_and_flags(...)

Core logic kept:
- NEAR -> FAR occlusion in IMAGE space using union_prev
- Camera visibility mask (from npy) restricts projection area
- Multi-camera: visible if ANY camera sees it (best_ratio = max)
- Ego: closest object becomes ego_id=100, not projected, always visible in BEV

Optional extension hook:
- BEV FOV gate can be added cleanly (loader style) without polluting function signatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2

try:
    from .projects import cam2image
except Exception:
    from utils.projects import cam2image


# ============================================================
# Data structures
# ============================================================

@dataclass
class RuntimeObject:
    local_id: int
    corners: np.ndarray      # (8, 3)
    bev_mask: np.ndarray     # (H, W) bool/uint8
    center_world: np.ndarray # (3,)


@dataclass
class VisibilityConfig:
    visible_ratio_thresh: float = 0.33
    min_pixels: int = 50
    ego_id: int = 100
    # projection quality knobs
    segments: int = 200
    min_valid_points: int = 10


@dataclass
class CameraContext:
    """All per-camera constants packed here."""
    view: str
    cam_key: str
    extrinsic_w2c: np.ndarray
    image_shape: Tuple[int, int]  # (H,W)
    vis_mask: Optional[np.ndarray]  # (H,W) bool (True=visible pixels)


# ============================================================
# Small helpers
# ============================================================

def ego_distance(center_world: np.ndarray) -> float:
    return float(np.linalg.norm(center_world))


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_camera_visibility_mask(view: str) -> Optional[np.ndarray]:
    """
    Load IMAGE-space camera visibility mask from:
        <project_root>/masks/forward_looking_camera_model/masks/{view}.npy

    Convention:
        True = visible pixels
    """
    root = _project_root()
    p = root / "masks" / "forward_looking_camera_model" / "masks" / f"{view}.npy"
    if not p.exists():
        return None
    cam_vis = np.load(p)
    return cam_vis.astype(bool)


def _ensure_unique_ego_id(objects: List[RuntimeObject], ego_old_id: int, ego_id: int = 100) -> None:
    ego_obj = None
    for o in objects:
        if o.local_id == ego_old_id:
            ego_obj = o
            break
    if ego_obj is None:
        raise RuntimeError("Ego object not found for id reassignment.")

    if ego_old_id == ego_id:
        return

    occupied = {o.local_id for o in objects}
    if ego_id in occupied:
        conflict = None
        for o in objects:
            if o.local_id == ego_id and o is not ego_obj:
                conflict = o
                break
        if conflict is not None:
            new_free = 1
            while new_free in occupied:
                new_free += 1
            conflict.local_id = new_free

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
    Returns: (H,W) bool
    """
    H, W = image_shape
    mask_u8 = np.zeros((H, W), dtype=np.uint8)

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

        uv, valid = cam2image(pts3d, extrinsic_w2c, K, D, xi)
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
# Engine
# ============================================================

class VisibilityEngine:
    """
    Stateful visibility evaluator.
    You pass repeated constants once (K,D,xi, config),
    and per-camera contexts are built once per frame.
    """

    def __init__(self, K: np.ndarray, D: np.ndarray, xi: float, config: VisibilityConfig):
        self.K = K
        self.D = D
        self.xi = float(xi)
        self.cfg = config

    def _assign_ego_id_once(self, objects: List[RuntimeObject]) -> None:
        if not objects:
            return
        pairs = sorted([(ego_distance(o.center_world), o.local_id) for o in objects], key=lambda x: x[0])
        _, ego_old_id = pairs[0]
        _ensure_unique_ego_id(objects, ego_old_id, ego_id=self.cfg.ego_id)

    def _render_near_to_far_one_camera(
        self,
        objects: List[RuntimeObject],
        cam: CameraContext,
    ) -> Tuple[Dict[int, int], Dict[int, int], List[RuntimeObject]]:
        """
        Return total_px and visible_px per object id for ONE camera.
        """
        H, W = cam.image_shape

        # Ensure ego is consistent (but do it once per engine call; safe here too)
        self._assign_ego_id_once(objects)

        # Sort near -> far
        pairs = sorted([(ego_distance(o.center_world), o.local_id) for o in objects], key=lambda x: x[0])
        if not pairs:
            return {}, {}, objects

        obj_map = {o.local_id: o for o in objects}

        # Camera-visible region
        if cam.vis_mask is None:
            cam_vis_mask = np.ones((H, W), dtype=bool)
        else:
            cam_vis_mask = cam.vis_mask
            if cam_vis_mask.shape != (H, W):
                raise ValueError(f"[{cam.view}] cam_vis_mask shape {cam_vis_mask.shape} != {(H, W)}")

        union_prev = np.zeros((H, W), dtype=bool)
        total_px: Dict[int, int] = {}
        visible_px: Dict[int, int] = {}

        for _, lid in pairs:
            # Ego: not projected
            if lid == self.cfg.ego_id:
                total_px[lid] = 0
                visible_px[lid] = 0
                continue

            obj = obj_map[lid]

            obj_mask = project_cuboid_to_mask(
                obj.corners,
                cam.extrinsic_w2c,
                self.K, self.D, self.xi,
                cam.image_shape,
                segments=self.cfg.segments,
                min_valid_points=self.cfg.min_valid_points,
                vis_mask=cam_vis_mask,
            )

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

    def _visibility_flags_one_camera(
        self,
        objects: List[RuntimeObject],
        cam: CameraContext,
    ) -> Tuple[Dict[int, bool], Dict[int, float], List[RuntimeObject]]:
        total_px, visible_px, new_objects = self._render_near_to_far_one_camera(objects, cam)

        visible_by_id: Dict[int, bool] = {}
        ratio_by_id: Dict[int, float] = {}

        for o in new_objects:
            lid = o.local_id

            if lid == self.cfg.ego_id:
                visible_by_id[lid] = True
                ratio_by_id[lid] = 1.0
                continue

            t = int(total_px.get(lid, 0))
            v = int(visible_px.get(lid, 0))
            r = float(v / max(t, 1))

            ratio_by_id[lid] = r
            visible_by_id[lid] = (t >= int(self.cfg.min_pixels)) and (r >= float(self.cfg.visible_ratio_thresh))

        return visible_by_id, ratio_by_id, new_objects

    def evaluate_any_camera(
        self,
        objects: List[RuntimeObject],
        cameras: List[CameraContext],
    ) -> Tuple[Dict[int, bool], Dict[int, float], List[RuntimeObject]]:
        """
        Multi-camera aggregation:
          visible if ANY camera sees it
          best_ratio = max ratio
        """
        self._assign_ego_id_once(objects)

        final_visible: Dict[int, bool] = {o.local_id: False for o in objects}
        best_ratio: Dict[int, float] = {o.local_id: 0.0 for o in objects}

        if self.cfg.ego_id in final_visible:
            final_visible[self.cfg.ego_id] = True
            best_ratio[self.cfg.ego_id] = 1.0

        for cam in cameras:
            vis_cam, ratio_cam, new_objects = self._visibility_flags_one_camera(objects, cam)

            for lid in vis_cam:
                if lid not in best_ratio:
                    best_ratio[lid] = 0.0
                if lid not in final_visible:
                    final_visible[lid] = False

                best_ratio[lid] = max(best_ratio[lid], float(ratio_cam.get(lid, 0.0)))
                if vis_cam.get(lid, False):
                    final_visible[lid] = True

            objects = new_objects

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


def build_camera_contexts(
    cam_images: Dict[str, Optional[np.ndarray]],
    extrinsics: Dict[str, np.ndarray],
    cam_name_map: Dict[str, str],
) -> List[CameraContext]:
    """
    Build per-camera context once per frame.
    """
    cams: List[CameraContext] = []
    for view, cam_key in cam_name_map.items():
        img = cam_images.get(view)
        if img is None:
            continue
        if cam_key not in extrinsics:
            continue

        H, W = img.shape[:2]
        cam_vis = load_camera_visibility_mask(view)

        cams.append(
            CameraContext(
                view=view,
                cam_key=cam_key,
                extrinsic_w2c=extrinsics[cam_key],
                image_shape=(H, W),
                vis_mask=cam_vis,
            )
        )
    return cams


# ============================================================
# Public API (unchanged)
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
    Same signature, cleaner internals.
    """
    objects = build_runtime_objects(obj_masks, cuboids)

    cfg = VisibilityConfig(
        visible_ratio_thresh=float(visible_ratio_thresh),
        min_pixels=int(min_pixels),
        ego_id=int(ego_id),
    )

    engine = VisibilityEngine(K=K, D=D, xi=float(xi), config=cfg)
    cameras = build_camera_contexts(cam_images, extrinsics, cam_name_map)

    visible_by_id, best_ratio, objects = engine.evaluate_any_camera(objects, cameras)

    bev_visible = filter_bev_by_visibility(bev_seg_rgb, objects, visible_by_id)
    return bev_visible, visible_by_id, best_ratio
