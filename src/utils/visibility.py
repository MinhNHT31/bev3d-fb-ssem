# -*- coding: utf-8 -*-
"""
visibility.py
=================================================

VISIBILITY FILTERING PIPELINE — VISIBILITY BY CAMERA

GOAL
----
Determine which BEV objects are visible from EACH camera independently.

KEY DESIGN PRINCIPLES
---------------------
1. WorldObject (BEV / world space) is the SINGLE source of truth
2. Cameras ONLY reason about visibility (never define objects)
3. FOV filtering is PURELY GEOMETRIC (BEV footprint corners)
4. Occlusion is computed in IMAGE SPACE
5. Visibility is computed PER CAMERA
6. ANY-camera visibility = OR over cameras (derived result)

WHY THIS DESIGN
---------------
- Eliminates "ghost objects"
- Makes debugging per-camera correct and deterministic
- Clean separation of geometry vs visibility
- Ready for research / dataset generation / paper-quality code
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
# BEV PARAMETERS (MUST MATCH DATASET)
# ============================================================

BEV_RESOLUTION = 2 / 48.0
OFFSET = 33.0
BEV_SHAPE = (600, 600)   # (H, W)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class WorldObject:
    """
    Canonical object representation in WORLD / BEV space.
    """
    object_id: int
    corners_world: np.ndarray      # (8,3) cuboid corners
    center_world: np.ndarray       # (3,)
    corners_bev: np.ndarray        # (4,2) BEV footprint (bottom face)
    bev_mask: np.ndarray           # (H,W) BEV segmentation mask


@dataclass
class CameraObject:
    """
    WorldObject mapped into ONE camera.
    Used ONLY for per-camera occlusion reasoning.
    """
    object_id: int
    corners_world: np.ndarray      # (8,3)
    corners_cam: np.ndarray        # (8,3) camera-space
    corners_bev: np.ndarray        # (4,2)
    uv_corners: np.ndarray         # (8,2)
    in_fov: bool


@dataclass
class VisibilityConfig:
    """
    Tunable parameters controlling visibility decision.
    """
    # Occlusion decision
    visible_ratio_thresh: float = 0.33
    min_pixels: int = 50

    # Cuboid → image projection
    segments: int = 200
    min_valid_points: int = 10

    # Ego handling
    ego_id: int = 100

    # FOV gate (BEV corners)
    min_fov_corners: int = 2


@dataclass
class CameraContext:
    """
    Context of ONE camera for visibility reasoning.
    """
    view: str
    cam_key: str
    extrinsic_w2c: np.ndarray
    image_shape: Tuple[int, int]
    cam_mask: Optional[np.ndarray]     # IMAGE-space validity mask
    fov_mask: Optional[np.ndarray]     # BEV-space FOV mask
    objects: List[CameraObject]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def ego_distance(center_world: np.ndarray) -> float:
    """Distance from world origin (used to choose ego)."""
    return float(np.linalg.norm(center_world))


def world_to_bev(X: float, Z: float) -> Tuple[int, int]:
    """
    Convert world (X,Z) → BEV pixel (u,v).
    """
    H, W = BEV_SHAPE
    u = int(X / BEV_RESOLUTION + W / 2.0)
    v = int(H / 2.0 - (Z / BEV_RESOLUTION - OFFSET))
    return u, v


def corners_world_to_bev(corners_world: np.ndarray) -> np.ndarray:
    """
    Convert cuboid bottom face (4 corners) to BEV footprint.
    """
    corners_world = np.asarray(corners_world, dtype=np.float64).reshape(8, 3)
    bottom = corners_world[:4]

    bev = []
    for X, _, Z in bottom:
        bev.append(world_to_bev(float(X), float(Z)))

    return np.asarray(bev, dtype=np.int32)


def assign_ego_id_once(world_objects: List[WorldObject], ego_id: int) -> None:
    """
    Assign ego_id to the closest object ONCE.
    """
    if not world_objects:
        return

    ego_obj = min(world_objects, key=lambda o: ego_distance(o.center_world))
    if ego_obj.object_id == ego_id:
        return

    occupied = {o.object_id for o in world_objects}
    if ego_id in occupied:
        for o in world_objects:
            if o.object_id == ego_id and o is not ego_obj:
                new_id = 1
                while new_id in occupied:
                    new_id += 1
                o.object_id = new_id
                break

    ego_obj.object_id = ego_id


def gate_by_bev_fov_corners(
    corners_bev: np.ndarray,
    fov_mask: Optional[np.ndarray],
    *,
    min_inside: int,
) -> bool:
    """
    BEV FOV gate using footprint corners.

    Object is kept if at least `min_inside` corners
    lie inside the camera FOV.
    """
    if fov_mask is None:
        return True

    H, W = fov_mask.shape
    inside = 0

    for u, v in corners_bev:
        if 0 <= v < H and 0 <= u < W:
            if fov_mask[v, u]:
                inside += 1

    return inside >= min_inside


# ============================================================
# MASK LOADERS
# ============================================================

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_camera_visibility_mask(view: str) -> Optional[np.ndarray]:
    """
    IMAGE-space camera validity mask.
    True = valid pixel.
    """
    p = _project_root() / "masks/forward_looking_camera_model/masks" / f"{view}.npy"
    if not p.exists():
        return None
    return np.load(p).astype(bool)


def load_fov_mask(view: str) -> Optional[np.ndarray]:
    """
    BEV-space camera FOV mask.
    """
    p = _project_root() / "masks/unity_data/bev_mask.npy"
    if not p.exists():
        return None

    bev = np.load(p)

    if view == "front":
        return bev[0, 0, :, 0].reshape(BEV_SHAPE).astype(bool)
    if view == "rear":
        return bev[2, 0, :, 1].reshape(BEV_SHAPE).astype(bool)
    if view == "left":
        return bev[1, 0, :, 2].reshape(BEV_SHAPE).astype(bool)
    if view == "right":
        return bev[3, 0, :, 2].reshape(BEV_SHAPE).astype(bool)

    return None


# ============================================================
# BUILD WORLD OBJECTS
# ============================================================

def build_world_objects(
    obj_masks: List[Dict[str, Any]],
    cuboids: List[Dict[str, Any]],
) -> List[WorldObject]:
    """
    Combine BEV segmentation + cuboid geometry.
    """
    objs: List[WorldObject] = []
    n = min(len(obj_masks), len(cuboids))

    for i in range(n):
        corners = np.asarray(cuboids[i]["corners"], dtype=np.float64).reshape(8, 3)
        oid = int(cuboids[i].get("local_id", i))

        objs.append(
            WorldObject(
                object_id=oid,
                corners_world=corners,
                center_world=corners.mean(axis=0),
                corners_bev=corners_world_to_bev(corners),
                bev_mask=obj_masks[i]["mask"],
            )
        )

    return objs


# ============================================================
# BUILD CAMERA CONTEXTS
# ============================================================

def build_camera_contexts(
    world_objects: List[WorldObject],
    cam_images: Dict[str, Optional[np.ndarray]],
    extrinsics: Dict[str, np.ndarray],
    K: np.ndarray,
    D: np.ndarray,
    xi: float,
    cam_name_map: Dict[str, str],
    cfg: VisibilityConfig,
) -> List[CameraContext]:
    """
    Build per-camera contexts with FOV gating.
    """
    cameras: List[CameraContext] = []

    for view, cam_key in cam_name_map.items():
        img = cam_images.get(view)
        if img is None or cam_key not in extrinsics:
            continue

        H, W = img.shape[:2]
        extrinsic = extrinsics[cam_key]

        cam_mask = load_camera_visibility_mask(view)
        if cam_mask is None:
            cam_mask = np.ones((H, W), dtype=bool)

        fov_mask = load_fov_mask(view)
        cam_objects: List[CameraObject] = []

        for obj in world_objects:
            # Ego always kept
            if obj.object_id != cfg.ego_id:
                if not gate_by_bev_fov_corners(
                    obj.corners_bev,
                    fov_mask,
                    min_inside=cfg.min_fov_corners,
                ):
                    continue

            uv, _ = cam2image(obj.corners_world, extrinsic, K, D, xi)

            pts_h = np.hstack([obj.corners_world, np.ones((8, 1))])
            pts_cam = (extrinsic @ pts_h.T).T[:, :3]

            cam_objects.append(
                CameraObject(
                    object_id=obj.object_id,
                    corners_world=obj.corners_world,
                    corners_cam=pts_cam,
                    corners_bev=obj.corners_bev,
                    uv_corners=uv,
                    in_fov=True,
                )
            )

        cameras.append(
            CameraContext(
                view=view,
                cam_key=cam_key,
                extrinsic_w2c=extrinsic,
                image_shape=(H, W),
                cam_mask=cam_mask,
                fov_mask=fov_mask,
                objects=cam_objects,
            )
        )

    return cameras


# ============================================================
# CUBOID → IMAGE MASK (OCCLUSION)
# ============================================================

EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]


def project_cuboid_to_mask(
    corners_world: np.ndarray,
    extrinsic_w2c: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    xi: float,
    image_shape: Tuple[int, int],
    *,
    segments: int,
    min_valid_points: int,
    cam_mask: Optional[np.ndarray],
) -> np.ndarray:
    """
    Project cuboid into IMAGE-space boolean mask.
    """
    H, W = image_shape
    all_pts = []

    for s, e in EDGES:
        p1, p2 = corners_world[s], corners_world[e]
        t = np.linspace(0, 1, segments).reshape(-1, 1)
        pts3d = p1 + (p2 - p1) * t

        uv, valid = cam2image(pts3d, extrinsic_w2c, K, D, xi)
        if valid is None:
            continue

        pts = uv[np.asarray(valid, bool)]
        pts = pts[
            (pts[:,0]>=0)&(pts[:,0]<W)&
            (pts[:,1]>=0)&(pts[:,1]<H)
        ]
        if pts.size > 0:
            all_pts.append(pts)

    if not all_pts:
        return np.zeros((H, W), bool)

    pts2d = np.vstack(all_pts)
    if pts2d.shape[0] < min_valid_points:
        return np.zeros((H, W), bool)

    hull = cv2.convexHull(np.round(pts2d).astype(np.int32))
    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)

    out = mask.astype(bool)
    if cam_mask is not None:
        out &= cam_mask

    return out


# ============================================================
# VISIBILITY ENGINE — PER CAMERA
# ============================================================

class VisibilityEngine:
    """
    Compute visibility PER CAMERA.
    """

    def __init__(self, K, D, xi, cfg: VisibilityConfig):
        self.K = K
        self.D = D
        self.xi = xi
        self.cfg = cfg

    def _sort_near_to_far(self, objs: List[CameraObject]) -> List[CameraObject]:
        return sorted(objs, key=lambda o: np.linalg.norm(o.corners_cam.mean(axis=0)))

    def visibility_one_camera(
        self,
        cam: CameraContext,
    ) -> Dict[int, float]:
        """
        Compute visibility ratio PER object for ONE camera.
        """
        H, W = cam.image_shape
        union_prev = np.zeros((H, W), dtype=bool)

        ratio_by_id: Dict[int, float] = {}

        for o in self._sort_near_to_far(cam.objects):
            oid = o.object_id

            if oid == self.cfg.ego_id:
                ratio_by_id[oid] = 1.0
                continue

            mask = project_cuboid_to_mask(
                o.corners_world,
                cam.extrinsic_w2c,
                self.K, self.D, self.xi,
                cam.image_shape,
                segments=self.cfg.segments,
                min_valid_points=self.cfg.min_valid_points,
                cam_mask=cam.cam_mask,
            )

            if not mask.any():
                ratio_by_id[oid] = 0.0
                continue

            total_px = int(mask.sum())
            visible_px = int((mask & ~union_prev).sum())
            ratio = visible_px / max(total_px, 1)

            ratio_by_id[oid] = ratio
            union_prev |= mask

        return ratio_by_id


# ============================================================
# PUBLIC API — VISIBILITY BY CAMERA
# ============================================================

def compute_visible_bev_and_flags(
    bev_seg_rgb,
    obj_masks,
    cuboids,
    cam_images,
    extrinsics,
    K, D, xi,
    cam_name_map,
    visible_ratio_thresh=0.33,
    min_pixels=50,
    ego_id=100,
):
    """
    MAIN ENTRY POINT — VISIBILITY BY CAMERA
    """

    world_objects = build_world_objects(obj_masks, cuboids)

    cfg = VisibilityConfig(
        visible_ratio_thresh=visible_ratio_thresh,
        min_pixels=min_pixels,
        ego_id=ego_id,
    )

    assign_ego_id_once(world_objects, cfg.ego_id)

    cameras = build_camera_contexts(
        world_objects,
        cam_images,
        extrinsics,
        K, D, xi,
        cam_name_map,
        cfg,
    )

    engine = VisibilityEngine(K, D, xi, cfg)

    # ----------------------------------------
    # Visibility PER CAMERA
    # ----------------------------------------
    visible_by_cam: Dict[str, Dict[int, bool]] = {}
    visible_any: Dict[int, bool] = {o.object_id: False for o in world_objects}

    for cam in cameras:
        ratio_by_id = engine.visibility_one_camera(cam)
        cam_visible = {}

        for oid, ratio in ratio_by_id.items():
            is_visible = ratio >= cfg.visible_ratio_thresh
            cam_visible[oid] = is_visible
            if is_visible:
                visible_any[oid] = True

        visible_by_cam[cam.view] = cam_visible

    # ----------------------------------------
    # Filter BEV by ANY camera
    # ----------------------------------------
    bev_out = bev_seg_rgb.copy()
    for obj in world_objects:
        if not visible_any.get(obj.object_id, False):
            bev_out[obj.bev_mask.astype(bool)] = 0

    return bev_out, visible_by_cam, visible_any
