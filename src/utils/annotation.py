#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FB-SSEM BEV → 3D Annotation (Debug-first, NO global_id)
=======================================================

Goal:
- From FB-SSEM BEV segmentation (RGB) + BEV depth (grayscale):
    1) segment connected components (objects)
    2) compute 2D OBB per object (via utils.bbox2d)
    3) lift to 3D cuboids using depth-derived height map (via utils.bbox3d)
    4) build BEV-centric JSON annotation (NO global_id)
    5) visualize annotation in Open3D (optionally with camera extrinsics)

Important conventions:
- local_id: per-frame index (0..N-1), reset every frame
- NO temporal identity, NO global_id
- Visualization is drawn FROM annotation JSON (sanity check schema correctness)

Expected FB-SSEM split root structure (you pass --dataset-root):
  imagesX/train/
    seg/bev/{frame_id}.png
    depth/{frame_id}.png
    cameraconfig/{frame_id}.txt

Extrinsics file (optional, for camera visualization):
  FB-SSEM/CameraCalibrationParameters/camera_positions_for_extrinsics.txt
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import open3d as o3d

# External utilities (existing in your repo)
from .bbox2d import compute_2d_boxes
from .bbox3d import cuboid_corners
from .camera import load_camera_bev_height, load_extrinsics

np.set_printoptions(precision=3, suppress=True)

# ============================================================
# IO
# ============================================================
def load_depth(path: Path) -> np.ndarray:
    """Load BEV depth as normalized float32 [0,1] from grayscale image."""
    depth = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if depth is None:
        raise FileNotFoundError(f"Cannot read depth: {path}")
    return depth.astype(np.float32) / 255.0


def load_seg(path: Path) -> np.ndarray:
    """Load BEV segmentation as RGB uint8 (preserve dataset colors)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read segmentation: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ============================================================
# Object extraction from BEV segmentation
# ============================================================
def segment_objects(seg_rgb: np.ndarray, min_area: int = 50) -> List[Dict]:
    """
    Convert RGB palette to a single integer label image, then find connected components per color.
    Returns list of objects: {mask(0/255), label(int), color([r,g,b] in 0..1)}
    """
    label_img = (
        seg_rgb[:, :, 0].astype(np.int32) * 256 * 256
        + seg_rgb[:, :, 1].astype(np.int32) * 256
        + seg_rgb[:, :, 2].astype(np.int32)
    )

    objects: List[Dict] = []
    for val in np.unique(label_img):
        if val == 0:
            continue

        class_mask = (label_img == val).astype(np.uint8)
        n, labels = cv2.connectedComponents(class_mask)

        for i in range(1, n):
            comp = (labels == i).astype(np.uint8)
            if int(comp.sum()) < int(min_area):
                continue

            rgb = np.mean(seg_rgb[labels == i], axis=0) / 255.0
            objects.append(
                {
                    "mask": comp * 255,  # uint8 0/255
                    "label": int(val),
                    "color": [float(c) for c in rgb],
                }
            )

    return objects


def get_2d_bounding_boxes(objects: List[Dict]) -> List[Dict]:
    """
    Compute 2D boxes/OBB from masks and carry over metadata.
    Each element in returned list is a dict from compute_2d_boxes plus:
      - color
      - label
    """
    raw_masks = [o["mask"] for o in objects]
    boxes = compute_2d_boxes(raw_masks)
    for b, o in zip(boxes, objects):
        b["color"] = o["color"]
        b["label"] = o["label"]
    return boxes


# ============================================================
# Height lifting
# ============================================================
def compute_height_map(
    depth_norm: np.ndarray,
    bev_cam_height: float,
    min_height: float = 0.1,
) -> np.ndarray:
    """
    Heuristic mapping consistent with your original code:
      height = (1 - depth) * bev_cam_height / 9
    then suppress values below min_height.
    """
    d = np.clip(depth_norm, 0.0, 1.0)
    height = (1.0 - d) * float(bev_cam_height) / 9.0
    height[height < float(min_height)] = 0.0
    return height


# ============================================================
# 3D cuboid construction
# ============================================================
def build_cuboids_from_2d_boxes(
    boxes_2d: List[Dict],
    height_map: np.ndarray,
    resolution: float,
    offset: float = 33.0,
    yshift: float = -0.3,
) -> List[Dict]:
    """
    Build cuboid corners for each 2D OBB, using height_map statistics inside mask.
    Returns internal list of cuboids with corners + debug meta.
    """
    H, W = height_map.shape
    cuboids: List[Dict] = []

    for box in boxes_2d:
        mask_bool = box["mask"].astype(bool)
        vals = height_map[mask_bool]
        h_max = float(vals.max()) if vals.size else 0.0

        # Keep your original heuristic (do not change behavior)
        if h_max >= 3.0:
            final_h = h_max
        elif h_max >= 2.4:
            final_h = h_max / 1.75
        elif h_max >= 2.3:
            final_h = h_max / 2.0
        else:
            final_h = h_max / 1.75

        corners = cuboid_corners(
            box["obb"],
            (H, W),
            resolution,
            min_height=0.0,
            max_height=float(final_h),
            offset=float(offset),
            yshift=float(yshift),
        )

        cuboids.append(
            {
                "obb_2d": box["obb"],
                "corners": corners,  # (8,3)
                "label": box.get("label", 0),
                "color": box.get("color", [1.0, 1.0, 0.0]),
                "h_max": float(h_max),
                "final_h": float(final_h),
            }
        )

    return cuboids


# ============================================================
# Annotation schema (BEV-centric, NO global_id)
# ============================================================
def cuboid_to_bev_obb(corners: np.ndarray) -> Dict:
    """
    Robust OBB params from corners with FB-SSEM axis convention:
      - BEV plane: (X, Z)
      - Height: Y
      - Yaw rotates around Y axis
    size is returned as world extents: [dx (X), dy (Y), dz (Z)]
    """
    corners = np.asarray(corners, dtype=np.float64)

    center = corners.mean(axis=0)

    min_xyz = corners.min(axis=0)
    max_xyz = corners.max(axis=0)

    dx = float(max_xyz[0] - min_xyz[0])  # X extent
    dy = float(max_xyz[1] - min_xyz[1])  # Y extent (height)
    dz = float(max_xyz[2] - min_xyz[2])  # Z extent

    # Yaw from PCA on XZ plane
    xz = corners[:, [0, 2]]
    xz = xz - xz.mean(axis=0)

    if xz.shape[0] >= 2:
        _, _, vh = np.linalg.svd(xz, full_matrices=False)
        direction = vh[0]  # principal axis on XZ
        yaw = float(np.arctan2(direction[1], direction[0]))  # atan2(z, x)
    else:
        yaw = 0.0

    return {
        "center": [float(center[0]), float(center[1]), float(center[2])],
        "size": [dx, dy, dz],
        "yaw": yaw,
        # "yaw_unit": "rad",
        # "axis_convention": "X-right, Y-up(height), Z-forward",
        # "yaw_axis": "Y",
    }

def build_bev_3d_annotation(
    cuboids: List[Dict],
    frame_id: int,
) -> Dict:
    """
    Build JSON-friendly annotation.
    - local_id: per-frame index
    """
    objects: List[Dict] = []

    for local_id, c in enumerate(cuboids):
        objects.append(
            {
                "frame": int(frame_id),
                "local_id": int(local_id),
                "label": c.get("color"),  # optional: keep if useful
                "obb": cuboid_to_bev_obb(c["corners"]),
                "meta": {
                    "h_max": float(c.get("h_max", 0.0)),
                    "final_h": float(c.get("final_h", 0.0)),
                },
            }
        )

    return {
        "frame": int(frame_id),
        "num_objects": int(len(objects)),
        "objects": objects,
    }


# ============================================================
# Visualization (from annotation)
# ============================================================
def unity_to_open3d(points: np.ndarray) -> np.ndarray:
    """
    Convert points from Unity (left-handed) to Open3D (right-handed).

    Accepts:
      - shape (3,)
      - shape (N, 3)

    Returns:
      - same shape as input
    """
    pts = np.asarray(points, dtype=np.float64)

    # If single point (3,), reshape to (1,3)
    single_point = False
    if pts.ndim == 1:
        assert pts.shape[0] == 3, "Point must have 3 coordinates"
        pts = pts[None, :]
        single_point = True

    pts = pts.copy()
    pts[:, 2] *= -1.0  # flip Z axis

    if single_point:
        return pts[0]
    return pts



def visualize_annotation(annotation: Dict, camera_geoms: Optional[List[o3d.geometry.Geometry]] = None):
    """Draw OBBs from annotation dict + optional camera geometries."""
    geoms: List[o3d.geometry.Geometry] = [
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0, 0, 0])
    ]

    for obj in annotation["objects"]:
        obb = obj["obb"]

        center_u = np.array(obb["center"], dtype=np.float64)
        center = unity_to_open3d(center_u)

        dx, dy, dz = [float(x) for x in obb["size"]]
        yaw_u = float(obb["yaw"])

        yaw = -yaw_u  # invert for RH system

        R = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw),  np.cos(yaw), 0.0],
                [0.0,          0.0,         1.0],
            ],
            dtype=np.float64,
        )

        box = o3d.geometry.OrientedBoundingBox(center, R, np.array([dx, dy, dz], dtype=np.float64))
        # box.color = obj.get("label", [1.0, 1.0, 0.0])
        box.color = [0.5, 1.0, 0.0]
        geoms.append(box)

        # Label text is not directly supported by default Open3D draw_geometries;
        # keeping geometry-only view for robust debug.

    if camera_geoms:
        geoms.extend(camera_geoms)

    o3d.visualization.draw_geometries(
        geoms,
        window_name="BEV 3D Annotation Debug",
        width=1280,
        height=720,
    )


# ============================================================
# Camera visualization (your function)
# ============================================================
def create_camera_visuals(extrinsic_matrix, color=[1, 0, 0], size=1.0):
    try:
        cam_pose = np.linalg.inv(extrinsic_matrix)
    except Exception:
        return []

    geoms = []

    w, h, z = size * 0.8, size * 0.6, size
    points_cam = np.array([
        [0, 0, 0],
        [-w, -h, z], [w, -h, z],
        [w,  h, z],  [-w,  h, z]
    ])
    ones = np.ones((5, 1))
    points_world = unity_to_open3d(
    (cam_pose @ np.hstack([points_cam, ones]).T).T[:, :3]
)

    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points_world)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.paint_uniform_color(color)
    geoms.append(ls)

    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.05 * size, cone_radius=0.15 * size,
        cylinder_height=1.0 * size, cone_height=0.3 * size
    )
    arrow.paint_uniform_color(color)

    arrow.transform(cam_pose)
    arrow_points = np.asarray(arrow.vertices)
    arrow.vertices = o3d.utility.Vector3dVector(
    unity_to_open3d(arrow_points)
)
    geoms.append(arrow)

    return geoms


def load_camera_geometries_for_debug(dataset_root: Path) -> List[o3d.geometry.Geometry]:
    """
    Load FB-SSEM extrinsics and build Open3D camera visuals.
    dataset_root: imagesX/{split}
    extrinsics expected at: FB-SSEM/CameraCalibrationParameters/camera_positions_for_extrinsics.txt
    resolved via: dataset_root.parent.parent / CameraCalibrationParameters / ...
    """
    parent_dir = dataset_root.parent.parent  # imagesX/{split} -> imagesX -> FB-SSEM/imagesX? depends on your layout
    extr_path = parent_dir / "CameraCalibrationParameters" / "camera_positions_for_extrinsics.txt"

    if not extr_path.exists():
        # Try one more level up (common when dataset_root is .../FB-SSEM/imagesX/train)
        alt = parent_dir.parent / "CameraCalibrationParameters" / "camera_positions_for_extrinsics.txt"
        extr_path = alt if alt.exists() else extr_path

    if not extr_path.exists():
        return []

    extrinsics = load_extrinsics(str(extr_path))

    cam_colors = {
        "Main Camera-front": [1, 0, 0],
        "Main Camera-left":  [0, 1, 0],
        "Main Camera-right": [0, 0, 1],
        "Main Camera-rear":  [1, 1, 0],
    }

    geoms: List[o3d.geometry.Geometry] = []
    for cam_name, color in cam_colors.items():
        if cam_name not in extrinsics:
            continue

        ext = extrinsics[cam_name]
        if isinstance(ext, dict):
            M = np.eye(4, dtype=np.float64)
            M[:3, :3] = np.array(ext["R"], dtype=np.float64)
            M[:3, 3] = np.array(ext["t"], dtype=np.float64).reshape(3)
            ext = M

        geoms.extend(create_camera_visuals(ext, color=color, size=1.0))

    return geoms


# ============================================================
# CLI / Main
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, help="Path to imagesX/{train|val|test}")
    ap.add_argument("--id", required=True, help="Frame id (filename stem)")
    ap.add_argument("--resolution", type=float, default=100 / (6 * 400))
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--offset", type=float, default=33.0)
    ap.add_argument("--yshift", type=float, default=-0.3)
    ap.add_argument("--vis", action="store_true", help="Visualize annotation in Open3D")
    ap.add_argument("--vis-cam", action="store_true", help="Also visualize camera extrinsics")
    ap.add_argument("--dump-json", action="store_true", help="Print JSON annotation to stdout")
    return ap.parse_args()


def main():
    args = parse_args()
    root = Path(args.dataset_root)

    seg_path = root / "seg" / "bev" / f"{args.id}.png"
    depth_path = root / "depth" / f"{args.id}.png"
    camcfg_path = root / "cameraconfig" / f"{args.id}.txt"

    if not seg_path.exists():
        raise FileNotFoundError(f"Missing seg: {seg_path}")
    if not depth_path.exists():
        raise FileNotFoundError(f"Missing depth: {depth_path}")
    if not camcfg_path.exists():
        raise FileNotFoundError(f"Missing cameraconfig: {camcfg_path}")

    seg = load_seg(seg_path)
    depth = load_depth(depth_path)
    bev_cam_height = load_camera_bev_height(str(camcfg_path))

    objects = segment_objects(seg, min_area=int(args.min_area))
    boxes_2d = get_2d_bounding_boxes(objects)

    height_map = compute_height_map(depth, bev_cam_height)
    cuboids = build_cuboids_from_2d_boxes(
        boxes_2d,
        height_map,
        resolution=float(args.resolution),
        offset=float(args.offset),
        yshift=float(args.yshift),
    )

    annotation = build_bev_3d_annotation(
        cuboids=cuboids,
        frame_id=int(args.id),
    )

    if args.dump_json:
        print(json.dumps(annotation, indent=2))

    if args.vis:
        cam_geoms = load_camera_geometries_for_debug(root) if args.vis_cam else []
        visualize_annotation(annotation, cam_geoms)


if __name__ == "__main__":
    main()
