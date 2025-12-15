import numpy as np
import open3d as o3d
from typing import Tuple, Dict, List
import cv2

from .projects import cam2image

# 3D cuboid construction and rendering helpers built on top of 2D OBBs.

# ============================================================
# SHIFT OBB TO AXLE (NO LOG)
# ============================================================
def shift2axle(obb: Dict, image_shape: Tuple[int, int], resolution: float, offset: float) -> Dict:
    (cx, cy) = obb["center"]
    (w_px, l_px) = obb["size"]
    yaw_deg = obb["angle"]

    H_img, W_img = image_shape

    Xc = (cx - W_img / 2.0) * resolution
    Zc = (H_img / 2.0-cy) * resolution + offset*resolution

    return {"center": (Xc, Zc), "size": (w_px, l_px), "angle": -yaw_deg}


# ============================================================
# CUBOID WIREFRAME LINES
# ============================================================
CUBOID_LINES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
]


# ============================================================
# PRIMARY FUNCTION: CUBOID CORNERS (CLEAN VERSION)
# ============================================================
def cuboid_corners(
    obb: Dict[str, object],
    image_shape: Tuple[int, int],
    resolution: float,
    min_height: float,
    max_height: float,
    offset: float = 1.45,
    yshift: float = -0.3
) -> np.ndarray:

    obb_shifted = shift2axle(obb, image_shape, resolution, offset)

    (Xc, Zc) = obb_shifted["center"]
    (w_px, l_px) = obb_shifted["size"]
    yaw_deg = obb_shifted["angle"]

    hw = (w_px * resolution) / 2.0
    hl = (l_px * resolution) / 2.0

    base = np.array([
        [ hw, 0.0,  hl],
        [ hw, 0.0, -hl],
        [-hw, 0.0, -hl],
        [-hw, 0.0,  hl]
    ], dtype=np.float64)

    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

    rot = (R @ base.T).T
    rot[:, 0] += Xc
    rot[:, 2] += Zc

    bottom_y = min_height + yshift
    top_y = max_height + yshift

    bottom = rot.copy()
    bottom[:, 1] = bottom_y

    top = rot.copy()
    top[:, 1] = top_y

    return np.vstack([bottom, top])


# ============================================================
# BUILD OPEN3D LINESET (CLEAN)
# ============================================================
def build_cuboid(corners: np.ndarray, color=(1.0, 1.0, 0.0)) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners.astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector(CUBOID_LINES)
    ls.colors = o3d.utility.Vector3dVector([color for _ in CUBOID_LINES])
    return ls


# ============================================================
# DRAW CUBOIDS WITH CURVED EDGES ON IMAGE (CLEAN)
# ============================================================
def draw_cuboids_curved(img, cuboids_list, Matrix, K, D, xi, segments=20):
    """
    cuboids_list: list of dicts, each expected to have keys "corners" and "color"
    """
    vis = img.copy()
    h, w = img.shape[:2]

    EDGES = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for obj in cuboids_list:
        # 1. Extract geometry and color
        corners = obj["corners"]
        
        # Default to yellow if no per-object color is provided
        rgb = obj.get("color", [1.0, 1.0, 0.0]) 

        # 2. Convert RGB floats [0,1] to BGR ints [0,255] for OpenCV drawing
        r = int(rgb[0] * 255)
        g = int(rgb[1] * 255)
        b = int(rgb[2] * 255)
        color_bgr = (b, g, r)

        for s, e in EDGES:
            p1, p2 = corners[s], corners[e]
            t = np.linspace(0, 1, segments).reshape(-1, 1)
            pts3d = p1 + (p2 - p1) * t

            uv, mask = cam2image(pts3d, Matrix, K, D, xi)
            
            # Skip edges without enough valid projections
            if np.sum(mask) < 2:
                continue

            pts = uv[mask].astype(np.int32)
            
            # Check if any projected points fall inside the image bounds
            in_bound = np.all((pts >= 0) & (pts < [w, h]), axis=1)

            if np.any(in_bound):
                # 3. Draw polylines with the converted color
                cv2.polylines(vis, [pts.reshape(-1, 1, 2)], False, color_bgr, 2, cv2.LINE_AA)

    return vis


# ============================================================
# 3D VISUALIZER (POINT CLOUD + CUBOIDS)
# ============================================================
def draw_3d_cuboids_with_pcd(pcd, cuboids: List[Dict]):
    geoms = [pcd]

    for c in cuboids:
        if "lineset" in c:
            geoms.append(c["lineset"])
        else:
            geoms.append(build_cuboid(c["corners"]))

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    geoms.append(axis)

    o3d.visualization.draw_geometries(geoms)


# ============================================================
# LOCAL TEST (NO LOG)
# ============================================================
if __name__ == "__main__":
    image_shape = (600, 600)
    resolution = 0.05
    offset = 1.0

    center_test = {"center": (300, 300), "size": (40, 80), "angle": 0}

    shifted = shift2axle(center_test, image_shape, resolution, offset)
    print("Shift test output:", shifted)
