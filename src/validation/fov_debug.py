#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fov_debug_bev_world_o3d_and_project.py

Create a 3D WALL from a BEV-space (600x600) black canvas (BEV grid),
then:
  (1) visualize the 3D world in Open3D
  (2) project the SAME 3D wall onto FRONT + REAR camera images (MEI)

Conventions match your repo:
- Extrinsic: WORLD -> CAMERA (loaded by utils.camera.load_extrinsics)  :contentReference[oaicite:3]{index=3}
- Projection: utils.projects.cam2image (Unified MEI)                   :contentReference[oaicite:4]{index=4}
- Flip: DISPLAY ONLY (after drawing), like your project_mei.py style
"""

import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import cv2
import open3d as o3d
import sys

# ============================================================
# Path setup (same style as your repo)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.camera import load_intrinsics, load_extrinsics  # :contentReference[oaicite:5]{index=5}
from utils.projects import cam2image                        # :contentReference[oaicite:6]{index=6}


# ============================================================
# Display flip (same semantic as your project_mei.py)
# ============================================================
def flip_for_display(img_bgr: np.ndarray, view: str) -> np.ndarray:
    if view in ["front", "rear"]:
        img_bgr = cv2.flip(img_bgr, 1)
        img_bgr = cv2.flip(img_bgr, 0)
    return img_bgr


# ============================================================
# BEV -> WORLD mapping (same math as bbox3d.shift2axle()) :contentReference[oaicite:7]{index=7}
# ============================================================
def bev_px_to_world_xz(
    u: np.ndarray,
    v: np.ndarray,
    bev_shape_hw: Tuple[int, int],
    resolution: float,
    offset: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert BEV pixel (u=x, v=y) into WORLD (X,Z) using your convention:

      X = (u - W/2) * resolution
      Z = (H/2 - v) * resolution + offset*resolution

    This is exactly the same mapping used by shift2axle() in bbox3d.py. :contentReference[oaicite:8]{index=8}
    """
    H, W = bev_shape_hw
    X = (u - (W / 2.0)) * resolution
    Z = ((H / 2.0) - v) * resolution + (offset * resolution)
    return X, Z


# ============================================================
# Build WALL from BEV canvas (600x600)
# ============================================================
def build_wall_from_bev_canvas(
    bev_shape_hw: Tuple[int, int],
    resolution: float,
    offset: float,
    # wall intent
    dist: float,
    x_left_m: float,
    height_m: float,
    # density
    n_z: int,
    n_y: int,
    thickness_m: float,
    n_x: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create ONE wall as a 3D point cloud, but generated "from BEV-space":
    - pick BEV pixels that correspond to desired world X=-x_left and Z in [-dist, +dist]
    - convert those BEV pixels -> (X,Z) using your BEV->WORLD mapping
    - extrude along Y [0..height]

    Returns:
      pts_world: (N,3)
      colors_rgb: (N,3) float in [0,1]
    """
    H, W = bev_shape_hw

    # ---- Convert desired WORLD ranges into BEV pixel ranges ----
    # Invert mapping for u (x pixel):
    # X = (u - W/2)*res -> u = X/res + W/2
    u_center = ( (-x_left_m) / resolution ) + (W / 2.0)

    if n_x <= 1:
        us = np.array([u_center], dtype=np.float64)
    else:
        # thickness in meters -> thickness in pixels
        thick_px = thickness_m / resolution
        us = np.linspace(u_center - thick_px / 2.0, u_center + thick_px / 2.0, n_x)

    # Invert mapping for v (y pixel):
    # Z = (H/2 - v)*res + offset*res
    # => (H/2 - v) = (Z - offset*res)/res
    # => v = H/2 - (Z/res - offset)
    zs = np.linspace(-dist, +dist, n_z, dtype=np.float64)
    vs = (H / 2.0) - (zs / resolution - offset)  # v pixels for each desired z

    ys = np.linspace(0.0, height_m, n_y, dtype=np.float64)

    pts = []
    cols = []

    # color by Z (blue->red)
    for zi, (z, v) in enumerate(zip(zs, vs)):
        t = (z + dist) / (2.0 * dist + 1e-9)  # 0..1
        color = np.array([t, 0.2, 1.0 - t], dtype=np.float64)

        for u in us:
            # clamp to image to keep mapping stable
            u_c = float(np.clip(u, 0, W - 1))
            v_c = float(np.clip(v, 0, H - 1))

            # BEV px -> WORLD XZ (using your mapping)
            X, Z = bev_px_to_world_xz(
                np.array([u_c]), np.array([v_c]),
                (H, W), resolution, offset
            )
            X = float(X[0])
            Z = float(Z[0])

            for y in ys:
                pts.append([X, float(y), Z])
                cols.append(color)

    return np.asarray(pts, dtype=np.float64), np.asarray(cols, dtype=np.float64)


# ============================================================
# Camera helpers (WORLD axis/center for Open3D + overlay)
# ============================================================
def camera_center_world(ext_w2c: np.ndarray) -> np.ndarray:
    Pc = np.array([[0, 0, 0, 1.0]], dtype=np.float64)
    C2W = np.linalg.inv(ext_w2c)
    Pw = (C2W @ Pc.T).T[:, :3]
    return Pw[0]

def camera_ox_world(ext_w2c: np.ndarray, dist: float) -> np.ndarray:
    t = np.linspace(-dist, +dist, 200, dtype=np.float64)
    Pc = np.zeros((t.shape[0], 4), dtype=np.float64)
    Pc[:, 0] = t
    Pc[:, 3] = 1.0
    C2W = np.linalg.inv(ext_w2c)
    Pw = (C2W @ Pc.T).T[:, :3]
    return Pw


# ============================================================
# Projection + drawing (MEI)
# ============================================================
def draw_points_mei(img_bgr, pts_world, colors_bgr_u8, ext_w2c, K, D, xi, radius=2):
    H, W = img_bgr.shape[:2]
    uv, valid = cam2image(pts_world, ext_w2c, K, D, xi, image_size=(W, H))  # :contentReference[oaicite:9]{index=9}

    if valid is None:
        valid = np.ones((pts_world.shape[0],), dtype=bool)

    pts2d = np.round(uv[valid]).astype(np.int32)
    cols = colors_bgr_u8[valid]

    inside = (
        (pts2d[:, 0] >= 0) & (pts2d[:, 0] < W) &
        (pts2d[:, 1] >= 0) & (pts2d[:, 1] < H)
    )
    pts2d = pts2d[inside]
    cols = cols[inside]

    out = img_bgr.copy()
    for (u, v), c in zip(pts2d, cols):
        cv2.circle(out, (int(u), int(v)), radius, (int(c[0]), int(c[1]), int(c[2])), -1, cv2.LINE_AA)
    return out

def draw_polyline_mei(img_bgr, pts_world, ext_w2c, K, D, xi, color_bgr, thickness=2):
    H, W = img_bgr.shape[:2]
    uv, valid = cam2image(pts_world, ext_w2c, K, D, xi, image_size=(W, H))  # :contentReference[oaicite:10]{index=10}
    if valid is None:
        valid = np.ones((pts_world.shape[0],), dtype=bool)

    pts2d = np.round(uv[valid]).astype(np.int32)
    inside = (
        (pts2d[:, 0] >= 0) & (pts2d[:, 0] < W) &
        (pts2d[:, 1] >= 0) & (pts2d[:, 1] < H)
    )
    pts2d = pts2d[inside]
    out = img_bgr.copy()
    if pts2d.shape[0] >= 2:
        cv2.polylines(out, [pts2d.reshape(-1, 1, 2)], False, color_bgr, thickness, cv2.LINE_AA)
    return out


# ============================================================
# Open3D builders
# ============================================================
def o3d_wall_pointcloud(pts_world: np.ndarray, colors_rgb: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors_rgb, 0.0, 1.0))
    return pcd

def o3d_point(pos_xyz: np.ndarray, color_rgb=(1.0, 0.0, 0.0)) -> o3d.geometry.TriangleMesh:
    sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
    sph.translate(pos_xyz.astype(np.float64))
    sph.paint_uniform_color(list(color_rgb))
    return sph

def o3d_lineset(points_xyz: np.ndarray, color_rgb=(255, 1.0, 1.0)) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector([[0, len(points_xyz)-1]])
    ls.colors = o3d.utility.Vector3dVector([list(color_rgb)])
    return ls

def o3d_polyline(points_xyz: np.ndarray, color_rgb=(1.0, 1.0, 1.0)) -> o3d.geometry.LineSet:
    lines = [[i, i+1] for i in range(len(points_xyz)-1)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([list(color_rgb) for _ in lines])
    return ls


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--extrinsics", required=True)

    ap.add_argument("--front-img", required=True)
    ap.add_argument("--rear-img", required=True)

    ap.add_argument("--front-cam", default="Main Camera-front")
    ap.add_argument("--rear-cam", default="Main Camera-rear")

    # BEV canvas params
    ap.add_argument("--bev-size", type=int, default=600)
    ap.add_argument("--resolution", type=float, default=100 / (6 * 400))  # same default you use often
    ap.add_argument("--offset", type=float, default=36.0)

    # wall intent params
    ap.add_argument("--dist", type=float, default=10.0)
    ap.add_argument("--x-left", type=float, default=8.0)
    ap.add_argument("--height", type=float, default=3.5)

    # wall density params
    ap.add_argument("--n-z", type=int, default=300)
    ap.add_argument("--n-y", type=int, default=120)
    ap.add_argument("--wall-thickness", type=float, default=0.05)
    ap.add_argument("--n-x", type=int, default=1)

    ap.add_argument("--show-black", action="store_true", help="Also show black canvas projection")
    args = ap.parse_args()

    # ---- Create a black BEV canvas (600x600) ----
    bev_hw = (args.bev_size, args.bev_size)
    bev_black = np.zeros((args.bev_size, args.bev_size, 3), dtype=np.uint8)

    # ---- Load calib ----
    K, D, xi = load_intrinsics(args.intrinsics)  # :contentReference[oaicite:11]{index=11}
    extr = load_extrinsics(args.extrinsics)      # :contentReference[oaicite:12]{index=12}

    if args.front_cam not in extr or args.rear_cam not in extr:
        raise KeyError(f"Missing camera in extrinsics. Have keys: {list(extr.keys())[:10]} ...")

    E_front = extr[args.front_cam]
    E_rear  = extr[args.rear_cam]

    # ---- Build ONE WALL from BEV-space -> WORLD ----
    wall_world, wall_colors_rgb = build_wall_from_bev_canvas(
        bev_shape_hw=bev_hw,
        resolution=float(args.resolution),
        offset=float(args.offset),
        dist=float(args.dist),
        x_left_m=float(args.x_left),
        height_m=float(args.height),
        n_z=int(args.n_z),
        n_y=int(args.n_y),
        thickness_m=float(args.wall_thickness),
        n_x=int(args.n_x),
    )

    # ---- Open3D visualize WORLD ----
    geoms = []
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0]))
    geoms.append(o3d_wall_pointcloud(wall_world, wall_colors_rgb))
    geoms.append(o3d_lineset(
        np.array([[-args.x_left, 0.0, -args.dist],
                  [-args.x_left, 0.0, +args.dist]], dtype=np.float64),
        color_rgb=(0.5, 0.5, 0.5)
    ))

    # camera centers + Ox polylines in WORLD
    for view, E in [("front", E_front), ("rear", E_rear)]:
        c = camera_center_world(E)
        ox = camera_ox_world(E, dist=float(args.dist))
        geoms.append(o3d_point(c, color_rgb=(1.0, 0.0, 0.0)))        # red sphere
        geoms.append(o3d_polyline(ox, color_rgb=(1.0, 1.0, 1.0)))    # white polyline

    # show 3D (non-blocking style is not standard; Open3D blocks until closed)
    o3d.visualization.draw_geometries(
        geoms,
        window_name="WORLD SPACE (wall + cameras)",
        width=1400,
        height=900,
    )

    # ---- Project wall onto FRONT & REAR images ----
    img_front = cv2.imread(str(Path(args.front_img)))
    img_rear  = cv2.imread(str(Path(args.rear_img)))
    if img_front is None or img_rear is None:
        raise FileNotFoundError("Cannot read front/rear images.")

    # wall colors for OpenCV (BGR uint8)
    wall_colors_bgr = (wall_colors_rgb[:, ::-1] * 255.0).astype(np.uint8)

    out_f = draw_points_mei(img_front, wall_world, wall_colors_bgr, E_front, K, D, xi, radius=2)
    out_f = draw_polyline_mei(out_f, camera_ox_world(E_front, float(args.dist)), E_front, K, D, xi, (255, 255, 255), 2)
    # draw camera center projection as red point
    cf = camera_center_world(E_front)[None, :]
    out_f = draw_points_mei(out_f, cf, np.array([[0, 0, 255]], dtype=np.uint8), E_front, K, D, xi, radius=7)

    out_r = draw_points_mei(img_rear, wall_world, wall_colors_bgr, E_rear, K, D, xi, radius=2)
    out_r = draw_polyline_mei(out_r, camera_ox_world(E_rear, float(args.dist)), E_rear, K, D, xi, (255, 255, 255), 2)
    cr = camera_center_world(E_rear)[None, :]
    out_r = draw_points_mei(out_r, cr, np.array([[0, 0, 255]], dtype=np.uint8), E_rear, K, D, xi, radius=7)

    # display-only flip
    cv2.imshow("FRONT projection", flip_for_display(out_f, "front"))
    cv2.imshow("REAR  projection",  flip_for_display(out_r, "rear"))

    if args.show_black:
        black_f = np.zeros_like(img_front)
        black_r = np.zeros_like(img_rear)

        bf = draw_points_mei(black_f, wall_world, wall_colors_bgr, E_front, K, D, xi, radius=2)
        bf = draw_polyline_mei(bf, camera_ox_world(E_front, float(args.dist)), E_front, K, D, xi, (255, 255, 255), 2)
        bf = draw_points_mei(bf, cf, np.array([[0, 0, 255]], dtype=np.uint8), E_front, K, D, xi, radius=7)

        br = draw_points_mei(black_r, wall_world, wall_colors_bgr, E_rear, K, D, xi, radius=2)
        br = draw_polyline_mei(br, camera_ox_world(E_rear, float(args.dist)), E_rear, K, D, xi, (255, 255, 255), 2)
        br = draw_points_mei(br, cr, np.array([[0, 0, 255]], dtype=np.uint8), E_rear, K, D, xi, radius=7)

        cv2.imshow("FRONT projection (black)", flip_for_display(bf, "front"))
        cv2.imshow("REAR  projection (black)",  flip_for_display(br, "rear"))

    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord("q")):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
