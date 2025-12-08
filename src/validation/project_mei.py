#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys

from utils.bbox3d import draw_cuboids_curved
from utils.camera import load_intrinsics, load_extrinsics, load_camera_bev_height
from utils.projects import cam2image
from utils.pipeline import (
    load_depth,
    compute_height_map,
    segment_objects,
    get_2d_bounding_boxes,
    get_3d_bounding_boxes,
)

logging.basicConfig(
    level=logging.ERROR,   # <-- only show real errors
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("CLEAN")


# ==========================================================
# CLEAN Y-PLANE DRAWER (no logging)
# ==========================================================
def draw_y_planes_on_front(img, extrinsic, K, D, xi, cuboids):
    if img is None or img.size == 0 or not cuboids:
        return img

    try:
        cam2world = np.linalg.inv(extrinsic)
    except:
        return img

    cam_center_world = (cam2world @ np.array([0,0,0,1], dtype=float).reshape(4,1)).flatten()
    cam_x, cam_y, cam_z = cam_center_world[:3]

    # Find cuboid bottom plane Y
    c = cuboids[0]["corners"]
    bottom_pts_cam = c[:4  ]
    pts_world = (cam2world @ np.hstack([bottom_pts_cam, np.ones((4,1))]).T).T
    y_world_mean = float(pts_world[:, 1].mean())

    # Create ground grids
    xs = np.linspace(cam_x - 20, cam_x + 20, 41)
    zs = np.linspace(cam_z + 2, cam_z + 40, 40)

    X, Z = np.meshgrid(xs, zs)

    # Plane 1: Ground Y=0
    Y0 = np.zeros_like(X)
    P0 = np.stack([X.ravel(), Y0.ravel(), Z.ravel()], axis=1)
    uv0, m0 = cam2image(P0, extrinsic, K, D, xi)
    uv0 = uv0[m0].astype(int)

    for (u, v) in uv0:
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.circle(img, (u, v), 1, (0, 0, 255), -1)

    # Plane 2: Cuboid bottom plane
    Y1 = np.full_like(X, y_world_mean)
    P1 = np.stack([X.ravel(), Y1.ravel(), Z.ravel()], axis=1)
    uv1, m1 = cam2image(P1, extrinsic, K, D, xi)
    uv1 = uv1[m1].astype(int)

    for (u, v) in uv1:
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.circle(img, (u, v), 1, (255, 0, 0), -1)

    return img


# ==========================================================
# CLEAN MAIN (no logging)
# ==========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--resolution", type=float, default=100/(6*400))
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--offset", type=float, default=30)
    ap.add_argument("--yshift", type=float, default=0)
    args = ap.parse_args()

    root = Path(args.dataset_root)
    sample_id = args.id

    bev_path = root / "seg" / "bev" / f"{sample_id}.png"
    depth_path = root / "depth" / f"{sample_id}.png"
    cfg_path = root / "cameraconfig" / f"{sample_id}.txt"

    if not bev_path.exists():
        logger.error(f"Missing BEV mask: {bev_path}")
        return

    bev_mask = (load_depth(str(bev_path)) > 0).astype("uint8") * 255
    obj_masks = segment_objects(bev_mask, min_area=args.min_area)
    depth_norm = load_depth(str(depth_path))

    cam_h = None
    if cfg_path.exists():
        try:
            cam_h = load_camera_bev_height(str(cfg_path))
        except:
            cam_h = None

    height_map = compute_height_map(depth_norm, cam_h)
    boxes_2d = get_2d_bounding_boxes(obj_masks)

    cuboids = get_3d_bounding_boxes(
        boxes_2d, height_map, args.resolution,
        args.offset, args.yshift
    )

    K, D, xi = load_intrinsics(Path(args.intrinsics))
    config_path = root.parent.parent / "CameraCalibrationParameters" / "camera_positions_for_extrinsics.txt"
    extrinsics_dict = load_extrinsics(config_path)

    cam_name_map = {
        "front": "Main Camera-front",
        "left":  "Main Camera-left",
        "right": "Main Camera-right",
        "rear":  "Main Camera-rear",
    }

    processed_images = {}

    for view in ["front", "left", "right", "rear"]:
        img_path = root / "rgb" / view / f"{sample_id}.png"
        if not img_path.exists():
            processed_images[view] = np.zeros((600,800,3), dtype=np.uint8)
            continue

        img = cv2.imread(str(img_path))
        if view in ["front", "rear"]:
            img = cv2.flip(img, 1)
            img = cv2.flip(img, 0)
    
        ext_key = cam_name_map[view]
        if ext_key in extrinsics_dict:

            Extrinsic = extrinsics_dict[ext_key]
            img = draw_cuboids_curved(img, cuboids, Extrinsic, K, D, xi)

            if view == "front":
                img = draw_y_planes_on_front(img, Extrinsic, K, D, xi, cuboids)

        if view in ["front", "rear"]:
            img = cv2.flip(img, 1)
            img = cv2.flip(img, 0)

        processed_images[view] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    depth_viz = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    if depth_viz is None:
        depth_viz = np.zeros((100,100), dtype=np.uint8)

    depth_viz = cv2.normalize(depth_viz, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_viz = cv2.cvtColor(depth_viz, cv2.COLOR_GRAY2RGB)

    bev_viz = cv2.imread(str(bev_path))
    if bev_viz is not None:
        bev_viz = cv2.cvtColor(bev_viz, cv2.COLOR_BGR2RGB)
    else:
        bev_viz = np.zeros((100,100,3), dtype=np.uint8)

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Offset={args.offset}, YShift={args.yshift}", fontsize=16, color='red')

    axes = [
        ("Left", processed_images["left"]),
        ("Front", processed_images["front"]),
        ("Right", processed_images["right"]),
        ("Depth", depth_viz),
        ("Rear", processed_images["rear"]),
        ("BEV", bev_viz),
    ]

    grid = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
    for (title, img), (r,c) in zip(axes, grid):
        axs[r,c].imshow(img)
        axs[r,c].set_title(title)
        axs[r,c].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
