#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import sys
from typing import List, Dict
from pathlib import Path
import open3d as o3d

# Import utils
from utils.bbox2d import compute_2d_boxes
from utils.bbox3d import cuboid_corners, build_cuboid
from utils.camera import load_camera_bev_height, load_extrinsics
np.set_printoptions(precision=3, suppress=True)


# ============================================================
# DATA PROCESSING
# ============================================================
def load_depth(path: str) -> np.ndarray:
    depth = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if depth is None:
        raise FileNotFoundError(f"Cannot read depth: {path}")
    return depth.astype(np.float32) / 255.0

def load_seg(path: str) -> np.ndarray:
    """
    Load segmentation image exactly as RGB array (H, W, 3),
    preserving all dataset class colors.

    Returns: RGB ndarray, dtype uint8
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read segmentation image: {path}")

    # convert from BGR → RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def segment_objects(seg_img: np.ndarray, min_area: int = 5, palette=None):
    """
    seg_img can be:
        - BEV mask (grayscale): pixel>0 means object
        - segmentation image (RGB multi-class)
    """
    # # Case 1: BEV binary mask
    # if seg_img.ndim == 2:
    #     mask = (seg_img > 0).astype(np.uint8)
    #     num, labels = cv2.connectedComponents(mask)
    #     objects = []
    #     for idx in range(1, num):
    #         comp = (labels == idx).astype(np.uint8)
    #         if comp.sum() < min_area:
    #             continue
    #         objects.append({
    #             "mask": comp * 255,
    #             "label": 1,
    #             "color": [1.0, 1.0, 1.0]   # default white
    #         })
    #     return objects

    # Case 2: segmentation RGB image
    seg_rgb = seg_img
    label_img = (
        seg_rgb[:, :, 0].astype(np.int32) * 256*256 +
        seg_rgb[:, :, 1].astype(np.int32) * 256 +
        seg_rgb[:, :, 2].astype(np.int32)
    )

    objects = []
    for val in np.unique(label_img):
        if val == 0:
            continue

        class_mask = (label_img == val).astype(np.uint8)
        num, labels = cv2.connectedComponents(class_mask)

        for idx in range(1, num):
            comp = (labels == idx).astype(np.uint8)
            if comp.sum() < min_area:
                continue

            rgb = np.mean(seg_rgb[labels == idx], axis=0) / 255.0
            objects.append({
                "mask": comp * 255,
                "label": int(val),
                "color": [float(c) for c in rgb],
            })
    return objects




def get_2d_bounding_boxes(obj_masks):
    # Allow downstream functions to pass along metadata (color/label) per object
    raw_masks = [o["mask"] if isinstance(o, dict) else o for o in obj_masks]
    boxes = compute_2d_boxes(raw_masks)
    for box, o in zip(boxes, obj_masks):
        if isinstance(o, dict):
            if "color" in o:
                box["color"] = o["color"]
            if "label" in o:
                box["label"] = o["label"]
    return boxes




def compute_height_map(depth_norm: np.ndarray, bev_cam_height: float,
                       min_height: float = 0.1) -> np.ndarray:
    """
    depth_norm: ảnh depth đã chuẩn hóa [0,1] từ BEV depth renderer.
    bev_cam_height: posi_y của BEVCamera (từ file cameraconfig).
    
    Giả định:
        d = 0   → điểm nằm trên mặt đất (Y ≈ 0)
        d = 1   → điểm ở vị trí cao bằng BEV camera (Y ≈ bev_cam_height)
    """
    d = np.clip(depth_norm, 0.0, 1.0)

    # Chiều cao thực tế (tương đối so với mặt đất Y=0)
    height = (1-d) * float(bev_cam_height) /9

    # Lọc noise: mọi thứ thấp hơn min_height coi như sát mặt đất → 0
    height[height < min_height] = 0.0
    return height



# ============================================================
# CLEAN 4-CLASS CLASSIFICATION
# ============================================================
def get_3d_bounding_boxes(
    boxes_2d,
    height_map,
    resolution,
    offset: float = 33,
    yshift: float = -0.3,
):
    H, W = height_map.shape
    cuboids = []


    for box in boxes_2d:
        mask_bool = box["mask"].astype(bool)
        color = box['color']*255

        vals = height_map[mask_bool]
        h_max = float(np.max(vals))

        if h_max >= 3.0:
            final_h = h_max
            color = color

        elif h_max >= 2.4:
            final_h = h_max / 2
            color = color

        elif h_max >= 2.3:
            final_h = h_max / 2.2
            color = color  

        else:
            final_h = h_max / 3.0
            color = color

        # ====== Xây cuboid ======
        corners = cuboid_corners(
            box["obb"],
            (H, W),
            resolution,
            min_height=0.0,
            max_height=final_h,
            offset=offset,
            yshift=yshift,
        )

        cuboids.append({
            "obb": box["obb"],
            "corners": corners,
            "lineset": build_cuboid(corners),
            "color": color,
            "h_max": h_max,
        })

    return cuboids


# ============================================================
# CAMERA VISUALIZATION
# ============================================================
def create_camera_visuals(extrinsic_matrix, color=[1, 0, 0], size=1.0):
    try:
        cam_pose = np.linalg.inv(extrinsic_matrix)
    except:
        return []

    geoms = []

    w, h, z = size * 0.8, size * 0.6, size
    points_cam = np.array([
        [0,0,0],
        [-w,-h,z], [w,-h,z],
        [w,h,z],  [-w,h,z]
    ])
    ones = np.ones((5,1))
    points_world = (cam_pose @ np.hstack([points_cam, ones]).T).T[:, :3]

    lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points_world)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.paint_uniform_color(color)
    geoms.append(ls)

    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.05*size, cone_radius=0.15*size,
        cylinder_height=1.0*size, cone_height=0.3*size
    )
    arrow.paint_uniform_color(color)
    arrow.transform(cam_pose)
    geoms.append(arrow)

    return geoms


def draw_3d_scene(cuboids, camera_geoms_list):
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0,0,0])
    geoms = [axis]

    for c in cuboids:
        ls = c["lineset"]
        ls.paint_uniform_color(c["color"])
        geoms.append(ls)

    geoms.extend(camera_geoms_list)
    o3d.visualization.draw_geometries(geoms, window_name="3D Viewer", width=1280, height=720)


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--resolution", type=float, default=100 / (6 * 400))
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--vis", action="store_true")
    ap.add_argument("--offset", type=float, default=30)
    ap.add_argument("--yshift", type=float, default=0)
    args = ap.parse_args()

    root = Path(args.dataset_root)
    bev_path = root / "seg" / "bev" / f"{args.id}.png"
    depth_path = root / "depth" / f"{args.id}.png"
    bev_height_path = root / "cameraconfig" / f"{args.id}.txt"
    parent_dir = root.parent.parent
    config_path = parent_dir / "CameraCalibrationParameters" / "camera_positions_for_extrinsics.txt"

    if not bev_path.exists():
        print("Error: BEV not found")
        return

    bev_seg = load_seg(str(bev_path))
    obj_masks = segment_objects(bev_seg, min_area=args.min_area)


    depth_norm = load_depth(str(depth_path))
    bev_cam_height = load_camera_bev_height(str(bev_height_path))
    height_map = compute_height_map(depth_norm, bev_cam_height)

    boxes_2d = get_2d_bounding_boxes(obj_masks)
    cuboids = get_3d_bounding_boxes(
        boxes_2d, height_map, args.resolution,
        offset=args.offset, yshift=args.yshift
    )

    camera_geoms = []
    if config_path.exists():
        extrinsics_dict = load_extrinsics(config_path)
        cam_colors = {
            "Main Camera-front": [1, 0, 0],
            "Main Camera-left":  [0, 1, 0],
            "Main Camera-right": [0, 0, 1],
            "Main Camera-rear":  [1, 1, 0],
        }
        for cam_name, color in cam_colors.items():
            if cam_name in extrinsics_dict:
                ext = extrinsics_dict[cam_name]
                if isinstance(ext, dict):
                    M = np.eye(4)
                    M[:3,:3] = np.array(ext["R"])
                    M[:3,3] = np.array(ext["t"]).flatten()
                    ext = M
                camera_geoms.extend(create_camera_visuals(ext, color, size=1.0))

    if args.vis:
        draw_3d_scene(cuboids, camera_geoms)


if __name__ == "__main__":
    main()
