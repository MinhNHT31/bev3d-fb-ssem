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


def segment_objects(mask_img: np.ndarray, min_area: int = 5) -> List[np.ndarray]:
    mask = (mask_img > 0).astype(np.uint8)
    num, labels = cv2.connectedComponents(mask)
    objs = []
    for idx in range(1, num):
        comp = (labels == idx).astype(np.uint8) * 255
        if cv2.countNonZero(comp) >= min_area:
            objs.append(comp)
    return objs


def get_2d_bounding_boxes(obj_masks: List[np.ndarray]) -> List[Dict]:
    return compute_2d_boxes(obj_masks)


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
    offset: float = 1.01,
    yshift: float = 0,
):
    H, W = height_map.shape
    cuboids = []

    for box in boxes_2d:
        mask_bool = box["mask"].astype(bool)
        if mask_bool.sum() < 15:
            continue

        vals = height_map[mask_bool]
        h_max = float(np.max(vals))

        # classification thresholds
        if h_max >= 3.0:       # BUS / CONTAINER
            final_h = h_max
            # Màu ĐỎ CAM (Rất nổi bật)
            color = [1.0, 0.2, 0.0] 
            
        elif h_max >= 2.7:    # LARGE CAR / TRUCK
            final_h = h_max / 1.6
            # Màu XANH LƠ (CYAN) - Tương phản tốt với đường nhựa
            color = [0.0, 1.0, 1.0]
            
        elif h_max >= 2.4:    # EV / SEDAN
            final_h = h_max / 2
            # Màu XANH LÁ MẠ (LIME) - Dễ nhìn
            color = [0.0, 1.0, 0.0]
            
        else:                  # NOISE / LOW OBSTACLES
            final_h = h_max / 2.5
            # Màu VÀNG (Yellow) hoặc TÍM (Magenta)
            color = [1.0, 1.0, 0.0]

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
            "color": color
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
DEFAULT_RES = 100 / (6 * 400)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--resolution", type=float, default=DEFAULT_RES)
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

    bev_mask = (load_depth(str(bev_path)) > 0).astype("uint8") * 255
    obj_masks = segment_objects(bev_mask, args.min_area)

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
