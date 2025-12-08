#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
from pathlib import Path
import logging
import sys
from tqdm import tqdm 
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Ensure `src/` is on sys.path because `utils/` lives under `src/` in this repo
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

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
    level=logging.ERROR,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("VIDEO_GEN")

# ==========================================================
# CLEAN Y-PLANE DRAWER
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
    bottom_pts_cam = c[:4]
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
# HELPER: DRAW TEXT
# ==========================================================
def draw_text(img, text):
    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)
    return img

# ==========================================================
# PROCESS SINGLE FRAME
# ==========================================================
def process_frame(sample_id, root, args, K, D, xi, extrinsics_dict):
    """
    Xử lý 1 frame và trả về ảnh ghép (composite image)
    """
    bev_path = root / "seg" / "bev" / f"{sample_id}.png"
    depth_path = root / "depth" / f"{sample_id}.png"
    cfg_path = root / "cameraconfig" / f"{sample_id}.txt"

    if not bev_path.exists():
        return None

    # 1. Pipeline 3D
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
    
    # 2. Xử lý các view Camera
    cam_name_map = {
        "front": "Main Camera-front",
        "left":  "Main Camera-left",
        "right": "Main Camera-right",
        "rear":  "Main Camera-rear",
    }
    
    # Kích thước chuẩn để ghép ảnh (Width, Height)
    target_size = (640, 480) 
    processed_images = {}

    for view in ["front", "left", "right", "rear"]:
        img_path = root / "rgb" / view / f"{sample_id}.png"
        if not img_path.exists():
            img = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        else:
            img = cv2.imread(str(img_path))
            # Flip logic (giữ nguyên như code cũ)
            if view in ["front", "rear"]:
                img = cv2.flip(img, 1) # Flip ngang
                img = cv2.flip(img, 0) # Flip dọc

            ext_key = cam_name_map[view]
            if ext_key in extrinsics_dict:
                Extrinsic = extrinsics_dict[ext_key]
                img = draw_cuboids_curved(img, cuboids, Extrinsic, K, D, xi)
                if view == "front":
                    img = draw_y_planes_on_front(img, Extrinsic, K, D, xi, cuboids)

            # Flip lại để hiển thị
            if view in ["front", "rear"]:
                img = cv2.flip(img, 1)
                img = cv2.flip(img, 0)
        
        # Resize về chuẩn để ghép
        img = cv2.resize(img, target_size)
        draw_text(img, view.capitalize())
        processed_images[view] = img

    # 3. Xử lý Depth & BEV
    # Depth
    depth_viz = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    if depth_viz is None:
        depth_viz = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
    depth_viz = cv2.normalize(depth_viz, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_viz = cv2.cvtColor(depth_viz, cv2.COLOR_GRAY2BGR)
    depth_viz = cv2.resize(depth_viz, target_size)
    draw_text(depth_viz, "Depth")

    # BEV
    bev_viz = cv2.imread(str(bev_path))
    if bev_viz is None:
        bev_viz = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    else:
        bev_viz = cv2.cvtColor(bev_viz, cv2.COLOR_BGR2RGB) # Code cũ dùng RGB cho plt, ở đây dùng BGR cho cv2
        bev_viz = cv2.cvtColor(bev_viz, cv2.COLOR_RGB2BGR) # Convert lại BGR để write video
    
    bev_viz = cv2.resize(bev_viz, target_size)
    draw_text(bev_viz, f"BEV (ID: {sample_id})")

    # 4. Ghép ảnh (Grid 2x3)
    # Row 1: Left | Front | Right
    row1 = cv2.hconcat([processed_images["left"], processed_images["front"], processed_images["right"]])
    
    # Row 2: Depth | Rear | BEV
    row2 = cv2.hconcat([depth_viz, processed_images["rear"], bev_viz])

    # Final Grid
    grid = cv2.vconcat([row1, row2])
    return grid

# ==========================================================
# MAIN LOOP
# ==========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--output", default="output_viz.mp4", help="Output video path")
    ap.add_argument("--fps", type=int, default=10, help="Video FPS")
    
    ap.add_argument("--resolution", type=float, default=100/(6*400))
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--offset", type=float, default=30)
    ap.add_argument("--yshift", type=float, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    root = Path(args.dataset_root)
    
    # 1. Tìm tất cả các frame
    # Giả sử ID là tên file trong seg/bev (ví dụ: 0.png, 1.png, ...)
    bev_files = list((root / "seg" / "bev").glob("*.png"))
    if not bev_files:
        print("Không tìm thấy dữ liệu trong seg/bev!")
        return

    # Sắp xếp ID theo số (0, 1, 2... thay vì 0, 1, 10, 11)
    try:
        sample_ids = sorted([p.stem for p in bev_files], key=lambda x: int(x))
    except ValueError:
        sample_ids = sorted([p.stem for p in bev_files]) # Fallback nếu tên không phải số

    print(f"Tìm thấy {len(sample_ids)} frames. Đang chuẩn bị tạo video...")

    # 2. Load Config Global
    K, D, xi = load_intrinsics(Path(args.intrinsics))
    config_path = root.parent.parent / "CameraCalibrationParameters" / "camera_positions_for_extrinsics.txt"
    extrinsics_dict = load_extrinsics(config_path)

    video_writer = None

    # 3. Vòng lặp xử lý
    for sample_id in tqdm(sample_ids):
        frame = process_frame(sample_id, root, args, K, D, xi, extrinsics_dict)
        
        if frame is None:
            continue

        # Khởi tạo VideoWriter tại frame đầu tiên (vì giờ mới biết kích thước ảnh)
        if video_writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Hoặc 'XVID' nếu lỗi
            video_writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))
            print(f"Video Resolution: {w}x{h}")

        video_writer.write(frame)

    if video_writer:
        video_writer.release()
        print(f"\n✅ Video đã được lưu tại: {args.output}")
    else:
        print("❌ Không tạo được video (có thể không load được frame nào).")

if __name__ == "__main__":
    main()