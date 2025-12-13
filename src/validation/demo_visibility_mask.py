#!/usr/bin/env python3
"""
Visibility Mask Generator (v3) – Occlusion-aware BEV mask

Mục tiêu:
    - Từ BEV segmentation + depth + 4 camera seg (front/left/right/rear),
      sinh ra 1 BEV mask với:
        0 = background
        1 = visible object
        2 = occluded object

Logic (theo mô tả của bạn):
    1. Từ BEV seg:
        - Tách từng object (instance) bằng segment_objects (dựa trên màu seg)
        - Mỗi object có: mask, color (RGB in [0,1]), label_id flatten

    2. Từ depth + camera height:
        - Tạo height_map trên BEV
        - Từ mask + height_map → get_2d_bounding_boxes → get_3d_bounding_boxes
        - Mỗi object BEV ↔ 1 cuboid 3D (corners trong world coords)

    3. Occlusion test cho MỖI object:
        - Sample các điểm trên bề mặt cuboid (6 mặt)
        - Dùng cam2image + intrinsics + extrinsics để project vào 4 camera
        - Với mỗi điểm projected (u,v), lấy pixel từ camera seg:
              seg_color = seg_img[v, u] (RGB, 0–255)
        - So sánh seg_color với màu của object trên BEV:
              obj_color = obj["color"] (0–1)  → scale lên 0–255
              nếu |seg_color - obj_color| < color_tol → coi như cùng object
        - Một "tia" (sample point) được coi là:
              + VISIBLE nếu ở ÍT NHẤT 1 camera, nó match màu object
              + OCCLUDED nếu ở tất cả camera, pixel đó không trùng màu object
        - Tính:
              visible_ratio = num_visible_samples / total_samples
          Nếu visible_ratio >= visibility_thresh (default 0.33) → object visible (1)
          Ngược lại → object occluded (2)

Lưu ý:
    - Do dataset là semantic seg (không phải instance seg),
      ta giả định mỗi object BEV thuộc 1 class màu (ground, non-driveable, EV, bus, car).
      Occlusion được xấp xỉ bằng việc: "pixel camera có màu khác class của object"
      → coi như bị vật khác / nền che.

Chạy:
    python src/validation/demo_visibility_mask.py \
        --dataset-root /path/to/FB-SSEM/images0/train \
        --intrinsics    /path/to/CameraCalibrationParameters/camera_intrinsics.yml \
        --id 164 \
        --output-bev visibility_164.png
"""

import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path

# ------------------------------------------------------------
# Thêm src/ vào sys.path (giống project_mei, make_video)
# ------------------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Local utils
from utils.camera import load_intrinsics, load_extrinsics, load_camera_bev_height
from utils.projects import cam2image
from utils.pipeline import (
    load_depth,
    load_seg,
    compute_height_map,
    segment_objects,
    get_2d_bounding_boxes,
    get_3d_bounding_boxes,
)


# ============================================================
# SAMPLING POINTS ON CUBOID SURFACES
# ============================================================
def sample_cuboid_points(corners: np.ndarray, grid=(6, 6)) -> np.ndarray:
    """
    Sample đều các điểm trên 6 mặt của cuboid.

    corners: (8,3), thứ tự giống bbox3d.cuboid_corners:
        0..3 = bottom, 4..7 = top
    grid: (gu, gv) số samples theo u,v

    Return: (N,3) world coords
    """
    faces = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]

    gu, gv = grid
    us = np.linspace(0.05, 0.95, gu)
    vs = np.linspace(0.05, 0.95, gv)

    pts = []
    for f in faces:
        p00, p10, p11, p01 = corners[f]  # 4 x 3
        for u in us:
            for v in vs:
                # bilinear trên quad
                p = (
                    (1 - u) * (1 - v) * p00
                    + u * (1 - v) * p10
                    + u * v * p11
                    + (1 - u) * v * p01
                )
                pts.append(p)

    return np.asarray(pts, dtype=np.float32)


# ============================================================
# VISIBILITY / OCCLUSION TEST
# ============================================================
def compute_cuboid_visibility(
    corners: np.ndarray,
    obj_color: np.ndarray,
    cam_segs: dict,
    extrinsics: dict,
    K: np.ndarray,
    D: np.ndarray,
    xi: float,
    cam_name_map: dict,
    visibility_thresh: float = 0.33,
    color_tol: float = 25.0,
) -> int:
    """
    Tính visibility cho 1 cuboid:

        1 -> visible
        2 -> occluded

    obj_color: RGB in [0,1] (từ BEV object)
    cam_segs: { "front": seg_rgb, ... } (RGB uint8)
    color_tol: ngưỡng chênh lệch màu cho phép (0–255), per-channel max diff.
    """
    pts_world = sample_cuboid_points(corners, grid=(6, 6))  # (N,3)

    total_samples = 0
    visible_samples = 0

    # scale obj_color lên [0,255] để so với camera seg
    obj_rgb_255 = (np.clip(obj_color, 0.0, 1.0) * 255.0).astype(np.float32)  # (3,)

    for view, seg_img in cam_segs.items():
        if seg_img is None:
            continue

        H, W = seg_img.shape[:2]
        ext_key = cam_name_map.get(view, None)
        if ext_key is None or ext_key not in extrinsics:
            continue

        Extrinsic = extrinsics[ext_key]

        # Project world -> cam image (fisheye MEI)
        uv, mask = cam2image(pts_world, Extrinsic, K, D, xi)
        uv = uv[mask]
        if uv.shape[0] == 0:
            continue

        # Integer pixel coords
        u = uv[:, 0].astype(int)
        v = uv[:, 1].astype(int)

        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(in_bounds):
            continue

        u = u[in_bounds]
        v = v[in_bounds]

        seg_vals = seg_img[v, u].astype(np.float32)  # (M,3) in [0,255]
        if seg_vals.size == 0:
            continue

        # Mỗi sample point = 1 "ray" –> check màu trên camera seg
        # Ray được coi là "hit đúng object" nếu màu đủ gần màu object trên BEV
        diff = np.abs(seg_vals - obj_rgb_255[None, :])  # (M,3)
        max_chan_diff = np.max(diff, axis=1)            # (M,)

        # visible nếu tất cả channel lệch < color_tol
        hits = max_chan_diff < color_tol

        visible_samples += int(np.sum(hits))
        total_samples += seg_vals.shape[0]

    if total_samples == 0:
        # Tất cả samples out-of-FOV / không chiếu vào ảnh:
        # coi như object occluded / không nhìn thấy.
        return 2

    visible_ratio = visible_samples / float(total_samples)

    # DEBUG (nếu cần)
    print(
        f"[DEBUG] visibility: visible_samples={visible_samples}, "
        f"total_samples={total_samples}, ratio={visible_ratio:.3f}"
    )

    return 1 if visible_ratio >= visibility_thresh else 2


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--resolution", type=float, default=100 / (6 * 400))
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--offset", type=float, default=33.0)
    ap.add_argument("--yshift", type=float, default=-0.3)
    ap.add_argument("--visibility-thresh", type=float, default=0.33)
    ap.add_argument("--color-tol", type=float, default=25.0)
    ap.add_argument("--output-bev", default="visibility_bev.png")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    sample_id = args.id

    # ---------------------------
    # 1. Load BEV segmentation (RGB)
    # ---------------------------
    bev_seg_path = root / "seg" / "bev" / f"{sample_id}.png"
    if not bev_seg_path.exists():
        print(f"[ERROR] Missing BEV seg: {bev_seg_path}")
        return

    bev_seg = load_seg(str(bev_seg_path))  # RGB uint8
    H, W = bev_seg.shape[:2]

    # Tách object theo màu (instance segmentation từ semantic seg)
    obj_masks = segment_objects(bev_seg, min_area=args.min_area)
    if not obj_masks:
        print("[INFO] No objects found in BEV seg.")
        return

    print(f"[DEBUG] Found {len(obj_masks)} BEV objects")

    # ---------------------------
    # 2. Load depth + height map
    # ---------------------------
    depth_path = root / "depth" / f"{sample_id}.png"
    if not depth_path.exists():
        print(f"[ERROR] Missing depth: {depth_path}")
        return

    depth_norm = load_depth(str(depth_path))

    cfg_path = root / "cameraconfig" / f"{sample_id}.txt"
    bev_cam_height = None
    if cfg_path.exists():
        try:
            bev_cam_height = load_camera_bev_height(str(cfg_path))
            print(f"[DEBUG] BEV cam height from config: {bev_cam_height}")
        except Exception as e:
            print(f"[WARN] Failed to read BEV camera height, fallback to 10.0: {e}")
            bev_cam_height = 10.0
    else:
        print("[WARN] cameraconfig not found, fallback bev_cam_height = 10.0")
        bev_cam_height = 10.0

    height_map = compute_height_map(depth_norm, bev_cam_height)

    # ---------------------------
    # 3. 2D + 3D bounding boxes
    # ---------------------------
    boxes_2d = get_2d_bounding_boxes(obj_masks)
    cuboids = get_3d_bounding_boxes(
        boxes_2d,
        height_map,
        args.resolution,
        offset=args.offset,
        yshift=args.yshift,
    )

    if not cuboids:
        print("[INFO] No cuboids computed.")
        return

    print(f"[DEBUG] Computed {len(cuboids)} cuboids")

    # ---------------------------
    # 4. Load intrinsics + extrinsics
    # ---------------------------
    K, D, xi = load_intrinsics(Path(args.intrinsics))

    extr_path = root.parent.parent / "CameraCalibrationParameters" / "camera_positions_for_extrinsics.txt"
    extrinsics = load_extrinsics(extr_path)

    cam_name_map = {
        "front": "Main Camera-front",
        "left": "Main Camera-left",
        "right": "Main Camera-right",
        "rear": "Main Camera-rear",
    }

    # ---------------------------
    # 5. Load camera segmentation maps (RGB)
    # ---------------------------
    cam_segs = {}
    for view in cam_name_map.keys():
        seg_path = root / "seg" / view / f"{sample_id}.png"
        if seg_path.exists():
            cam_segs[view] = load_seg(str(seg_path))
            print(f"[DEBUG] Loaded camera seg: {view} -> {seg_path}")
        else:
            cam_segs[view] = None
            print(f"[WARN] Missing camera seg for view={view}: {seg_path}")

    # ---------------------------
    # 6. Compute BEV visibility mask
    # ---------------------------
    bev_visibility = np.zeros((H, W), dtype=np.uint8)  # 0 = background

    # Zipping: giả định thứ tự cuboids tương ứng thứ tự boxes_2d/obj_masks
    for idx, (obj, cub) in enumerate(zip(obj_masks, cuboids)):
        obj_color = np.asarray(obj.get("color", [1.0, 1.0, 1.0]), dtype=np.float32)

        print(f"[DEBUG] Object {idx}: color={obj_color}, label={obj.get('label', -1)}")

        label = compute_cuboid_visibility(
            corners=cub["corners"],
            obj_color=obj_color,
            cam_segs=cam_segs,
            extrinsics=extrinsics,
            K=K,
            D=D,
            xi=xi,
            cam_name_map=cam_name_map,
            visibility_thresh=args.visibility_thresh,
            color_tol=args.color_tol,
        )

        print(f"[DEBUG] Object {idx}: visibility_label={label}")
        bev_visibility[obj["mask"] > 0] = label

    # ---------------------------
    # 7. Save result (0/1/2 and visualization)
    # ---------------------------
    # Raw label map: 0/1/2 dùng để train
    raw_out_path = Path(args.output_bev).with_suffix(".raw.png")
    cv2.imwrite(str(raw_out_path), bev_visibility)
    print(f"[OK] Saved raw visibility labels → {raw_out_path}")

    # Visualization: 0=0, 1=180, 2=255
    vis = np.zeros_like(bev_visibility, dtype=np.uint8)
    vis[bev_visibility == 1] = 180
    vis[bev_visibility == 2] = 255

    vis_out_path = Path(args.output_bev)
    cv2.imwrite(str(vis_out_path), vis)
    print(f"[OK] Saved visibility visualization → {vis_out_path}")


if __name__ == "__main__":
    main()
