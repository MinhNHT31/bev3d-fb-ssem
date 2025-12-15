#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional
import numpy as np
import os

from utils.pipeline import (
    load_depth,
    compute_height_map,
    segment_objects,
    get_2d_bounding_boxes,
    get_3d_bounding_boxes,
)
from utils.camera import load_camera_bev_height

DEFAULT_CAMERA_HEIGHT = 40.0

# CLI helper to turn BEV segmentation + depth into 3D bbox JSON annotations.

def export_bboxes_json(cuboids, path: Path):
    """Export 3D bbox annotations to JSON."""
    ann = []
    for c in cuboids:
        obb = c["obb"]
        entry = {
            "center": obb["center"],
            "size": obb["size"],
            "angle_deg": obb["angle"],
            "height": c["height"],
            "corners": c["corners"].tolist(),
        }
        ann.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(ann, f, indent=2)


def process_sample(
    bev_path: Path,
    depth_path: Path,
    cam_height: float,
    resolution: float,
    min_area: int,
    ann_dir: Path,
    json_name: Optional[str] = None,
):
    # —— Load BEV mask ——
    bev_mask = load_depth(str(bev_path))
    bev_mask = (bev_mask > 0).astype("uint8") * 255
    obj_masks = segment_objects(bev_mask, min_area=min_area)

    # —— Load Depth → Height ——
    depth_norm = load_depth(str(depth_path))
    height_map = compute_height_map(depth_norm, cam_height)

    # —— 2D → 3D ——
    boxes_2d = get_2d_bounding_boxes(obj_masks)
    
    cuboids  = get_3d_bounding_boxes(boxes_2d, height_map, resolution)

    # —— Save JSON ——
    json_path = ann_dir / (json_name or f"{bev_path.stem}.json")
    export_bboxes_json(cuboids, json_path)

    return json_path


def main():
    ap = argparse.ArgumentParser(description="Export ONLY 3D bbox annotations (no point cloud)", allow_abbrev=False)
    ap.add_argument("--bev", help="Path to BEV segmentation mask (single mode)")
    ap.add_argument("--depth", help="Path to depth map (single mode)")
    ap.add_argument("--dataset-root", help="Dataset root containing seg/bev and depth folders")
    ap.add_argument("--resolution", type=float, default=100/(6*400))
    ap.add_argument("--camera-height", type=float, default=None)
    ap.add_argument("--camera-config-dir", type=str, default=None)
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--out-dir", type=str, default="outputs_bboxes")
    ap.add_argument("--default-camera-height", type=float, default=DEFAULT_CAMERA_HEIGHT)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir = os.path.join(out_dir, "export_annotations")
    out_dir = Path(out_dir)

    # ============================================================
    #   BATCH MODE — dataset-root
    # ============================================================
    if args.dataset_root:
        root = Path(args.dataset_root)
        seg_dir   = root / "seg" / "bev"
        depth_dir = root / "depth"
        cfg_dir   = Path(args.camera_config_dir) if args.camera_config_dir else (root / "cameraconfig")

        if not seg_dir.exists() or not depth_dir.exists():
            raise FileNotFoundError(f"Expected seg/bev and depth folders under: {root}")

        # IDs
        ids = [p.stem for p in sorted(seg_dir.glob("*.png"))]

        # Output folders
        run_dir = out_dir / f"{root.name}_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ann_dir = run_dir / "ann"
        ann_dir.mkdir(parents=True, exist_ok=True)

        # Process each sample
        for sample_id in ids:
            bev_path   = seg_dir / f"{sample_id}.png"
            depth_path = depth_dir / f"{sample_id}.png"

            if not bev_path.exists() or not depth_path.exists():
                print(f"[warn] Skipping {sample_id}: missing BEV or depth.")
                continue

            # Load camera height
            cam_h = args.camera_height
            cfg_path = cfg_dir / f"{sample_id}.txt"

            if cam_h is None and cfg_path.exists():
                try:
                    cam_h = load_camera_bev_height(str(cfg_path))
                    print(f"[info] {sample_id}: loaded camera height {cam_h}")
                except Exception as exc:
                    print(f"[warn] {sample_id}: failed to load camera height: {exc}")

            if cam_h is None:
                cam_h = args.default_camera_height

            json_out = process_sample(
                bev_path, depth_path, cam_h,
                args.resolution, args.min_area,
                ann_dir,
                json_name=f"{sample_id}.json"
            )

            print(f"[info] Saved bbox JSON → {json_out}")

    # ============================================================
    #   SINGLE MODE — direct file input
    # ============================================================
    else:
        if not args.bev or not args.depth:
            ap.error("Provide --bev and --depth OR --dataset-root.")

        cam_h = args.camera_height or args.default_camera_height

        run_dir = out_dir / f"single_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ann_dir = run_dir / "ann"
        ann_dir.mkdir(parents=True, exist_ok=True)

        json_path = process_sample(
            Path(args.bev),
            Path(args.depth),
            cam_h,
            args.resolution,
            args.min_area,
            ann_dir,
            json_name=args.json_name,
        )

        print(f"[info] Saved bbox JSON → {json_path}")


if __name__ == "__main__":
    main()
