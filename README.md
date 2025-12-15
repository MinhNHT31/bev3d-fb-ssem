# BEV 3D Cuboids on FB-SSEM

Pipeline and utilities for turning FB-SSEM BEV segmentation + depth into 3D cuboids, projecting them onto camera views, and exporting annotations/visualizations.

## Repository Layout
- `data/fb-ssem.sh` – bulk downloader for the FB-SSEM dataset (aria2c + unzip required).
- `requirements.txt` – Python dependencies (OpenCV, NumPy, Open3D, etc.). Visualization scripts also use `matplotlib` and `tqdm`.
- `src/utils/` – core geometry pipeline: mask loading, object segmentation, 2D OBBs, 3D cuboids, camera intrinsics/extrinsics helpers, MEI projection.
- `src/annotation/export_annotations.py` – batch/single export of 3D bbox JSON from BEV + depth.
- `src/validation/` – visual tools:
  - `project_mei.py` project cuboids onto camera images for a single frame (matplotlib viewer).
  - `make_video.py` compose a 2x3 grid video over a dataset split.
  - `demo_visibility_mask.py` build occlusion-aware BEV visibility masks (0 bg, 1 visible, 2 occluded).
- `src/test/test_cam2image.py` – quick grid projection sanity check for intrinsics/extrinsics.

## Setup
1. Python 3.9+ recommended. Install deps:
   ```bash
   pip install -r requirements.txt
   pip install matplotlib tqdm  # needed for viewers/video
   ```
2. Open3D may need system GL/GUI packages. On headless servers set `export PYOPENGL_PLATFORM=osmesa` or use virtual framebuffer for viewers.
3. Dataset download (optional helper):
   ```bash
   bash data/fb-ssem.sh
   ```
   This creates `FB-SSEM/` with calibration files and image splits.

## Expected Dataset Layout
Point scripts at a split root such as `FB-SSEM/images0/train` containing:
```
<split_root>/
  depth/<id>.png
  seg/
    bev/<id>.png
    front/<id>.png
    left/<id>.png
    right/<id>.png
    rear/<id>.png
  cameraconfig/<id>.txt           # per-frame BEV camera height (posi_y)
CameraCalibrationParameters/
  camera_intrinsics.yml           # MEI intrinsics (K, D, xi)
  camera_positions_for_extrinsics.txt  # Unity-style extrinsics per camera
```
For extrinsics, scripts look two levels up from `--dataset-root` (e.g., `split_root/../../CameraCalibrationParameters/...`).

## Core Pipeline (utils/pipeline.py)
1) Load BEV mask and segmentation colors (`load_seg` / `segment_objects`).
2) Load normalized depth and convert to a height map using BEV camera height (`compute_height_map`).
3) Extract 2D oriented boxes (`get_2d_bounding_boxes`).
4) Lift to 3D cuboids with heuristic heights (`get_3d_bounding_boxes`), optionally visualized via Open3D cameras (`create_camera_visuals`, `draw_3d_scene`).

## Usage
### Export 3D bbox annotations
Batch over a split:
```bash
python src/annotation/export_annotations.py \
  --dataset-root FB-SSEM/images0/train \
  --camera-config-dir FB-SSEM/images0/train/cameraconfig \
  --out-dir outputs_bboxes \
  --resolution 0.0416  --min-area 50
```
- Saves JSON under `outputs_bboxes/export_annotations/<run>/ann/ID.json` with `center/size/angle_deg/height/corners`.
- If no per-frame camera height is found, falls back to `--default-camera-height` (40.0 by default).

Single pair of files:
```bash
python src/annotation/export_annotations.py \
  --bev path/to/seg/bev/164.png --depth path/to/depth/164.png \
  --camera-height 40 --out-dir outputs_bboxes_single
```

### Project cuboids onto camera images (single frame viewer)
```bash
python src/validation/project_mei.py \
  --dataset-root FB-SSEM/images0/train \
  --intrinsics FB-SSEM/CameraCalibrationParameters/camera_intrinsics.yml \
  --id 164 --offset 33 --yshift -0.3 --min-area 50
```
- Opens a 2x3 matplotlib figure (Left/Front/Right on top, Depth/Rear/BEV below). Uses extrinsics from `CameraCalibrationParameters`.

### Visibility / occlusion-aware BEV mask
```bash
python src/validation/demo_visibility_mask.py \
  --dataset-root FB-SSEM/images0/train \
  --intrinsics FB-SSEM/CameraCalibrationParameters/camera_intrinsics.yml \
  --id 164 --output-bev visibility_164.png \
  --visibility-thresh 0.33 --color-tol 25 --offset 33 --yshift -0.3
```
- Produces `visibility_164.raw.png` (labels 0/1/2) and `visibility_164.png` visualization (0/180/255).

### Composite video over a split
```bash
python src/validation/make_video.py \
  --dataset-root FB-SSEM/images0/train \
  --intrinsics FB-SSEM/CameraCalibrationParameters/camera_intrinsics.yml \
  --output seg_video.mp4 --fps 10 --offset 33 --yshift -0.3
```
- Writes to `videos/seg_video.mp4`. Each frame is a 2x3 grid (Left/Front/Right, Depth/Rear/BEV) with projected cuboids and ground grid on the front view.

### Extrinsics/intrinsics sanity check
Edit paths in `src/test/test_cam2image.py` and run it to project a ground grid onto a sample image for quick calibration validation.

## Tunable Parameters
- `--resolution`  meters per pixel for BEV scaling (`100/(6*400)` default ≈ 0.0416).
- `--min-area`  minimum BEV mask area to accept an object.
- `--offset`, `--yshift`  adjust cuboid placement vs. BEV origin.
- `--visibility-thresh`, `--color-tol`  control occlusion decision in visibility masks.

## Troubleshooting
- If Open3D/GUI windows fail on servers, disable `--vis` flags or use offscreen setups (Xvfb, OSMesa).
- Ensure intrinsics/extrinsics files match the dataset split; wrong calibration will misplace cuboids.
- Missing camera view images default to black placeholders in video/visibility scripts.

