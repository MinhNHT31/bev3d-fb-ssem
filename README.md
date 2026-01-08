# FB-SSEM BEV Visibility & Occlusion Toolkit

Utilities to lift FB-SSEM BEV semantic segmentations and depth maps into 3D cuboids, project them into camera views, and generate occlusion-aware BEV masks. The main pipeline uses camera semantic segmentation as an implicit Z-buffer: rays that match the BEV object color are counted as visible; mismatches are treated as occluded.

## Dataset Layout
Point commands to the dataset root that holds calibration and splits:
```
FB-SSEM/
  CameraCalibrationParameters/
    camera_intrinsics.yml
    camera_positions_for_extrinsics.txt
  images0/, images1/, ...
    train|val|test/
      depth/<id>.png
      seg/
        bev/<id>.png
        front/<id>.png
        left/<id>.png
        right/<id>.png
        rear/<id>.png
      cameraconfig/<id>.txt   # optional per-frame BEV camera height (posi_y)
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install matplotlib tqdm   # viewers and video helpers
```
Headless note: if Open3D or GUI backends fail, set `PYOPENGL_PLATFORM=osmesa` or use Xvfb.

## Run the Full BEV Visibility Pipeline (main.py)
Process every BEV file across all splits:
```bash
python src/main.py \
  --dataset-root /path/to/FB-SSEM \
  --keep-occlusion \        # keep label 2; drop to 0 if omitted
  --num-workers 8 \
  --visibility-thresh 0.33 \
  --color-tol 25 \
  --resolution 0.0416 \
  --min-area 50 \
  --offset 33 \
  --yshift -0.3
```
Outputs per BEV live beside inputs:
- `seg/bev_raw/<id>.png` — labels 0/1/2 (or 0/1 if `--keep-occlusion` is off).
- `seg/bev_occ/<id>.png` — visualization 0/180/255.
Intrinsics/extrinsics are loaded once from `CameraCalibrationParameters`.

### Important flags
- `--keep-occlusion` keep occluded pixels as label 2; otherwise set to 0.
- `--visibility-thresh` minimum visible ray ratio to call an object visible.
- `--color-tol` per-channel tolerance for matching segmentation colors.
- `--resolution`, `--min-area`, `--offset`, `--yshift` shape how BEV masks lift into cuboids.

## Validation & Visualization
- **Single-frame visibility mask**  
  ```bash
  python src/validation/demo_visibility_mask.py \
    --dataset-root FB-SSEM/images0/train \
    --intrinsics FB-SSEM/CameraCalibrationParameters/camera_intrinsics.yml \
    --id 164 --offset 33 --yshift -0.3 \
    --visibility-thresh 0.33 --color-tol 25 --keep-occlusion
  ```
  Saves `visibility_<id>.raw.png` (0/1/2) and `visibility_<id>.png` (0/180/255).

- **Single-frame projection viewer (matplotlib 2x3 grid)**  
  ```bash
  python src/validation/project_mei.py \
    --dataset-root FB-SSEM/images0/train \
    --intrinsics FB-SSEM/CameraCalibrationParameters/camera_intrinsics.yml \
    --id 164 --offset 33 --yshift -0.3
  ```
  Shows Left/Front/Right (top) and Depth/Rear/BEV (bottom) with projected cuboids.

- **Composite video over a split**  
  ```bash
  python src/validation/make_video.py \
    --dataset-root FB-SSEM/images0/train \
    --intrinsics FB-SSEM/CameraCalibrationParameters/camera_intrinsics.yml \
    --output seg_video.mp4 --fps 10 --offset 33 --yshift -0.3
  ```
  Writes `videos/seg_video.mp4` with a 2x3 grid per frame.

## Occlusion Logic (core idea)
1) Load BEV semantic segmentation and split into instances (`segment_objects`).  
2) Convert depth + BEV camera height into a per-pixel height map (`compute_height_map`).  
3) Fit 2D oriented boxes and lift to 3D cuboids (`get_2d_bounding_boxes`, `get_3d_bounding_boxes`).  
4) Sample cuboid faces, project to each camera (`utils.projects.cam2image` + intrinsics/extrinsics).  
5) Count rays whose pixel color matches the BEV object within `--color-tol`.  
6) If `visible_ratio >= --visibility-thresh`, label pixels as visible (1); otherwise occluded (2 or 0).  
7) Write raw mask and a visualization map next to the source BEV.

## Important functions used by main.py
- `utils.pipeline.load_seg` / `load_depth` — read segmentation (RGB) and normalized depth.
- `utils.pipeline.segment_objects` — connected components per color with mask/color metadata.
- `utils.pipeline.compute_height_map` — depth + BEV camera height → per-pixel height.
- `utils.pipeline.get_2d_bounding_boxes` — oriented boxes per instance.
- `utils.pipeline.get_3d_bounding_boxes` — lift boxes into cuboids with heuristics.
- `utils.visibility.compute_cuboid_visibility` — ray-level visibility from camera segmentations.
- `utils.projects.cam2image` — MEI projection helper for all camera views.

## Results
- Videos from `make_video.py` are stored under `videos/`.
- Replace this placeholder with your YouTube demo link: **[https://youtu.be/mtobtOk-yd0]**
