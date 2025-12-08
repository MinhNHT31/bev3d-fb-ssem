from pathlib import Path

img_path = Path("/media/minhnht/4d8c66d1-9c3e-4d4e-bc7a-e6010e3c0823/MNHT/FB-SSEM/images0/train/rgb/front/100.png")
cfg_extr = Path("/media/minhnht/4d8c66d1-9c3e-4d4e-bc7a-e6010e3c0823/MNHT/FB-SSEM/CameraCalibrationParameters/camera_positions_for_extrinsics.txt")
mei_yaml = Path("/media/minhnht/4d8c66d1-9c3e-4d4e-bc7a-e6010e3c0823/MNHT/FB-SSEM/CameraCalibrationParameters/camera_intrinsics.yml")

import numpy as np
import cv2
from utils.camera import load_intrinsics, load_extrinsics
from utils.projects import cam2image

# ========== CONFIG ==========
intrinsic_path = mei_yaml
extrinsic_path = cfg_extr
cam_name = "Main Camera-front"
image_path = img_path 
# ============================

# Load camera params
K, D, xi = load_intrinsics(intrinsic_path)
ext_dict = load_extrinsics(extrinsic_path)
Extrinsic = ext_dict[cam_name]

# Load test image
img = cv2.imread(image_path)
img = cv2.flip(img, 0)
H, W = img.shape[:2]

# ------ CREATE 3D GRID IN WORLD ------

# World grid on ground plane (Y=0)
xs = np.linspace(-20, 20, 41)
zs = np.linspace(0, 40, 41)
X, Z = np.meshgrid(xs, zs)

# Build 3D points (flatten)
points = np.vstack([X.ravel(), np.zeros_like(X).ravel(), Z.ravel()]).T

# ------ PROJECT GRID TO IMAGE ------
uv, mask = cam2image(points, Extrinsic, K, D, xi)

# Draw projected points
for (u, v) in uv[mask].astype(int):
    if 0 <= u < W and 0 <= v < H:
        img = cv2.circle(img, (u, v), 2, (0, 0, 255), -1)
img = cv2.flip(img, 0)
cv2.imshow("Grid Projection Debug", img)
cv2.waitKey(0)
