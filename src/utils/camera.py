import numpy as np
from pathlib import Path
import cv2
import csv
from typing import Dict

# Camera helpers convert Unity-style calibration files into OpenCV-friendly
# intrinsics/extrinsics for the projection utilities.


# ============================================================
# Utility: Build Extrinsic Matrix (World → Camera)
# ============================================================
def get_extrinsics(config_dict):
    """
    Computes the camera extrinsic matrix (World -> Camera) from a Unity-style 
    camera configuration.

    config_dict contains:
        pos : (x, y, z) camera position in world coordinates
        rot : (rot_x, rot_y, rot_z) Euler rotations in degrees (Unity order)

    Rotation order used here follows Unity convention:
        R_cam = Ry * Rx * Rz   (Yaw → Pitch → Roll)

    The final extrinsic matrix represents the transformation:
        P_cam = Extrinsic * P_world
    """

    # --- Camera position in world coordinate ---
    pos = np.array(config_dict["pos"])
    
    # --- Euler angles (degrees → radians) ---
    rot = np.array(config_dict["rot"])
    rx, ry, rz = np.deg2rad([rot[0], rot[1], rot[2]])

    # --- Rotation matrices for each axis ---
    # Pitch (X-axis)
    Rx_m = np.array([
        [1,        0,         0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)]
    ])

    # Yaw (Y-axis)
    Ry_m = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [ 0,          1,         0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    # Roll (Z-axis)
    Rz_m = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0,           0,          1]
    ])

    # ================================================================
    # Unity camera rotation: Apply Yaw → Pitch → Roll (Ry * Rx * Rz)
    # ================================================================
    R_cam = Ry_m @ Rx_m @ Rz_m

    # Convert Unity world ↔ camera convention to a View Matrix form.
    # View rotation is the transpose (inverse) of world rotation.
    R_view = R_cam.T

    # Translation in camera coordinates:
    t_view = -R_view @ pos.reshape(3, 1)

    # # 4×4 extrinsic matrix
    # View_Matrix = np.eye(4)
    # View_Matrix[:3, :3] = R_view
    # View_Matrix[:3, 3] = t_view.flatten()


    # Extrinsic = View_Matrix
    Extrinsic = R_view, t_view
    return Extrinsic

# def unity2opencv(Extrinsic_unity):
#     T = np.diag([1, -1, 1, 1])
#     Extrinsic_cv = T @ Extrinsic_unity
#     return Extrinsic_cv

def unity2opencv(Extrinsic_unity):
    """
    Convert Unity-style extrinsic (R, t) to OpenCV convention.
    """

    R_u, t_u = Extrinsic_unity  # R: 3x3, t: 3x1

    # Axis conversion matrix (Unity → OpenCV)
    # C1 = np.diag([-1, 1, -1])
    C2 = np.diag([1, -1, 1])

    # Convert rotation and translation
    R_cv = C2 @ R_u
    t_cv = C2 @ t_u

    # Build 4x4 extrinsic
    Extrinsic_cv = np.eye(4)
    Extrinsic_cv[:3, :3] = R_cv
    Extrinsic_cv[:3, 3] = t_cv.flatten()

    return Extrinsic_cv


# ============================================================
# Load Extrinsics for All Cameras from CSV File
# ============================================================
def load_extrinsics(path: str) -> Dict[str, np.ndarray]:
    """
    Loads camera extrinsic parameters from a camera configuration .txt/.csv file.

    Expected columns:
        cam_name, posi_x, posi_y, posi_z, rot_x, rot_y, rot_z
    Some datasets may use rota_x, rota_y, rota_z instead.

    Returns:
        dict: { camera_name : 4x4 Extrinsic matrix (World → Camera) }
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Camera config not found: {path}")

    extrinsics = {}

    # Read CSV rows; skipinitialspace cleans stray spaces after commas
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, skipinitialspace=True)

        for row_idx, row in enumerate(reader):
            if not row:
                continue

            cam_name = (row.get("cam_name") or "").strip()
            if not cam_name or cam_name.startswith("#"):
                continue

            try:
                # Helper to parse and validate numerical fields
                def get_float(key):
                    val = (row.get(key) or "").strip()
                    if not val:
                        raise ValueError(f"Empty value for {key}")
                    return float(val)

                # Position
                posi_x = get_float("posi_x")
                posi_y = get_float("posi_y")
                posi_z = get_float("posi_z")

                # Rotation (supports 'rot_x' or 'rota_x' fallback)
                rot_x = get_float("rot_x") if "rot_x" in row else get_float("rota_x")
                rot_y = get_float("rot_y") if "rot_y" in row else get_float("rota_y")
                rot_z = get_float("rot_z") if "rot_z" in row else get_float("rota_z")

                config = {
                    "pos": [posi_x, posi_y, posi_z],
                    "rot": [rot_x, rot_y, rot_z],
                }

                extrinsics[cam_name] = unity2opencv(get_extrinsics(config))

            except Exception as exc:
                print(f"[Warn] Skipping invalid row {row_idx} in {path}: {exc}")
                continue

    return extrinsics



# ============================================================
# Load MEI Intrinsics (K, Distortion, xi)
# ============================================================
def load_intrinsics(path: str) -> np.ndarray:
    """
    Loads MEI intrinsic parameters from a YAML file:
        K  : 3x3 intrinsic matrix
        D  : distortion coefficients
        xi : MEI mirror parameter
    """

    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)

    K = fs.getNode("K").mat()
    D = fs.getNode("D").mat().flatten()

    xi_node = fs.getNode("xi")
    xi = xi_node.real() if xi_node.isReal() else xi_node.mat()[0][0]

    return K, D, xi



# ============================================================
# Load BEV Camera Height Only
# ============================================================
def load_camera_bev_height(path: str) -> float:
    """
    Returns the BEV camera height (posi_y) from cameraconfig file.

    Only extracts the entry where `cam_name == "BEVCamera"`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Camera config not found: {path}")

    with path.open(newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)

        for row in reader:
            cam_name = (row.get("cam_name") or "").strip()
            if cam_name == "BEVCamera":
                try:
                    return float((row["posi_y"] or "").strip())
                except Exception as exc:
                    raise ValueError(f"Invalid posi_y value in {path}: {exc}")

    # If BEVCamera entry was not found
    raise ValueError(f"BEVCamera not found in {path}")
