import logging
import numpy as np
from typing import Tuple

# Cấu hình logging
logger = logging.getLogger("GeometryDebug")
logger.setLevel(logging.DEBUG)

# Tạo handler để in ra màn hình console
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def cam2image(
    points: np.ndarray,
    Extrinsic: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    xi: float,
    z_epsilon: float = 1e-1,
    image_size: Tuple[int, int] = (800, 600)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects 3D world points into image pixels using the Unified MEI Camera Model.
    This rewritten version assumes the **camera coordinate system is OpenCV-style**:
        +X → right
        +Y → down
        +Z → forward

    Args:
        points (Nx3 float): 3D points in WORLD coordinate system.
        Extrinsic (4x4 float): World → Camera matrix.
        K (3x3 float): Intrinsic matrix.
        D (float[]): Distortion coefficients (k1, k2, ...).
        xi (float): Mei mirror parameter.

    Returns:
        uv (Nx2 float): Pixel coordinates.
        mask (N bool): Visibility mask (point in front of camera & valid).
    """

    N = points.shape[0]
    if N == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)

    # ---------------------------------------------------------
    # 1. World → Camera
    # ---------------------------------------------------------
    pts_h = np.hstack([points, np.ones((N, 1))])        # Nx4
    pts_cam = (Extrinsic @ pts_h.T).T                  # Nx4

    X = pts_cam[:, 0]
    Y = pts_cam[:, 1]
    Z = pts_cam[:, 2]

    # Only points in front of camera
    mask = Z > z_epsilon
    if not np.any(mask):
        return np.zeros((N, 2)), mask

    Xv = X[mask]
    Yv = Y[mask]
    Zv = Z[mask]

    # ---------------------------------------------------------
    # 2. Unified MEI Projection
    # ---------------------------------------------------------
    rho = np.sqrt(Xv * Xv + Yv * Yv + Zv * Zv)
    denom = Zv + xi * rho
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)  # avoid divide-by-zero

    xu = Xv / denom
    yu = Yv / denom

    # ---------------------------------------------------------
    # 3. Radial Distortion (k1, k2) — Mei model uses OpenCV style
    # ---------------------------------------------------------
    r2 = xu * xu + yu * yu
    r4 = r2 * r2

    k1 = D[0] if len(D) > 0 else 0.0
    k2 = D[1] if len(D) > 1 else 0.0

    radial = 1.0 + k1 * r2 + k2 * r4

    xd = xu * radial
    yd = yu * radial

    # ---------------------------------------------------------
    # 4. Intrinsic scaling → Pixel coordinates
    # ---------------------------------------------------------
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = fx * xd + cx
    v = fy * yd + cy

    # ---------------------------------------------------------
    # 5. Output assembly
    # ---------------------------------------------------------
    uv = np.zeros((N, 2), dtype=float)
    uv[mask, 0] = u
    uv[mask, 1] = v

    return uv, mask
