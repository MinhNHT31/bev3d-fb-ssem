import numpy as np
from typing import Tuple

def cam2image(
    points: np.ndarray, 
    Extrinsic: np.ndarray, 
    K: np.ndarray, 
    D: np.ndarray, 
    xi: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects 3D world points onto a 2D image plane using the MEI (Unified Camera) model.
    This model is specifically designed for Fisheye and Omnidirectional cameras.

    Args:
        points (np.ndarray): Nx3 array of 3D points in World or Local coordinates.
        Extrinsic (np.ndarray): 4x4 Matrix transforming World/Local -> Camera Coordinate System.
        K (np.ndarray): 3x3 Intrinsic Matrix (contains focal lengths fx, fy and centers cx, cy).
        D (np.ndarray): Distortion coefficients array (k1, k2, ...).
        xi (float): The mirror parameter specific to the Mei model. Controls the sphere projection.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - uv (Nx2): Projected 2D pixel coordinates [u, v].
            - mask (N,): Boolean mask indicating valid points (True = visible/in front of camera).
    """
    N = points.shape[0]
    
    # 0. Handle empty input to prevent errors
    if N == 0: 
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)
    
    # ---------------------------------------------------------
    # 1. Coordinate Transformation: World -> Camera Frame
    # ---------------------------------------------------------
    # Convert 3D points to Homogeneous coordinates: [x, y, z] -> [x, y, z, 1]
    pts_homo = np.hstack([points, np.ones((N, 1))])
    
    # Apply Extrinsic Matrix: P_cam = Extrinsic * P_world
    # The result is Nx4, we transpose twice to handle matrix multiplication shapes correctly.
    pts_cam = (Extrinsic @ pts_homo.T).T 
    
    # Extract X, Y, Z in Camera Frame
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    
    # ---------------------------------------------------------
    # 2. Filtering: Safety Clipping
    # ---------------------------------------------------------
    # Only keep points physically in front of the camera (Z > epsilon).
    # Points with Z <= 0 are behind the lens and cannot be projected correctly.
    mask = z > 0.01 
    
    # If no points are valid, return empty result immediately
    if not np.any(mask): 
        return np.zeros((N, 2)), mask

    # Filter arrays to process only valid points (Optimization)
    x_v, y_v, z_v = x[mask], y[mask], z[mask]

    # ---------------------------------------------------------
    # 3. Mei Projection (Spherical Projection)
    # ---------------------------------------------------------
    # Calculate Euclidean distance from the camera center (rho)
    rho = np.sqrt(x_v**2 + y_v**2 + z_v**2)
    
    # Calculate the projection denominator according to Mei model formula
    # Formula: x_norm = X / (Z + xi * rho)
    denom = z_v + (xi * rho)
    
    # Prevent division by zero
    denom[np.abs(denom) < 1e-6] = 1e-6
    
    # Calculate normalized coordinates on the unit sphere/plane
    xu, yu = x_v / denom, y_v / denom
    
    # ---------------------------------------------------------
    # 4. Apply Distortion (Radial Only)
    # ---------------------------------------------------------
    # Calculate squared radius from the image center
    r2 = xu**2 + yu**2
    r4 = r2**2
    
    # Parse distortion coefficients (k1, k2) from input array D
    k1 = D[0] if len(D) > 0 else 0.0
    k2 = D[1] if len(D) > 1 else 0.0
    
    # Calculate radial scaling factor
    # Polynomial model: 1 + k1*r^2 + k2*r^4 + ...
    radial = 1.0 + k1 * r2 + k2 * r4
    
    # Apply distortion to the normalized coordinates
    xd = xu * radial
    yd = yu * radial
    
    # ---------------------------------------------------------
    # 5. Apply Intrinsics: Metric -> Pixel
    # ---------------------------------------------------------
    # Map normalized coordinates to pixel grid using Focal Length (fx, fy) 
    # and Principal Point (cx, cy) form Matrix K.
    # u = fx * x_distorted + cx
    # v = fy * y_distorted + cy
    u = K[0, 0] * xd + K[0, 2]
    v = K[1, 1] * yd + K[1, 2]
    
    # ---------------------------------------------------------
    # 6. Final Output Assembly
    # ---------------------------------------------------------
    # Create an output array filled with zeros (matching original N size)
    uv = np.zeros((N, 2))
    
    # Place valid projected points back into their original indices using the mask
    uv[mask] = np.stack([u, v], axis=1)
    
    return uv, mask
