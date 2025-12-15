import numpy as np
import logging
from utils.projects import cam2image

logger = logging.getLogger("visibility")
logger.setLevel(logging.INFO)

# Visibility heuristics to decide whether a 3D cuboid remains visible in camera
# segmentation maps based on sampled surface points and color agreement.

# ============================================================
# Sample points on cuboid surfaces
# ============================================================
def sample_cuboid_points(corners: np.ndarray, grid=(6, 6)) -> np.ndarray:
    """
    Uniformly sample points across all 6 faces of a cuboid.
    corners: (8,3) world coordinates of cuboid corners.
    Return: (N,3) sampled points.
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
        p00, p10, p11, p01 = corners[f]

        for u in us:
            for v in vs:
                pts.append(
                    (1-u)*(1-v)*p00 +
                    u*(1-v)*p10 +
                    u*v*p11 +
                    (1-u)*v*p01
                )

    return np.asarray(pts, np.float32)



# ============================================================
# Compute visibility (WITH final label inside)
# ============================================================
def compute_cuboid_visibility(
    corners,
    obj_color,
    cam_segs,
    extrinsics,
    K, D, xi,
    cam_name_map,
    visibility_thresh,
    color_tol,
    keep_occlusion,
):
    """
    Return final label:
        1 = visible
        2 = occluded (if keep_occlusion=True)
        0 = occluded (if keep_occlusion=False)

    Parameters:
        corners : (8,3) cuboid corners
        obj_color : [0–1] RGB from BEV
        cam_segs : dict(view → RGB segmentation)
        extrinsics : dict extrinsic matrices
        K,D,xi : camera intrinsics
        cam_name_map : mapping view → extrinsic key
        visibility_thresh : ratio threshold to count as visible
        color_tol : per-channel color difference tolerance
        keep_occlusion : whether to keep occluded label as 2 or drop to 0
    """

    pts_world = sample_cuboid_points(corners, grid=(6, 6))

    total_samples = 0
    visible_samples = 0

    # scale BEV color → 0..255
    obj_rgb_255 = (np.clip(obj_color, 0, 1) * 255).astype(np.float32)

    # ------------------------------------------------------------------
    # Iterate each camera and test visibility with its segmentation map
    # ------------------------------------------------------------------
    for view, seg_img in cam_segs.items():
        if seg_img is None:
            logger.debug(f"[{view}] No seg image → skip")
            continue

        H, W = seg_img.shape[:2]
        ext_key = cam_name_map.get(view)

        if ext_key not in extrinsics:
            logger.debug(f"[{view}] missing extrinsic → skip")
            continue

        Extr = extrinsics[ext_key]

        # Project cuboid → camera
        uv, mask = cam2image(pts_world, Extr, K, D, xi)
        uv = uv[mask]

        if uv.size == 0:
            logger.debug(f"[{view}] No projected points in FOV")
            continue

        u = uv[:, 0].astype(int)
        v = uv[:, 1].astype(int)

        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(in_bounds):
            logger.debug(f"[{view}] projected points all out of bounds")
            continue

        u = u[in_bounds]
        v = v[in_bounds]

        seg_vals = seg_img[v, u].astype(np.float32)

        # Compare camera pixel color vs BEV object color
        diff = np.abs(seg_vals - obj_rgb_255[None, :])
        hits = np.max(diff, axis=1) < color_tol   # True = visible ray

        visible_samples += int(np.sum(hits))
        total_samples += seg_vals.shape[0]

    # ------------------------------------------------------------------
    # If no samples fall inside any FOV, treat as occluded
    # ------------------------------------------------------------------
    if total_samples == 0:
        logger.debug("Total samples=0 → occluded")
        return 2 if keep_occlusion else 0

    visible_ratio = visible_samples / total_samples
    logger.debug(f"visible={visible_samples}, total={total_samples}, ratio={visible_ratio:.3f}")

    # ------------------------------------------------------------------
    # Determine final label
    # ------------------------------------------------------------------
    if visible_ratio >= visibility_thresh:
        return 1  # visible

    else:
        # object is occluded → optionally keep label 2 or drop to 0
        return 2 if keep_occlusion else 0
