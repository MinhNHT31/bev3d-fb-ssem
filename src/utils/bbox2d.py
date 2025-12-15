import cv2
import numpy as np
from typing import List, Dict, Tuple


# 2D bounding-box utilities for extracting oriented boxes from BEV masks.

def _normalize_rect(rect: Tuple[Tuple[float, float], Tuple[float, float], float]):
    """
    Normalize minAreaRect so width >= length and angle is consistent.
    Angle is wrapped to [-180, 180].
    """
    (cx, cy), (w, l), angle = rect
    if w < l:
        w, l = l, w
        angle += 90.0
    # Wrap angle to [-180, 180]
    angle = ((angle + 180.0) % 360.0) - 180.0
    return (cx, cy), (w, l), angle


def compute_2d_boxes(obj_masks: List[np.ndarray]) -> List[Dict]:
    """
    Compute oriented + axis-aligned boxes for each object mask.
    Returns list of dict:
      {
        "mask": obj_mask,
        "aabb": (x, y, w, h),
        "obb": {
            "center": (cx, cy),
            "size": (w, l),
            "angle": angle_deg,
            "corners": 4x2 float32 (clockwise)
        }
      }
    """
    results: List[Dict] = []
    for obj in obj_masks:
        cs, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        cnt = max(cs, key=cv2.contourArea)

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, l), angle = _normalize_rect(rect)
        box_pts = cv2.boxPoints(((cx, cy), (w, l), angle))  # OpenCV returns clockwise
        x, y, bw, bh = cv2.boundingRect(box_pts.astype(np.int32))

        results.append(
            {
                "mask": obj.copy(),
                "aabb": (int(x), int(y), int(bw), int(bh)),
                "obb": {
                    "center": (float(cx), float(cy)),
                    "size": (float(w)+3, float(l)+3),
                    "angle": float(-angle),
                    "corners": box_pts.astype(np.float32),
                },
            }
        )
    return results

def draw_2d_bboxes(bev_gray: np.ndarray, boxes_2d: List[Dict]) -> np.ndarray:
    """Overlay AABB + OBB on BEV grayscale."""
    vis = cv2.cvtColor((bev_gray * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for box in boxes_2d:
        x, y, w, h = box["aabb"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
        corners = box["obb"]["corners"].astype(np.int32)
        cv2.polylines(vis, [corners], True, (0, 128, 255), 2)
    return vis
