"""
ml.depth

Absolute depth processing from the RealSense depth sensor.

Replaces the former MiDaS-based relative depth estimation.  The RealSense
D4xx series publishes depth as a 16-bit image where each pixel encodes
distance in millimetres.  Converting to metres and comparing against a
configurable safety distance yields a physically interpretable, scene-stable
proximity score.
"""

import numpy as np


# 10th-percentile within a bounding box → closest *valid* depth in that region
DEPTH_PERCENTILE = 10

# Default safety distance in metres.
# Proximity score = max(0, 1 - depth_m / d_safe).
D_SAFE_DEFAULT = 1.5

# Pixels below this threshold are treated as missing / sensor noise (metres).
DEPTH_MIN_M = 0.1


def depth_mm_to_m(depth_image_mm: np.ndarray) -> np.ndarray:
    """Convert a RealSense uint16 depth image (millimetres) to float32 metres.

    Zero pixels are preserved as zeros (missing-data sentinel).
    """
    out = depth_image_mm.astype(np.float32) / 1000.0
    out[depth_image_mm == 0] = 0.0
    return out


def extract_bbox_proximities(
    depth_image_m: np.ndarray,
    human_bboxes: list,
    d_safe: float = D_SAFE_DEFAULT,
) -> list[float]:
    """Compute per-person proximity scores from an absolute depth image.

    Parameters
    ----------
    depth_image_m : np.ndarray (H, W) float32
        Depth in metres.  Zeros and NaNs indicate missing data.
    human_bboxes : list of [x1, y1, x2, y2]
    d_safe : float
        Safety distance in metres.  A person at or beyond d_safe scores 0.

    Returns
    -------
    list[float]
        Proximity score in [0, 1] per bounding box.
        1 = person is at the camera, 0 = person is at or beyond d_safe.
    """
    scores = []
    for bbox in human_bboxes:
        x1, y1, x2, y2 = (int(c) for c in bbox)
        region = depth_image_m[y1:y2, x1:x2]
        valid = region[(region >= DEPTH_MIN_M) & np.isfinite(region)]
        if valid.size > 0:
            depth_m = float(np.percentile(valid, DEPTH_PERCENTILE))
        else:
            depth_m = d_safe  # no valid depth → treat as outside safety zone
        scores.append(float(np.clip(1.0 - depth_m / d_safe, 0.0, 1.0)))
    return scores


def depth_to_visualization(
    depth_image_m: np.ndarray,
    d_safe: float = D_SAFE_DEFAULT,
) -> np.ndarray:
    """Convert an absolute depth image to an 8-bit single-channel map.

    Closer pixels are brighter (matching the former MiDaS convention).
    Missing data (zeros) is rendered as black.
    """
    normalized = np.clip(depth_image_m / d_safe, 0.0, 1.0)
    inverted = 1.0 - normalized
    inverted[depth_image_m == 0] = 0.0
    return (inverted * 255).astype(np.uint8)
