"""
ml.depth

Absolute depth processing from a robot-mounted depth sensor.

The default configuration is calibrated for the RealSense D4xx family — clean
absolute depth from ~0.3 m onwards, zero-pixels rare and treated as
"unmeasured / far". Two extra parameters extend coverage to sensors with a
hard near-clip dead zone (e.g. PrimeSense Xtion / Carmine) without changing
RealSense behaviour:

- ``depth_near_clip_m`` (default 0.0): the closest distance the sensor can
  measure. ``0.0`` disables the dead-zone fallback → identical to the
  pre-T2.7 behaviour. A positive value (e.g. 0.6 m for Xtion) tells RiskAM
  that "all-zero depth in a person-shaped bbox" plausibly means "person is
  closer than the sensor's near-clip" rather than "no person there".
- ``near_clip_bbox_min_frac`` (default 0.05): minimum bbox area fraction
  (of the full image) for the dead-zone fallback to fire. Guards against
  treating distant noise-filled bboxes as "very close".

The fallback assigns ``depth = 0`` (max proximity) when both conditions hold
— a safety-conservative choice given the bbox could be anywhere in the
[0, near_clip] dead zone. With ``depth_near_clip_m = 0`` the new branch is
unreachable and ``extract_bbox_depths`` falls back to ``d_safe`` exactly as
before.
"""

import numpy as np


# 10th-percentile within a bounding box → closest *valid* depth in that region
DEPTH_PERCENTILE = 10

# Default safety distance in metres.
# Proximity score = max(0, 1 - depth_m / d_safe).
D_SAFE_DEFAULT = 1.5

# Pixels below this threshold are treated as missing / sensor noise (metres).
DEPTH_MIN_M = 0.1

# Default near-clip distance in metres. 0.0 = no dead-zone fallback (RealSense
# behaviour pre-T2.7). Set to the sensor's spec-sheet near-clip (e.g. 0.6 for
# Xtion, ~0.3 for D435) to enable the close-fallback semantics.
NEAR_CLIP_M_DEFAULT = 0.0

# Default minimum bbox-area fraction for the close-fallback to trigger. 5% of
# the frame is roughly the on-screen footprint of a standing adult at the
# Xtion's near-clip distance — smaller bboxes with zero valid depth are more
# plausibly "noise / occluded background" than "person too close to measure".
NEAR_CLIP_BBOX_MIN_FRAC_DEFAULT = 0.05

# Maximum valid-pixel fraction *within* a bbox for the close-fallback to fire.
# Default 0.0 reproduces the pre-T2.7 / RealSense condition exactly (fallback
# only on strictly zero valid pixels). Structured-light sensors like Xtion
# routinely return a few percent of valid background pixels even for persons
# in the dead zone — the percentile of those stray pixels is unreliable
# (likely background bleed-through, not the person), so a small positive
# threshold (e.g. 0.05 for Xtion) lets the close-fallback fire on "mostly
# empty" bboxes too. Combined with NEAR_CLIP_BBOX_MIN_FRAC_DEFAULT the
# decision is "large bbox + sparse valid signal → person likely in dead zone".
NEAR_CLIP_VALID_FRAC_MAX_DEFAULT = 0.0

# Percentile of valid-depth pixels used as the upper bound when visualising
# scenes whose working range exceeds ``d_safe`` (pre-T2.7 the visualisation
# clipped strictly at ``d_safe``).
VIZ_UPPER_PERCENTILE = 95


def depth_mm_to_m(depth_image_mm: np.ndarray) -> np.ndarray:
    """Convert a RealSense uint16 depth image (millimetres) to float32 metres.

    Zero pixels are preserved as zeros (missing-data sentinel).
    """
    out = depth_image_mm.astype(np.float32) / 1000.0
    out[depth_image_mm == 0] = 0.0
    return out


def extract_bbox_depths(
    depth_image_m: np.ndarray,
    human_bboxes: list,
    d_safe: float = D_SAFE_DEFAULT,
    depth_near_clip_m: float = NEAR_CLIP_M_DEFAULT,
    near_clip_bbox_min_frac: float = NEAR_CLIP_BBOX_MIN_FRAC_DEFAULT,
    near_clip_valid_frac_max: float = NEAR_CLIP_VALID_FRAC_MAX_DEFAULT,
) -> list[float]:
    """Extract a single representative depth (metres) per bounding box.

    Uses the 10th-percentile within the bbox as the "closest valid surface".

    When a bbox has *too few* valid depth pixels — concretely, when the
    valid-pixel fraction is at or below ``near_clip_valid_frac_max`` — two
    fallback paths exist:

    - **Close fallback** (T2.7): ``depth_near_clip_m > 0`` AND the bbox
      occupies at least ``near_clip_bbox_min_frac`` of the full frame.
      This captures the Xtion / structured-light failure mode where a
      person standing inside the sensor's near-clip dead zone returns
      all-or-mostly zeros despite being the closest object in the scene.
      Returns ``0.0`` (max proximity) — safety-conservative, since the
      person is somewhere in [0, near_clip] but we cannot measure where.

    - **Far fallback** (legacy, RealSense default): otherwise, returns
      ``d_safe`` (proximity 0). With ``depth_near_clip_m = 0`` and
      ``near_clip_valid_frac_max = 0`` (defaults), the close-fallback
      branch is unreachable, the percentile path runs whenever any valid
      pixel exists, and behaviour is bit-identical to pre-T2.7.
    """
    image_area = depth_image_m.shape[0] * depth_image_m.shape[1]
    depths: list[float] = []
    for bbox in human_bboxes:
        x1, y1, x2, y2 = (int(c) for c in bbox)
        region = depth_image_m[y1:y2, x1:x2]
        valid = region[(region >= DEPTH_MIN_M) & np.isfinite(region)]
        valid_frac = valid.size / region.size if region.size > 0 else 0.0
        if valid_frac > near_clip_valid_frac_max:
            depths.append(float(np.percentile(valid, DEPTH_PERCENTILE)))
            continue
        bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
        triggers_close_fallback = (
            depth_near_clip_m > 0.0
            and image_area > 0
            and (bbox_area / image_area) >= near_clip_bbox_min_frac
        )
        depths.append(0.0 if triggers_close_fallback else d_safe)
    return depths


def depths_to_proximities(
    depths_m: list[float],
    d_safe: float = D_SAFE_DEFAULT,
) -> list[float]:
    """Linear proximity from depth: 1 at the camera, 0 at/beyond ``d_safe``."""
    return [float(np.clip(1.0 - d / d_safe, 0.0, 1.0)) for d in depths_m]


def extract_bbox_proximities(
    depth_image_m: np.ndarray,
    human_bboxes: list,
    d_safe: float = D_SAFE_DEFAULT,
    depth_near_clip_m: float = NEAR_CLIP_M_DEFAULT,
    near_clip_bbox_min_frac: float = NEAR_CLIP_BBOX_MIN_FRAC_DEFAULT,
    near_clip_valid_frac_max: float = NEAR_CLIP_VALID_FRAC_MAX_DEFAULT,
) -> list[float]:
    """Compute per-person proximity scores from an absolute depth image.

    Parameters
    ----------
    depth_image_m : np.ndarray (H, W) float32
        Depth in metres.  Zeros and NaNs indicate missing data.
    human_bboxes : list of [x1, y1, x2, y2]
    d_safe : float
        Safety distance in metres.  A person at or beyond d_safe scores 0.
    depth_near_clip_m : float
        Sensor near-clip in metres. ``0`` (default) disables the close-fallback
        — pre-T2.7 / RealSense behaviour. Positive values (e.g. 0.6 for Xtion)
        enable the dead-zone fallback for large bboxes with no valid depth.
    near_clip_bbox_min_frac : float
        Minimum bbox area fraction (of the full frame) for the close-fallback
        to fire. Ignored when ``depth_near_clip_m == 0``.
    near_clip_valid_frac_max : float
        Maximum valid-pixel fraction within a bbox for the close-fallback to
        fire. ``0`` (default) means strictly zero valid pixels (RealSense).
        Set to a small positive value (e.g. 0.05 for Xtion) to also fire on
        "mostly empty" bboxes whose few valid pixels are likely background
        bleed-through, not the person.

    Returns
    -------
    list[float]
        Proximity score in [0, 1] per bounding box.
        1 = person is at the camera, 0 = person is at or beyond d_safe.
    """
    return depths_to_proximities(
        extract_bbox_depths(
            depth_image_m,
            human_bboxes,
            d_safe=d_safe,
            depth_near_clip_m=depth_near_clip_m,
            near_clip_bbox_min_frac=near_clip_bbox_min_frac,
            near_clip_valid_frac_max=near_clip_valid_frac_max,
        ),
        d_safe=d_safe,
    )


def depth_to_visualization(
    depth_image_m: np.ndarray,
    d_safe: float = D_SAFE_DEFAULT,
) -> np.ndarray:
    """Convert an absolute depth image to an 8-bit single-channel map.

    Closer pixels are brighter (matching the former MiDaS convention).
    Missing data (zeros) is rendered as black.

    The normalisation upper bound is ``max(d_safe, p95 of valid pixels)`` so
    the visualisation remains depth-varied on sensors whose working range
    exceeds ``d_safe`` (e.g. Xtion: valid pixels typically 2–5 m). When all
    valid pixels are within ``d_safe`` (the standard RealSense / SamXL
    deployment case) the upper bound stays at ``d_safe`` and behaviour is
    bit-identical to the pre-T2.7 visualisation.
    """
    valid = depth_image_m[(depth_image_m > 0) & np.isfinite(depth_image_m)]
    upper = d_safe
    if valid.size > 0:
        upper = max(d_safe, float(np.percentile(valid, VIZ_UPPER_PERCENTILE)))
    normalized = np.clip(depth_image_m / upper, 0.0, 1.0)
    inverted = 1.0 - normalized
    inverted[depth_image_m == 0] = 0.0
    return (inverted * 255).astype(np.uint8)
