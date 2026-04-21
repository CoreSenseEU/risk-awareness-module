"""
ml.featextr

Feature extraction pipeline for risk awareness.

Orchestrates human detection (ByteTrack), absolute-depth proximity extraction
(RealSense), gaze scoring (2-D head pose), x-offset scoring, and approach-
velocity scoring into a single feature dictionary consumed by score.RiskScorer.

When ``depth_image_m`` is not provided (e.g. offline processing without a
depth sensor) proximity scores default to 0; all other sub-scores still work.
"""

from time import time

import numpy as np

from riskam.ml import depth as depth_mod, humandet
from riskam.ml.depth import D_SAFE_DEFAULT


def extract_human_risk_awareness_features(
    image: np.ndarray,
    depth_image_m: np.ndarray | None = None,
    d_safe: float = D_SAFE_DEFAULT,
    gaze_sigma_yaw: float = humandet.SIGMA_YAW_DEFAULT,
    gaze_sigma_pitch: float = humandet.SIGMA_PITCH_DEFAULT,
    gaze_frontal_pitch_ratio: float = humandet.FRONTAL_PITCH_RATIO_DEFAULT,
    track_bboxes: bool = True,
) -> tuple[list, np.ndarray | None, dict | None, list[int | None]]:
    """Extract per-person risk features from one frame.

    Parameters
    ----------
    image : np.ndarray  (BGR)
    depth_image_m : np.ndarray (H, W) float32, optional
        Absolute depth in metres from the RealSense sensor.
        Pass ``None`` when running offline; proximity scores will be 0.
    d_safe : float
        Safety distance (metres) used for proximity normalisation.
    gaze_sigma_yaw, gaze_sigma_pitch, gaze_frontal_pitch_ratio
        Head-pose gaze parameters forwarded to humandet.
    track_bboxes : bool
        Enable ByteTrack across frames.

    Returns
    -------
    human_bboxes : list of [x1, y1, x2, y2]
    depth_viz : np.ndarray (H, W, uint8) or None
        Visualisation-ready depth map (closer = brighter).
    risk_features : dict or None
        Keys: ``"proximity"``, ``"gaze"``, ``"x_offset"``, ``"approach"``.
        None if no humans are detected.
    track_ids : list[int | None]
    """
    t0 = time()

    human_bboxes, offset_scores, keypoints_np, track_ids = humandet.detect_humans(
        image, track_bboxes
    )

    # ── Depth / proximity ─────────────────────────────────────────────────────
    if depth_image_m is not None:
        bbox_proximities = depth_mod.extract_bbox_proximities(
            depth_image_m, human_bboxes, d_safe=d_safe
        )
        depth_viz = depth_mod.depth_to_visualization(depth_image_m, d_safe=d_safe)
    else:
        bbox_proximities = [0.0] * len(human_bboxes)
        depth_viz = None

    # ── Gaze ──────────────────────────────────────────────────────────────────
    gaze = humandet.gaze_scores(
        keypoints_np,
        sigma_yaw=gaze_sigma_yaw,
        sigma_pitch=gaze_sigma_pitch,
        frontal_pitch_ratio=gaze_frontal_pitch_ratio,
    )

    # ── Velocity / approach ───────────────────────────────────────────────────
    if depth_image_m is not None:
        humandet.update_velocity(track_ids, bbox_proximities)
    approach = humandet.approach_scores(track_ids)

    # ── Assemble ──────────────────────────────────────────────────────────────
    _ = time() - t0  # timing available for diagnostics if needed

    if len(human_bboxes) == 0:
        return human_bboxes, depth_viz, None, track_ids

    risk_features = {
        "proximity": np.array(bbox_proximities, dtype=float),
        "x_offset": np.array(offset_scores, dtype=float),
        "gaze": np.array(gaze, dtype=float),
        "approach": np.array(approach, dtype=float),
    }

    return human_bboxes, depth_viz, risk_features, track_ids
