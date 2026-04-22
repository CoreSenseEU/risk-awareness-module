"""
ml.featextr

Feature extraction pipeline for risk awareness.

Orchestrates human detection (ByteTrack), per-bbox depth extraction, and the
four risk sub-scores (proximity, gaze, x_offset, approach) into a single
:class:`FrameExtraction` result.

Required inputs (enforced at entry): RGB image + absolute depth (metres).
Optional inputs: ``cmd_vel`` enables the path-aware x_offset. Without it
x_offset falls back to the centre-offset heuristic (status = FALLBACK).

The legacy call signature ``extract_human_risk_awareness_features(image,
depth_image_m=..., ...)`` is preserved for backwards compatibility; it just
assembles a :class:`FrameInputs` and delegates to :func:`extract`.
"""

import numpy as np

from riskam.ml import depth as depth_mod, humandet
from riskam.ml.depth import D_SAFE_DEFAULT
from riskam.ml.subscores import (
    FrameExtraction,
    FrameInputs,
    RobotVelocity,
    SubScoreStatus,
    compute_approach,
    compute_gaze,
    compute_proximity,
    compute_x_offset,
)


def extract(
    inputs: FrameInputs,
    d_safe: float = D_SAFE_DEFAULT,
    gaze_sigma_yaw: float = humandet.SIGMA_YAW_DEFAULT,
    gaze_sigma_pitch: float = humandet.SIGMA_PITCH_DEFAULT,
    gaze_frontal_pitch_ratio: float = humandet.FRONTAL_PITCH_RATIO_DEFAULT,
    gaze_algorithm: str = humandet.GAZE_ALGORITHM_DEFAULT,
    track_bboxes: bool = True,
) -> FrameExtraction:
    """Run the full per-frame feature-extraction pipeline.

    Parameters
    ----------
    inputs : FrameInputs
        RGB + depth (required), cmd_vel (optional).
    """
    inputs.validate()

    human_bboxes, keypoints_np, track_ids = humandet.detect_humans(
        inputs.rgb, track_bboxes
    )

    # Depth visualisation is cheap and always available (depth is required).
    depth_viz = depth_mod.depth_to_visualization(inputs.depth_m, d_safe=d_safe)

    if len(human_bboxes) == 0:
        return FrameExtraction(
            human_bboxes=[],
            depth_viz=depth_viz,
            features=None,
            subscore_status={
                "proximity": SubScoreStatus.ACTIVE,
                "gaze": SubScoreStatus.ACTIVE,
                "x_offset": (
                    SubScoreStatus.ACTIVE
                    if inputs.cmd_vel is not None
                    else SubScoreStatus.FALLBACK
                ),
                "approach": SubScoreStatus.UNAVAILABLE,
            },
            subscore_reasons={"approach": "no detections"},
            track_ids=[],
        )

    # Depth-per-bbox in metres — fed to both the proximity sub-score *and*
    # to the velocity tracker so slope has correct m/s units.
    bbox_depths_m = depth_mod.extract_bbox_depths(
        inputs.depth_m, human_bboxes, d_safe=d_safe
    )

    proximity = compute_proximity(human_bboxes, inputs.depth_m, d_safe=d_safe)
    gaze = compute_gaze(
        keypoints_np,
        sigma_yaw=gaze_sigma_yaw,
        sigma_pitch=gaze_sigma_pitch,
        frontal_pitch_ratio=gaze_frontal_pitch_ratio,
        algorithm=gaze_algorithm,
    )

    image_h, image_w = inputs.rgb.shape[:2]
    x_offset = compute_x_offset(human_bboxes, image_w, image_h, inputs.cmd_vel)

    humandet.update_velocity(track_ids, bbox_depths_m)
    approach = compute_approach(track_ids)

    features = {
        "proximity": proximity.values,
        "gaze": gaze.values,
        "x_offset": x_offset.values,
        "approach": approach.values,
    }
    status = {
        "proximity": proximity.status,
        "gaze": gaze.status,
        "x_offset": x_offset.status,
        "approach": approach.status,
    }
    reasons = {
        name: r.reason
        for name, r in (
            ("proximity", proximity),
            ("gaze", gaze),
            ("x_offset", x_offset),
            ("approach", approach),
        )
        if r.reason
    }

    return FrameExtraction(
        human_bboxes=human_bboxes,
        depth_viz=depth_viz,
        features=features,
        subscore_status=status,
        subscore_reasons=reasons,
        track_ids=track_ids,
    )


def extract_human_risk_awareness_features(
    image: np.ndarray,
    depth_image_m: np.ndarray,
    cmd_vel: RobotVelocity | None = None,
    d_safe: float = D_SAFE_DEFAULT,
    gaze_sigma_yaw: float = humandet.SIGMA_YAW_DEFAULT,
    gaze_sigma_pitch: float = humandet.SIGMA_PITCH_DEFAULT,
    gaze_frontal_pitch_ratio: float = humandet.FRONTAL_PITCH_RATIO_DEFAULT,
    gaze_algorithm: str = humandet.GAZE_ALGORITHM_DEFAULT,
    track_bboxes: bool = True,
) -> FrameExtraction:
    """Backwards-compatible wrapper around :func:`extract`.

    Older callers can keep their positional-argument style; the wrapper just
    constructs a :class:`FrameInputs` and returns the orchestrator output.
    """
    return extract(
        FrameInputs(rgb=image, depth_m=depth_image_m, cmd_vel=cmd_vel),
        d_safe=d_safe,
        gaze_sigma_yaw=gaze_sigma_yaw,
        gaze_sigma_pitch=gaze_sigma_pitch,
        gaze_frontal_pitch_ratio=gaze_frontal_pitch_ratio,
        gaze_algorithm=gaze_algorithm,
        track_bboxes=track_bboxes,
    )
