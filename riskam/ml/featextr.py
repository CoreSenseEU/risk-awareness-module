"""
ml.featextr

Feature extraction pipeline for risk awareness.

The pipeline is split in two so the parameter-independent half can be
cached on disk:

  1. :func:`extract_primitives` — runs YOLO/pose, ByteTrack matching, and
     depth-per-bbox extraction. Output is fully determined by ``(model,
     RGB frame, depth frame)`` and is cached by :mod:`riskam.feature_cache`.
  2. :func:`extract_features` — converts those primitives into the four
     risk sub-scores (proximity, gaze, x_offset, approach) using the
     parameters of the current sweep cell. Cheap (~µs).

:func:`extract` is a convenience wrapper that runs both back-to-back.
:func:`extract_with_cache` is the same wrapper plus a cache lookup.

Required inputs (enforced at entry): RGB + absolute depth. Optional: cmd_vel.
"""

from __future__ import annotations

import numpy as np

from riskam.feature_cache import CachedFrameFeatures, FeatureCache
from riskam.ml import depth as depth_mod, humandet
from riskam.ml.depth import (
    D_SAFE_DEFAULT,
    NEAR_CLIP_BBOX_MIN_FRAC_DEFAULT,
    NEAR_CLIP_M_DEFAULT,
    NEAR_CLIP_VALID_FRAC_MAX_DEFAULT,
)
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


# ── Parameter-independent half: cacheable ────────────────────────────────────


def extract_primitives(
    inputs: FrameInputs,
    d_safe: float = D_SAFE_DEFAULT,
    depth_near_clip_m: float = NEAR_CLIP_M_DEFAULT,
    near_clip_bbox_min_frac: float = NEAR_CLIP_BBOX_MIN_FRAC_DEFAULT,
    near_clip_valid_frac_max: float = NEAR_CLIP_VALID_FRAC_MAX_DEFAULT,
    track_bboxes: bool = True,
) -> CachedFrameFeatures:
    """Run YOLO/pose detection, ByteTrack, and depth-per-bbox extraction.

    Output is determined by ``(model_state, rgb, depth_m)`` and the
    ``track_bboxes`` flag for the parameter-independent half. ``d_safe`` and
    the close-fallback knobs (``depth_near_clip_m``, ``near_clip_bbox_min_frac``)
    only affect the missing-depth fallback inside :func:`extract_bbox_depths`,
    so cached primitives reflect the calibration in force at cache write time.
    """
    inputs.validate()

    bboxes, keypoints, track_ids = humandet.detect_humans(
        inputs.rgb, track_bboxes
    )
    depth_viz = depth_mod.depth_to_visualization(inputs.depth_m, d_safe=d_safe)

    if not bboxes:
        return CachedFrameFeatures(
            human_bboxes=[],
            keypoints_np=None,
            track_ids=[],
            bbox_depths_m=[],
            depth_viz=depth_viz,
        )

    bbox_depths_m = depth_mod.extract_bbox_depths(
        inputs.depth_m,
        bboxes,
        d_safe=d_safe,
        depth_near_clip_m=depth_near_clip_m,
        near_clip_bbox_min_frac=near_clip_bbox_min_frac,
        near_clip_valid_frac_max=near_clip_valid_frac_max,
    )
    return CachedFrameFeatures(
        human_bboxes=bboxes,
        keypoints_np=keypoints,
        track_ids=track_ids,
        bbox_depths_m=bbox_depths_m,
        depth_viz=depth_viz,
    )


# ── Parameter-dependent half: cheap, never cached ────────────────────────────


def extract_features(
    primitives: CachedFrameFeatures,
    image_shape: tuple[int, int],
    cmd_vel: RobotVelocity | None = None,
    d_safe: float = D_SAFE_DEFAULT,
    gaze_sigma_yaw: float = humandet.SIGMA_YAW_DEFAULT,
    gaze_sigma_pitch: float = humandet.SIGMA_PITCH_DEFAULT,
    gaze_frontal_pitch_ratio: float = humandet.FRONTAL_PITCH_RATIO_DEFAULT,
    gaze_algorithm: str = humandet.GAZE_ALGORITHM_DEFAULT,
) -> FrameExtraction:
    """Compute the four sub-scores from cached primitives + scoring params.

    ``image_shape`` is ``(H, W)`` — needed for the x_offset normalisation.
    """
    if not primitives.human_bboxes:
        return FrameExtraction(
            human_bboxes=[],
            depth_viz=primitives.depth_viz,
            features=None,
            subscore_status={
                "proximity": SubScoreStatus.ACTIVE,
                "gaze": SubScoreStatus.ACTIVE,
                "x_offset": (
                    SubScoreStatus.ACTIVE
                    if cmd_vel is not None
                    else SubScoreStatus.FALLBACK
                ),
                "approach": SubScoreStatus.UNAVAILABLE,
            },
            subscore_reasons={"approach": "no detections"},
            track_ids=[],
        )

    proximity = compute_proximity(primitives.bbox_depths_m, d_safe=d_safe)
    gaze = compute_gaze(
        primitives.keypoints_np,
        sigma_yaw=gaze_sigma_yaw,
        sigma_pitch=gaze_sigma_pitch,
        frontal_pitch_ratio=gaze_frontal_pitch_ratio,
        algorithm=gaze_algorithm,
    )
    image_h, image_w = image_shape
    x_offset = compute_x_offset(
        primitives.human_bboxes, image_w, image_h, cmd_vel
    )

    humandet.update_velocity(primitives.track_ids, primitives.bbox_depths_m)
    approach = compute_approach(primitives.track_ids)

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
        human_bboxes=primitives.human_bboxes,
        depth_viz=primitives.depth_viz,
        features=features,
        subscore_status=status,
        subscore_reasons=reasons,
        track_ids=primitives.track_ids,
    )


# ── Convenience wrappers ─────────────────────────────────────────────────────


def extract(
    inputs: FrameInputs,
    d_safe: float = D_SAFE_DEFAULT,
    depth_near_clip_m: float = NEAR_CLIP_M_DEFAULT,
    near_clip_bbox_min_frac: float = NEAR_CLIP_BBOX_MIN_FRAC_DEFAULT,
    near_clip_valid_frac_max: float = NEAR_CLIP_VALID_FRAC_MAX_DEFAULT,
    gaze_sigma_yaw: float = humandet.SIGMA_YAW_DEFAULT,
    gaze_sigma_pitch: float = humandet.SIGMA_PITCH_DEFAULT,
    gaze_frontal_pitch_ratio: float = humandet.FRONTAL_PITCH_RATIO_DEFAULT,
    gaze_algorithm: str = humandet.GAZE_ALGORITHM_DEFAULT,
    track_bboxes: bool = True,
) -> FrameExtraction:
    """Run the full per-frame feature-extraction pipeline (no cache)."""
    primitives = extract_primitives(
        inputs,
        d_safe=d_safe,
        depth_near_clip_m=depth_near_clip_m,
        near_clip_bbox_min_frac=near_clip_bbox_min_frac,
        near_clip_valid_frac_max=near_clip_valid_frac_max,
        track_bboxes=track_bboxes,
    )
    return extract_features(
        primitives,
        image_shape=inputs.rgb.shape[:2],
        cmd_vel=inputs.cmd_vel,
        d_safe=d_safe,
        gaze_sigma_yaw=gaze_sigma_yaw,
        gaze_sigma_pitch=gaze_sigma_pitch,
        gaze_frontal_pitch_ratio=gaze_frontal_pitch_ratio,
        gaze_algorithm=gaze_algorithm,
    )


def extract_with_cache(
    inputs: FrameInputs,
    cache: FeatureCache,
    run: str,
    rgb_stem: str,
    d_safe: float = D_SAFE_DEFAULT,
    depth_near_clip_m: float = NEAR_CLIP_M_DEFAULT,
    near_clip_bbox_min_frac: float = NEAR_CLIP_BBOX_MIN_FRAC_DEFAULT,
    near_clip_valid_frac_max: float = NEAR_CLIP_VALID_FRAC_MAX_DEFAULT,
    gaze_sigma_yaw: float = humandet.SIGMA_YAW_DEFAULT,
    gaze_sigma_pitch: float = humandet.SIGMA_PITCH_DEFAULT,
    gaze_frontal_pitch_ratio: float = humandet.FRONTAL_PITCH_RATIO_DEFAULT,
    gaze_algorithm: str = humandet.GAZE_ALGORITHM_DEFAULT,
    track_bboxes: bool = True,
) -> FrameExtraction:
    """Run the pipeline using a feature cache for the inference half.

    Cache miss → run :func:`extract_primitives` and write back; cache hit
    → load primitives from disk and skip YOLO/ByteTrack entirely.
    """
    primitives = cache.get(run, rgb_stem)
    if primitives is None:
        primitives = extract_primitives(
            inputs,
            d_safe=d_safe,
            depth_near_clip_m=depth_near_clip_m,
            near_clip_bbox_min_frac=near_clip_bbox_min_frac,
            near_clip_valid_frac_max=near_clip_valid_frac_max,
            track_bboxes=track_bboxes,
        )
        cache.put(run, rgb_stem, primitives)
    return extract_features(
        primitives,
        image_shape=inputs.rgb.shape[:2],
        cmd_vel=inputs.cmd_vel,
        d_safe=d_safe,
        gaze_sigma_yaw=gaze_sigma_yaw,
        gaze_sigma_pitch=gaze_sigma_pitch,
        gaze_frontal_pitch_ratio=gaze_frontal_pitch_ratio,
        gaze_algorithm=gaze_algorithm,
    )


def extract_human_risk_awareness_features(
    image: np.ndarray,
    depth_image_m: np.ndarray,
    cmd_vel: RobotVelocity | None = None,
    d_safe: float = D_SAFE_DEFAULT,
    depth_near_clip_m: float = NEAR_CLIP_M_DEFAULT,
    near_clip_bbox_min_frac: float = NEAR_CLIP_BBOX_MIN_FRAC_DEFAULT,
    near_clip_valid_frac_max: float = NEAR_CLIP_VALID_FRAC_MAX_DEFAULT,
    gaze_sigma_yaw: float = humandet.SIGMA_YAW_DEFAULT,
    gaze_sigma_pitch: float = humandet.SIGMA_PITCH_DEFAULT,
    gaze_frontal_pitch_ratio: float = humandet.FRONTAL_PITCH_RATIO_DEFAULT,
    gaze_algorithm: str = humandet.GAZE_ALGORITHM_DEFAULT,
    track_bboxes: bool = True,
) -> FrameExtraction:
    """Backwards-compatible wrapper around :func:`extract`."""
    return extract(
        FrameInputs(rgb=image, depth_m=depth_image_m, cmd_vel=cmd_vel),
        d_safe=d_safe,
        depth_near_clip_m=depth_near_clip_m,
        near_clip_bbox_min_frac=near_clip_bbox_min_frac,
        near_clip_valid_frac_max=near_clip_valid_frac_max,
        gaze_sigma_yaw=gaze_sigma_yaw,
        gaze_sigma_pitch=gaze_sigma_pitch,
        gaze_frontal_pitch_ratio=gaze_frontal_pitch_ratio,
        gaze_algorithm=gaze_algorithm,
        track_bboxes=track_bboxes,
    )
