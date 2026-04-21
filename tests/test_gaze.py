"""Unit tests for the 2-D head-pose gaze estimation in ml.humandet."""

import math

import numpy as np
import pytest

from riskam.ml.humandet import (
    FRONTAL_PITCH_RATIO_DEFAULT,
    SIGMA_PITCH_DEFAULT,
    SIGMA_YAW_DEFAULT,
    _headpose_gaze,
    gaze_scores,
)


def _make_kpts(
    nose_x=100.0, nose_y=120.0,
    leye_x=115.0, leye_y=100.0,   # person's left eye = image right
    reye_x=85.0,  reye_y=100.0,   # person's right eye = image left
    n_total=17,
) -> np.ndarray:
    """Build a minimal 17-keypoint array for one person."""
    kpts = np.zeros((n_total, 2), dtype=float)
    kpts[0] = [nose_x, nose_y]
    kpts[1] = [leye_x, leye_y]
    kpts[2] = [reye_x, reye_y]
    return kpts


class TestHeadposeGaze:
    """Direct tests for the _headpose_gaze private helper."""

    def test_frontal_face_high_score(self):
        # Symmetric face, nose 0.7 × inter-eye below eyes.
        inter_eye = 30.0
        nose_y_offset = FRONTAL_PITCH_RATIO_DEFAULT * inter_eye
        kpts = _make_kpts(
            nose_x=100.0, nose_y=100.0 + nose_y_offset,
            leye_x=115.0, leye_y=100.0,
            reye_x=85.0,  reye_y=100.0,
        )
        score = _headpose_gaze(kpts, SIGMA_YAW_DEFAULT, SIGMA_PITCH_DEFAULT, FRONTAL_PITCH_RATIO_DEFAULT)
        assert score > 0.8, f"Expected high score for frontal face, got {score:.3f}"

    def test_strong_yaw_low_score(self):
        # Nose shifted 2 σ-yaw widths to the right → low gaze score.
        inter_eye = 30.0
        yaw_offset = 2.0 * SIGMA_YAW_DEFAULT * inter_eye
        nose_y_offset = FRONTAL_PITCH_RATIO_DEFAULT * inter_eye
        kpts = _make_kpts(
            nose_x=100.0 + yaw_offset, nose_y=100.0 + nose_y_offset,
            leye_x=115.0, leye_y=100.0,
            reye_x=85.0,  reye_y=100.0,
        )
        score = _headpose_gaze(kpts, SIGMA_YAW_DEFAULT, SIGMA_PITCH_DEFAULT, FRONTAL_PITCH_RATIO_DEFAULT)
        # exp(-2^2/2) ≈ 0.14 for yaw, any pitch score ≤ 1 → combined ≤ 0.14
        assert score < 0.2, f"Expected low score for large yaw, got {score:.3f}"

    def test_strong_pitch_low_score(self):
        # Nose far below expected (person looking up / tilted back).
        inter_eye = 30.0
        nose_y_offset = (FRONTAL_PITCH_RATIO_DEFAULT + 2.0 * SIGMA_PITCH_DEFAULT) * inter_eye
        kpts = _make_kpts(
            nose_x=100.0, nose_y=100.0 + nose_y_offset,
            leye_x=115.0, leye_y=100.0,
            reye_x=85.0,  reye_y=100.0,
        )
        score = _headpose_gaze(kpts, SIGMA_YAW_DEFAULT, SIGMA_PITCH_DEFAULT, FRONTAL_PITCH_RATIO_DEFAULT)
        assert score < 0.2, f"Expected low score for large pitch deviation, got {score:.3f}"

    def test_zero_keypoints_return_zero(self):
        kpts = np.zeros((17, 2), dtype=float)
        score = _headpose_gaze(kpts, SIGMA_YAW_DEFAULT, SIGMA_PITCH_DEFAULT, FRONTAL_PITCH_RATIO_DEFAULT)
        assert score == 0.0

    def test_degenerate_inter_eye_distance(self):
        kpts = _make_kpts(leye_x=100.0, reye_x=100.0)  # both eyes at same position
        score = _headpose_gaze(kpts, SIGMA_YAW_DEFAULT, SIGMA_PITCH_DEFAULT, FRONTAL_PITCH_RATIO_DEFAULT)
        assert score == 0.0

    def test_score_in_range(self):
        kpts = _make_kpts()
        score = _headpose_gaze(kpts, SIGMA_YAW_DEFAULT, SIGMA_PITCH_DEFAULT, FRONTAL_PITCH_RATIO_DEFAULT)
        assert 0.0 <= score <= 1.0

    def test_symmetric_yaw_is_zero(self):
        """Nose perfectly centred between eyes should produce zero yaw offset."""
        inter_eye = 30.0
        mid_x = 100.0
        kpts = _make_kpts(
            nose_x=mid_x,
            leye_x=mid_x + inter_eye / 2,
            reye_x=mid_x - inter_eye / 2,
            nose_y=100.0 + FRONTAL_PITCH_RATIO_DEFAULT * inter_eye,
            leye_y=100.0, reye_y=100.0,
        )
        score = _headpose_gaze(kpts, SIGMA_YAW_DEFAULT, SIGMA_PITCH_DEFAULT, FRONTAL_PITCH_RATIO_DEFAULT)
        expected_yaw = math.exp(0)  # yaw_offset = 0 → exp(0) = 1
        # combined = 1.0 * pitch_score (pitch should be near 1 as well)
        assert score > 0.8

    def test_insufficient_keypoints(self):
        kpts = np.zeros((2, 2), dtype=float)
        score = _headpose_gaze(kpts, SIGMA_YAW_DEFAULT, SIGMA_PITCH_DEFAULT, FRONTAL_PITCH_RATIO_DEFAULT)
        assert score == 0.0


class TestGazeScoresBatch:
    def test_none_returns_empty(self):
        assert gaze_scores(None) == []

    def test_empty_array_returns_empty(self):
        assert gaze_scores(np.zeros((0, 17, 2))) == []

    def test_batch_length_matches_persons(self):
        kpts = np.stack([_make_kpts(), _make_kpts()])
        scores = gaze_scores(kpts)
        assert len(scores) == 2

    def test_all_scores_in_range(self):
        kpts = np.stack([_make_kpts(), _make_kpts(nose_x=150)])
        scores = gaze_scores(kpts)
        for s in scores:
            assert 0.0 <= s <= 1.0
