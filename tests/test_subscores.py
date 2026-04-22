"""Unit tests for riskam.ml.subscores — the sub-score input-availability contract."""

from types import SimpleNamespace

import numpy as np
import pytest

from riskam.ml import humandet
from riskam.ml.subscores import (
    FrameInputs,
    RobotVelocity,
    SubScoreStatus,
    compute_approach,
    compute_gaze,
    compute_proximity,
    compute_x_offset,
)


# ── FrameInputs validation ───────────────────────────────────────────────────


class TestFrameInputsValidation:
    def test_valid_inputs_pass(self):
        rgb = np.zeros((10, 20, 3), dtype=np.uint8)
        depth = np.full((10, 20), 1.0, dtype=np.float32)
        FrameInputs(rgb=rgb, depth_m=depth).validate()  # should not raise

    def test_missing_rgb_raises(self):
        depth = np.full((10, 20), 1.0, dtype=np.float32)
        with pytest.raises(ValueError, match="rgb is required"):
            FrameInputs(rgb=None, depth_m=depth).validate()

    def test_missing_depth_raises(self):
        rgb = np.zeros((10, 20, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="depth_m is required"):
            FrameInputs(rgb=rgb, depth_m=None).validate()

    def test_wrong_rgb_shape_raises(self):
        rgb = np.zeros((10, 20), dtype=np.uint8)  # 2-D
        depth = np.full((10, 20), 1.0, dtype=np.float32)
        with pytest.raises(ValueError, match="rgb must be 3-D"):
            FrameInputs(rgb=rgb, depth_m=depth).validate()

    def test_wrong_depth_shape_raises(self):
        rgb = np.zeros((10, 20, 3), dtype=np.uint8)
        depth = np.zeros((10, 20, 3), dtype=np.float32)  # 3-D
        with pytest.raises(ValueError, match="depth_m must be 2-D"):
            FrameInputs(rgb=rgb, depth_m=depth).validate()


# ── RobotVelocity ────────────────────────────────────────────────────────────


class TestRobotVelocity:
    def test_speed(self):
        v = RobotVelocity(linear_x=0.3, linear_y=0.4)
        assert v.speed == pytest.approx(0.5)

    def test_from_twist(self):
        twist = SimpleNamespace(linear=SimpleNamespace(x=0.2, y=-0.1))
        v = RobotVelocity.from_twist(twist)
        assert v.linear_x == pytest.approx(0.2)
        assert v.linear_y == pytest.approx(-0.1)


# ── compute_proximity — ACTIVE only ──────────────────────────────────────────


class TestComputeProximity:
    def test_active_with_depth(self):
        # Signature is now compute_proximity(bbox_depths_m, d_safe) — caller
        # is expected to have extracted per-bbox depths via
        # depth.extract_bbox_depths (or from a cached primitives record).
        r = compute_proximity([0.5], d_safe=1.5)
        assert r.status == SubScoreStatus.ACTIVE
        assert r.values.shape == (1,)
        assert 0.5 < r.values[0] < 1.0


# ── compute_x_offset — ACTIVE vs FALLBACK ────────────────────────────────────


class TestComputeXOffset:
    def _bboxes(self):
        # One centred, one far-right. image_width=200.
        return [[80, 0, 120, 100], [150, 0, 200, 100]]

    def test_fallback_without_cmd_vel(self):
        r = compute_x_offset(self._bboxes(), image_width=200, image_height=100, cmd_vel=None)
        assert r.status == SubScoreStatus.FALLBACK
        assert "cmd_vel" in r.reason
        # Centre bbox scores higher than edge bbox under centre-offset heuristic.
        assert r.values[0] > r.values[1]

    def test_fallback_when_stationary(self):
        v = RobotVelocity(linear_x=0.0, linear_y=0.0)  # speed = 0
        r = compute_x_offset(self._bboxes(), image_width=200, image_height=100, cmd_vel=v)
        assert r.status == SubScoreStatus.FALLBACK
        assert "stationary" in r.reason

    def test_active_when_moving_forward(self):
        v = RobotVelocity(linear_x=0.5, linear_y=0.0)  # pure forward motion
        r = compute_x_offset(self._bboxes(), image_width=200, image_height=100, cmd_vel=v)
        assert r.status == SubScoreStatus.ACTIVE
        # Forward motion → dangerous zone is image centre → centre bbox wins.
        assert r.values[0] > r.values[1]

    def test_active_when_moving_left_dangerous_lane_shifts(self):
        # +vy = left → robot's left appears on the LEFT of the image (negative norm_x)
        # → dangerous zone shifts LEFT, so a left-of-centre bbox should score high.
        v = RobotVelocity(linear_x=0.1, linear_y=0.5)
        left_bbox = [0, 0, 40, 100]     # far left
        right_bbox = [160, 0, 200, 100]  # far right
        r = compute_x_offset([left_bbox, right_bbox], image_width=200, image_height=100, cmd_vel=v)
        assert r.status == SubScoreStatus.ACTIVE
        assert r.values[0] > r.values[1]


# ── compute_approach — UNAVAILABLE vs ACTIVE ─────────────────────────────────


class TestComputeApproach:
    def setup_method(self):
        humandet.reset_velocity_history()

    def test_unavailable_with_only_none_track_ids(self):
        r = compute_approach([None, None])
        assert r.status == SubScoreStatus.UNAVAILABLE
        assert "no track continuity" in r.reason

    def test_unavailable_when_no_velocity_history(self):
        # Tracks present but velocity history never populated → all 0.5.
        r = compute_approach([1, 2])
        assert r.status == SubScoreStatus.UNAVAILABLE

    def test_active_once_velocity_has_spread(self):
        # Simulate two depth observations spaced apart for a single track.
        # _velocity_history uses monotonic() internally, so we poke it directly.
        import time as _time

        tid = 42
        humandet._velocity_history[tid] = humandet.deque(maxlen=humandet.VELOCITY_WINDOW)
        now = _time.monotonic()
        humandet._velocity_history[tid].append((now - 0.3, 2.0))
        humandet._velocity_history[tid].append((now, 1.0))  # approaching

        r = compute_approach([tid])
        assert r.status == SubScoreStatus.ACTIVE
        assert r.values[0] > 0.5  # approaching → score > 0.5


# ── End-to-end featextr contract ─────────────────────────────────────────────


class TestFeatextrContractEndToEnd:
    def test_extract_validates_inputs_early(self):
        """Missing depth should fail at validate(), not silently produce zeros."""
        from riskam.ml.featextr import extract

        rgb = np.zeros((10, 20, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="depth_m is required"):
            extract(FrameInputs(rgb=rgb, depth_m=None, cmd_vel=None))
