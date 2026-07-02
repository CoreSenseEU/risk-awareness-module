"""Synthetic-scenario tests for riskam.kinematics (no GPU, no dataset).

Scenes are constructed directly in planar robot-frame coordinates and
projected back to bbox centre-x through the same CameraModel the tracker
uses, so the tests exercise the full pixel → bearing → position → velocity
→ CPA → hazard chain.
"""

import math

import pytest

from riskam.kinematics import (
    CameraModel,
    KinematicParams,
    KinematicStatus,
    KinematicTracker,
    PlanarTwist,
    SceneRiskSmoother,
    fuse_risk,
    hazard_score,
)

IMAGE_W = 640
CAMERA = CameraModel.from_hfov(58.0, IMAGE_W)
# ema_tau_s=0: most tests assert instantaneous physics; smoothing has its
# own test class below.
PARAMS = KinematicParams(d_safe_m=2.5, tau_s=2.0, beta=1.0, ema_tau_s=0.0)


def bbox_at(x_m: float, y_m: float) -> tuple[list, float]:
    """Project a planar robot-frame position to (bbox, z_depth)."""
    u = CAMERA.cx_px - (y_m / x_m) * CAMERA.fx_px
    return [u - 20, 100, u + 20, 300], x_m


def feed(tracker, t, positions, ego=None):
    """positions: list of (track_id, x_m, y_m)."""
    bboxes, depths, tids = [], [], []
    for tid, x, y in positions:
        bbox, d = bbox_at(x, y)
        bboxes.append(bbox)
        depths.append(d)
        tids.append(tid)
    return tracker.update(t, bboxes, depths, tids, ego=ego)


class TestCameraModel:
    def test_bearing_zero_at_centre(self):
        assert CAMERA.bearing_rad(CAMERA.cx_px) == 0.0

    def test_bearing_edge_is_half_fov(self):
        assert CAMERA.bearing_rad(IMAGE_W) == pytest.approx(
            math.radians(29.0), abs=1e-9
        )
        assert CAMERA.bearing_rad(0) == pytest.approx(
            -math.radians(29.0), abs=1e-9
        )

    def test_intrinsics_and_hfov_agree_when_consistent(self):
        # A from_intrinsics model built with the pinhole fx must give the
        # same bearings as the HFOV fallback it was derived from.
        m = CameraModel.from_intrinsics(CAMERA.fx_px, CAMERA.cx_px)
        for u in (0.0, 123.4, CAMERA.cx_px, 500.0):
            assert m.bearing_rad(u) == CAMERA.bearing_rad(u)


class TestHazardAndFusion:
    def test_contact_now_is_max_hazard(self):
        assert hazard_score(0.0, 0.0, PARAMS) == 1.0

    def test_far_miss_is_zero(self):
        assert hazard_score(PARAMS.d_safe_m + 0.1, 0.0, PARAMS) == 0.0

    def test_future_cpa_decays_exponentially(self):
        now = hazard_score(0.0, 0.0, PARAMS)
        later = hazard_score(0.0, PARAMS.tau_s, PARAMS)
        assert later == pytest.approx(now * math.exp(-1.0))

    def test_footprint_shifts_the_miss_distance(self):
        fat = KinematicParams(d_safe_m=2.5, footprint_radius_m=0.5)
        # A 0.5 m predicted miss grazes the 0.5 m-radius robot surface.
        assert hazard_score(0.5, 0.0, fat) == 1.0
        assert hazard_score(0.5, 0.0, PARAMS) < 1.0

    def test_zero_hazard_kills_risk_regardless_of_awareness(self):
        # The additive pathology: awareness must never manufacture risk.
        for awareness in (0.0, 0.5, 1.0):
            assert fuse_risk(0.0, awareness, beta=5.0) == 0.0

    def test_awareness_scales_hazard(self):
        assert fuse_risk(0.4, 1.0, beta=1.0) == pytest.approx(0.4)
        assert fuse_risk(0.4, 0.0, beta=1.0) == pytest.approx(0.8)
        assert fuse_risk(0.9, 0.0, beta=1.0) == 1.0  # clipped


class TestHeadOnCloser:
    def test_tcpa_and_dmin(self):
        # Straight down the optical axis at 1 m/s from 4 m.
        tracker = KinematicTracker(PARAMS, CAMERA)
        out = None
        for i, t in enumerate([0.0, 0.2, 0.4, 0.6]):
            out = feed(tracker, t, [(7, 4.0 - t * 1.0, 0.0)])
        pk = out[0]
        assert pk.status is KinematicStatus.FULL
        assert pk.closing_ms == pytest.approx(1.0, rel=1e-6)
        assert pk.d_min_m == pytest.approx(0.0, abs=1e-9)
        assert pk.t_cpa_s == pytest.approx(3.4, rel=1e-6)  # 3.4 m left at 1 m/s
        # Trajectory-max hazard: f(t) = ((v·t + d_safe − d0)/d_safe)·e^(−t/τ)
        # peaks at t* = (d0 − d_safe)/v + τ = 2.9 s.
        expected = (1.0 * 2.0 / 2.5) * math.exp(-2.9 / 2.0)
        assert pk.hazard == pytest.approx(expected, rel=1e-2)

    def test_hazard_rises_as_it_gets_closer(self):
        # Start at 3.9 m so no sample lands exactly on d_safe = 2.5 m,
        # which would (correctly) trip the far-fallback sentinel censor.
        tracker = KinematicTracker(PARAMS, CAMERA)
        hazards = []
        for t in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            out = feed(tracker, t, [(7, 3.9 - t * 1.0, 0.0)])
            hazards.append(out[0].hazard)
        assert hazards[2:] == sorted(hazards[2:])  # monotone once FULL


class TestLateralPasser:
    def test_dmin_is_the_lateral_offset(self):
        # Crosses the path 1.5 m to the left, walking at 1 m/s: geometry
        # says the miss distance is ~1.5 m even while range decreases —
        # the F4 "in the path" emergence the additive metric can't see.
        tracker = KinematicTracker(PARAMS, CAMERA)
        out = None
        for t in [0.0, 0.2, 0.4, 0.6]:
            out = feed(tracker, t, [(3, 3.0 - t * 1.0, 1.5)])
        pk = out[0]
        assert pk.status is KinematicStatus.FULL
        assert pk.d_min_m == pytest.approx(1.5, rel=1e-6)
        # Miss ≥ d_safe·(1 − …) → low hazard despite approach.
        head_on = KinematicTracker(PARAMS, CAMERA)
        for t in [0.0, 0.2, 0.4, 0.6]:
            ref = feed(head_on, t, [(3, 3.0 - t * 1.0, 0.0)])
        assert pk.hazard < ref[0].hazard


class TestRecedingAndStatic:
    def test_receding_person_low_hazard(self):
        tracker = KinematicTracker(PARAMS, CAMERA)
        out = None
        for t in [0.0, 0.2, 0.4]:
            out = feed(tracker, t, [(4, 2.0 + t * 1.0, 0.0)])
        pk = out[0]
        # Closest approach is now: d_min = current range, t_cpa = 0.
        assert pk.t_cpa_s == 0.0
        assert pk.d_min_m == pytest.approx(pk.d_m, rel=1e-6)
        assert pk.closing_ms == 0.0  # clamped at approaching-only

    def test_static_person_degrades_to_proximity(self):
        # Sub-noise speed → hazard = clip(1 − r/d_safe): today's proximity.
        tracker = KinematicTracker(PARAMS, CAMERA)
        out = None
        for t in [0.0, 0.2, 0.4]:
            out = feed(tracker, t, [(5, 1.25, 0.0)])
        pk = out[0]
        assert pk.status is KinematicStatus.STATIC
        assert pk.hazard == pytest.approx(1.0 - 1.25 / 2.5, rel=1e-6)

    def test_single_observation_no_ego(self):
        tracker = KinematicTracker(PARAMS, CAMERA)
        pk = feed(tracker, 0.0, [(6, 2.0, 0.0)])[0]
        assert pk.status is KinematicStatus.STATIC
        assert pk.vx_ms is None
        assert pk.d_min_m == pytest.approx(2.0)


class TestStaticOdomFallback:
    def test_single_obs_with_ego_forward_motion(self):
        # Robot drives forward at 1 m/s toward a person 3 m ahead seen for
        # the first time: static-human assumption → closing at 1 m/s.
        tracker = KinematicTracker(PARAMS, CAMERA)
        ego = PlanarTwist(t_s=0.0, vx_ms=1.0, vy_ms=0.0)
        pk = feed(tracker, 0.0, [(9, 3.0, 0.0)], ego=ego)[0]
        assert pk.status is KinematicStatus.STATIC_ODOM
        assert pk.closing_ms == pytest.approx(1.0)
        assert pk.t_cpa_s == pytest.approx(3.0)
        assert pk.d_min_m == pytest.approx(0.0, abs=1e-9)

    def test_pure_rotation_gives_tangential_motion(self):
        # ω-only ego: v_rel = −ω×p; for p = (2, 0), ω = 0.5 →
        # v_rel = (0, −1): purely tangential, no closing.
        tracker = KinematicTracker(PARAMS, CAMERA)
        ego = PlanarTwist(t_s=0.0, vx_ms=0.0, vy_ms=0.0, wz_rads=0.5)
        pk = feed(tracker, 0.0, [(9, 2.0, 0.0)], ego=ego)[0]
        assert pk.status is KinematicStatus.STATIC_ODOM
        assert pk.closing_ms == pytest.approx(0.0, abs=1e-9)
        assert pk.tan_speed_ms == pytest.approx(1.0, rel=1e-6)


class TestSentinelsAndGaps:
    def test_dead_zone_sentinel_is_max_hazard(self):
        tracker = KinematicTracker(PARAMS, CAMERA)
        bbox, _ = bbox_at(1.0, 0.0)
        pk = tracker.update(0.0, [bbox], [0.0], [11])[0]
        assert pk.status is KinematicStatus.DEAD_ZONE
        assert pk.hazard == 1.0
        assert pk.d_min_m == 0.0

    def test_dead_zone_does_not_pollute_history(self):
        # d = 0.0 between two real observations must not produce a huge
        # spurious velocity.
        tracker = KinematicTracker(PARAMS, CAMERA)
        feed(tracker, 0.0, [(11, 2.0, 0.0)])
        bbox, _ = bbox_at(2.0, 0.0)
        tracker.update(0.1, [bbox], [0.0], [11])  # sentinel frame
        pk = feed(tracker, 0.2, [(11, 1.9, 0.0)])[0]
        assert pk.status is KinematicStatus.FULL
        assert pk.closing_ms == pytest.approx(0.5, rel=1e-6)  # 0.1 m / 0.2 s

    def test_far_fallback_sentinel_excluded_from_history(self):
        tracker = KinematicTracker(PARAMS, CAMERA)
        feed(tracker, 0.0, [(12, PARAMS.d_safe_m, 0.0)])
        pk = feed(tracker, 0.2, [(12, PARAMS.d_safe_m, 0.0)])[0]
        # Two sentinel frames → still no velocity history.
        assert pk.status is KinematicStatus.STATIC
        assert pk.hazard == 0.0

    def test_gap_resets_track_history(self):
        tracker = KinematicTracker(PARAMS, CAMERA)
        feed(tracker, 0.0, [(13, 4.0, 0.0)])
        feed(tracker, 0.2, [(13, 3.8, 0.0)])
        # 2 s gap: same ID may be a re-used track — history must restart.
        pk = feed(tracker, 2.2, [(13, 1.0, 0.0)])[0]
        assert pk.status is KinematicStatus.STATIC
        assert pk.n_obs == 1

    def test_untracked_detection_uses_single_obs_path(self):
        tracker = KinematicTracker(PARAMS, CAMERA)
        pk = feed(tracker, 0.0, [(None, 2.0, 0.0)])[0]
        assert pk.status is KinematicStatus.STATIC
        pk = feed(tracker, 0.2, [(None, 1.9, 0.0)])[0]
        assert pk.status is KinematicStatus.STATIC  # no history for None

    def test_no_detections_returns_empty(self):
        tracker = KinematicTracker(PARAMS, CAMERA)
        assert tracker.update(0.0, [], [], []) == []

    def test_reset_clears_state(self):
        tracker = KinematicTracker(PARAMS, CAMERA)
        feed(tracker, 0.0, [(14, 3.0, 0.0)])
        tracker.reset()
        pk = feed(tracker, 0.2, [(14, 2.9, 0.0)])[0]
        assert pk.n_obs == 1


class TestTrajectoryContinuity:
    """The v_eps boundary must not be a hazard cliff (jumpy-video fix)."""

    def test_slow_approacher_matches_static(self):
        # 6 cm/s toward the robot from 2 m: barely above the noise floor.
        # The old closest-approach-moment formula scored this ~0 (t_cpa
        # ≈ 33 s → exp crushes it) while a static person at 2 m scored
        # 0.2 — a discontinuity that flickered with velocity noise.
        slow = KinematicTracker(PARAMS, CAMERA)
        out = None
        for t in [0.0, 0.4, 0.8]:
            out = feed(slow, t, [(1, 2.0 - 0.06 * t, 0.0)])
        assert out[0].status is KinematicStatus.FULL
        static_hazard = 1.0 - 2.0 / 2.5
        assert out[0].hazard == pytest.approx(static_hazard, abs=0.02)

    def test_approacher_never_below_static_at_same_range(self):
        for v in [0.06, 0.2, 0.5, 1.0, 2.0]:
            tracker = KinematicTracker(PARAMS, CAMERA)
            out = None
            for t in [0.0, 0.2, 0.4]:
                out = feed(tracker, t, [(1, 2.0 - v * t, 0.0)])
            r = out[0].d_m
            static_hazard = max(0.0, 1.0 - r / 2.5)
            assert out[0].hazard >= static_hazard - 1e-9


class TestChannelSmoothing:
    SMOOTH = KinematicParams(d_safe_m=2.5, tau_s=2.0, beta=1.0, ema_tau_s=0.5)

    def test_first_observation_is_unsmoothed(self):
        tracker = KinematicTracker(self.SMOOTH, CAMERA)
        pk = feed(tracker, 0.0, [(1, 1.25, 0.0)])[0]
        assert pk.hazard == pytest.approx(1.0 - 1.25 / 2.5, rel=1e-6)

    def test_ema_damps_a_depth_spike(self):
        tracker = KinematicTracker(self.SMOOTH, CAMERA)
        feed(tracker, 0.0, [(1, 2.0, 0.0)])
        # Depth noise teleports the person to 1.0 m for one frame (dt =
        # 0.1 → alpha = 1 − e^(−0.2) ≈ 0.18): the smoothed hazard moves
        # only a fraction of the raw jump.
        pk = feed(tracker, 0.1, [(1, 1.0, 0.0)])[0]
        h_before = 1.0 - 2.0 / 2.5
        assert pk.hazard < h_before + 0.35 * (1.0 - h_before)

    def test_awareness_channel_smoothed_and_fused(self):
        tracker = KinematicTracker(self.SMOOTH, CAMERA)
        bbox, d = bbox_at(1.25, 0.0)
        k1 = tracker.update(0.0, [bbox], [d], [1], awareness=[1.0])[0]
        assert k1.awareness == pytest.approx(1.0)
        assert k1.risk == pytest.approx(k1.hazard)  # aware → no penalty
        # Gaze flickers to 0 for one frame; smoothed awareness stays high.
        k2 = tracker.update(0.1, [bbox], [d], [1], awareness=[0.0])[0]
        assert k2.awareness > 0.7
        assert k2.risk == pytest.approx(
            k2.hazard * (1.0 + self.SMOOTH.beta * (1.0 - k2.awareness))
        )

    def test_no_awareness_means_no_risk_output(self):
        tracker = KinematicTracker(self.SMOOTH, CAMERA)
        pk = feed(tracker, 0.0, [(1, 1.5, 0.0)])[0]
        assert pk.awareness is None
        assert pk.risk is None

    def test_gap_restarts_the_ema(self):
        tracker = KinematicTracker(self.SMOOTH, CAMERA)
        feed(tracker, 0.0, [(1, 2.4, 0.0)])  # low hazard
        # Same ID reappears 5 s later, very close: no stale averaging.
        pk = feed(tracker, 5.0, [(1, 0.5, 0.0)])[0]
        assert pk.hazard == pytest.approx(1.0 - 0.5 / 2.5, rel=1e-6)


class TestSceneRiskSmoother:
    def test_rise_is_instant(self):
        s = SceneRiskSmoother(release_tau_s=1.0)
        assert s.update(0.0, 0.1) == 0.1
        assert s.update(0.1, 0.9) == 0.9  # alarm never delayed

    def test_release_decays_exponentially(self):
        # Bridges a detector dropout: person vanishes for one frame.
        s = SceneRiskSmoother(release_tau_s=1.0)
        s.update(0.0, 0.8)
        assert s.update(0.5, 0.0) == pytest.approx(0.8 * math.exp(-0.5))
        assert s.update(1.0, 0.0) == pytest.approx(0.8 * math.exp(-1.0))

    def test_release_is_evidence_bounded(self):
        # A person who genuinely LEFT must not leave a long ghost: the
        # release extrapolates at most hold_max_s past the last
        # detection-backed value, then the empty scene reports zero.
        s = SceneRiskSmoother(release_tau_s=1.0, hold_max_s=1.0)
        s.update(0.0, 0.8)
        assert s.update(0.9, 0.0) > 0.0   # inside the grace window
        assert s.update(1.5, 0.0) == 0.0  # beyond it: no memory
        # And the support point has moved on — no resurrection later.
        assert s.update(1.6, 0.0) == 0.0

    def test_held_flag_reports_memory_vs_measurement(self):
        s = SceneRiskSmoother(release_tau_s=1.0, hold_max_s=1.0)
        s.update(0.0, 0.8)
        assert not s.held            # measurement
        s.update(0.1, 0.0)
        assert s.held                # release bridging a dropout
        s.update(0.2, 0.9)
        assert not s.held            # fresh measurement dominates again

    def test_new_value_above_release_wins(self):
        s = SceneRiskSmoother(release_tau_s=1.0)
        s.update(0.0, 0.8)
        assert s.update(0.1, 0.85) == 0.85

    def test_zero_tau_is_passthrough(self):
        s = SceneRiskSmoother(release_tau_s=0.0)
        s.update(0.0, 0.8)
        assert s.update(0.1, 0.0) == 0.0

    def test_reset(self):
        s = SceneRiskSmoother(release_tau_s=1.0)
        s.update(0.0, 0.8)
        s.reset()
        assert s.update(10.0, 0.0) == 0.0
