"""Tests for riskam.mocap_gt (synthetic tracks + consistency with
riskam.kinematics — no dataset needed)."""

import math

import numpy as np
import pytest

from riskam.data.thor_magni import RigidBodyTrack, ThorMagniRun
from riskam.kinematics import KinematicParams, trajectory_hazard
from riskam.mocap_gt import (
    _bridge_short_gaps,
    build_gt_table,
    cpa_vec,
    helmet_facing_sign,
    smooth_and_differentiate,
    trajectory_hazard_vec,
)

PARAMS = KinematicParams(d_safe_m=1.5, tau_s=2.0, footprint_radius_m=0.5)


# ── Consistency with the scalar kinematics implementation ────────────────────


def test_trajectory_hazard_vec_matches_scalar():
    rng = np.random.default_rng(42)
    n = 200
    x = rng.uniform(0.3, 6.0, n)
    y = rng.uniform(-3.0, 3.0, n)
    vx = rng.uniform(-2.0, 2.0, n)
    vy = rng.uniform(-2.0, 2.0, n)
    vx[::7] = np.nan  # static rows
    vy[::7] = np.nan
    _, _, t_cpa, _ = cpa_vec(x, y, vx, vy, PARAMS.v_eps_ms)
    vec = trajectory_hazard_vec(x, y, vx, vy, t_cpa, PARAMS)
    for i in range(n):
        vxi = None if np.isnan(vx[i]) else vx[i]
        vyi = None if np.isnan(vy[i]) else vy[i]
        scalar = trajectory_hazard(x[i], y[i], vxi, vyi, t_cpa[i], PARAMS)
        assert vec[i] == pytest.approx(scalar, abs=1e-12), i


def test_cpa_vec_head_on():
    # Person 4 m ahead, closing at 1 m/s straight along -x.
    closing, tan_speed, t_cpa, d_min = cpa_vec(
        np.array([4.0]), np.array([0.0]), np.array([-1.0]), np.array([0.0]), 0.05
    )
    assert closing[0] == pytest.approx(1.0)
    assert tan_speed[0] == pytest.approx(0.0)
    assert t_cpa[0] == pytest.approx(4.0)
    assert d_min[0] == pytest.approx(0.0)


def test_cpa_vec_static_and_nan_rows():
    closing, tan_speed, t_cpa, d_min = cpa_vec(
        np.array([2.0, 2.0]),
        np.array([1.0, 1.0]),
        np.array([np.nan, 0.01]),  # NaN and sub-noise speed
        np.array([np.nan, 0.01]),
        0.05,
    )
    r = math.hypot(2.0, 1.0)
    for i in range(2):
        assert closing[i] == 0.0 and t_cpa[i] == 0.0
        assert d_min[i] == pytest.approx(r)


# ── Smoothing / differentiation ──────────────────────────────────────────────


def test_smooth_and_differentiate_recovers_linear_velocity():
    t = np.arange(0, 10, 0.01)
    pos = 0.7 * t + 3.0
    smoothed, vel = smooth_and_differentiate(pos, 0.01)
    assert np.allclose(smoothed, pos, atol=1e-9)
    assert np.allclose(vel, 0.7, atol=1e-9)


def test_smooth_and_differentiate_nan_segments():
    pos = np.concatenate([np.full(100, 1.0), np.full(30, np.nan), np.arange(100) * 0.01])
    _, vel = smooth_and_differentiate(pos, 0.01)
    assert np.all(np.isnan(vel[100:130]))
    assert np.allclose(vel[:100], 0.0, atol=1e-9)
    assert np.allclose(vel[135:225], 1.0, atol=1e-6)  # away from segment edges


def test_bridge_short_gaps_interior_only():
    v = np.array([np.nan, 1.0, np.nan, np.nan, 4.0, np.nan, np.nan, np.nan, 8.0, np.nan])
    out = _bridge_short_gaps(v, max_gap=2)
    assert np.isnan(out[0]) and np.isnan(out[-1])  # ends never extrapolated
    assert out[2] == pytest.approx(2.0) and out[3] == pytest.approx(3.0)
    assert np.all(np.isnan(out[5:8]))  # 3-long gap > max_gap stays


# ── Facing sign ──────────────────────────────────────────────────────────────


def test_helmet_facing_sign_detects_mirrored_mount():
    n = 500
    vel = np.tile([1.2, 0.0], (n, 1))  # walking +x
    facing = np.tile([-1.0, 0.05], (n, 1))  # axis mounted backwards
    sign, conf = helmet_facing_sign(facing, vel)
    assert sign == -1.0
    assert conf > 0.9


def test_helmet_facing_sign_insufficient_walking():
    n = 500
    vel = np.zeros((n, 2))  # never walks
    facing = np.tile([1.0, 0.0], (n, 1))
    sign, conf = helmet_facing_sign(facing, vel)
    assert sign == 1.0 and conf == 0.0


# ── End-to-end on a synthetic run ────────────────────────────────────────────


def _synthetic_run(n=2000, dt=0.01):
    """Robot at origin facing +world-x; person walks straight toward it."""
    t = np.arange(n) * dt
    eye = np.tile(np.eye(3), (n, 1, 1))
    robot = RigidBodyTrack(
        name="DARKO_Robot",
        role="static",
        centroid_m=np.zeros((n, 3)),
        rot=eye.copy(),
    )
    # Person starts 8 m ahead, walks toward the robot at 1.25 m/s, stops
    # 1 m short, keeps facing it (helmet X = walking direction).
    x = np.maximum(8.0 - 1.25 * t, 1.0)
    centroid = np.stack([x, np.zeros(n), np.full(n, 1.7)], axis=1)
    rot = np.tile(np.eye(3), (n, 1, 1))
    rot[:, 0, 0] = -1.0  # helmet X points along world -x (walking direction)
    rot[:, 1, 1] = -1.0  # keep it a proper rotation (det +1)
    helmet = RigidBodyTrack(
        name="Helmet_1", role="Visitors-Alone", centroid_m=centroid, rot=rot
    )
    return ThorMagniRun(
        file_id="synthetic_SC0_R1",
        scenario="SC0",
        time_s=t,
        bodies={"DARKO_Robot": robot, "Helmet_1": helmet},
    )


def test_build_gt_table_head_on_approach():
    table, meta = build_gt_table(_synthetic_run(), PARAMS, out_hz=25.0)
    assert set(table.body) == {"Helmet_1"}
    approach = table[(table.t_s > 1.0) & (table.t_s < 5.0)]
    # Head-on: closing ≈ walking speed, miss distance ≈ 0.
    assert approach.closing_ms.mean() == pytest.approx(1.25, abs=0.02)
    assert approach.d_min_m.abs().max() < 0.05
    assert (approach.tan_speed_ms < 0.05).all()
    # Facing the robot the whole way (helmet X = walking direction = toward robot).
    assert approach.facing_angle_rad.max() < 0.05
    assert meta["facing_signs"]["Helmet_1"]["sign"] == 1.0
    # Standing 1 m away at the end: hazard = static shortfall
    # clip(1 − (1.0 − 0.5)/1.5) = 2/3, well above 0.5.
    end = table[table.t_s > 18.5]
    assert end.hazard_gt.min() > 0.5
    assert (np.diff(table.hazard_gt) >= -1e-6).mean() > 0.95


def test_build_gt_table_static_robot_twist_near_zero():
    table, _ = build_gt_table(_synthetic_run(), PARAMS, out_hz=25.0)
    assert table.robot_speed_ms.abs().max() < 1e-6
    assert table.robot_wz_rads.abs().max() < 1e-6
