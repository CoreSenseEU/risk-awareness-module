"""
riskam.mocap_gt

Ground-truth kinematics tables from THÖR-MAGNI motion capture — Layer 1
(measurement validity) of the evaluation plan (``docs/private/paper-plan.md``
§6). For every (frame, participant) pair the mocap yields the same kinematic
quantities RiskAM derives from vision (``riskam/kinematics.py``): planar
relative position in the robot frame, relative velocity, closing speed,
``t_cpa``, ``d_min``, and the trajectory hazard — plus a head-orientation
awareness reference the vision gaze sub-score is validated against (Layer 4a).

Frame conventions (derived empirically — see
``scripts/calibrate_thor_magni_frames.py`` for the evidence):

- QTM rotation matrices are **body→world**; the world frame is z-up.
- The DARKO robot's forward direction is its body **+X** axis (velocity/axis
  alignment +0.999 on the differential-drive SC3A runs), so REP-103 x-forward
  / y-left is exactly (body X, body Y).
- A helmet's facing direction is its body **±X** axis with a per-helmet
  mounting sign; the sign is self-calibrated per run from walking alignment
  (people face their walking direction; |mean cos| ≈ 0.9 makes the sign
  unambiguous). The calibration confidence is reported per helmet.

Relative velocity is the derivative of the **robot-frame** relative position
(transform first, then differentiate). That is deliberate: the vision stack
differentiates positions measured in the rotating camera frame, so its
velocity estimate includes the rotational transport term ω×p — the ground
truth must include it too, or the comparison would blame the metric for the
frame physics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from riskam.data.thor_magni import ROBOT_BODY, ThorMagniRun
from riskam.kinematics import KinematicParams

# Empirical frame constants (scripts/calibrate_thor_magni_frames.py).
ROBOT_FORWARD_AXIS = 0
HELMET_FACING_AXIS = 0
# Below this |mean walking-alignment cosine| the facing sign is guesswork
# (participant barely walked); the sign defaults to +1 and the confidence
# column lets analyses filter.
FACING_SIGN_MIN_CONF = 0.5

# Smoothing/differentiation of the 100 Hz mocap tracks.
SAVGOL_WINDOW_S = 0.5
SAVGOL_POLY = 2
MAX_INTERP_GAP_S = 0.3  # occlusion gaps up to this are bridged linearly
MIN_WALK_SPEED_MS = 0.3  # samples that constrain the facing sign
MIN_PLANAR_AXIS_NORM = 0.3  # facing undefined when the axis is near-vertical

OUT_HZ_DEFAULT = 25.0


# ── NaN-aware smoothing and differentiation ──────────────────────────────────


def _bridge_short_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate interior NaN runs of length ≤ ``max_gap``."""
    out = values.copy()
    isnan = np.isnan(values)
    if not isnan.any() or isnan.all():
        return out
    idx = np.arange(len(values))
    # Boundaries of NaN runs.
    starts = np.flatnonzero(isnan & ~np.roll(isnan, 1))
    ends = np.flatnonzero(isnan & ~np.roll(isnan, -1)) + 1
    if isnan[0]:
        starts = np.concatenate([[0], starts[starts != 0]])
    if isnan[-1]:
        ends = np.concatenate([ends[ends != len(values)], [len(values)]])
    interp = np.interp(idx, idx[~isnan], values[~isnan])
    for s, e in zip(starts, ends):
        if s == 0 or e == len(values):
            continue  # never extrapolate past the track's ends
        if e - s <= max_gap:
            out[s:e] = interp[s:e]
    return out


def smooth_and_differentiate(
    values: np.ndarray, dt: float, window_s: float = SAVGOL_WINDOW_S
) -> tuple[np.ndarray, np.ndarray]:
    """Savitzky–Golay smoothed signal and its first derivative.

    Applied per contiguous non-NaN segment; segments shorter than the
    window stay raw with NaN derivative (too little support for a fit).
    """
    window = max(SAVGOL_POLY + 2, int(round(window_s / dt)))
    if window % 2 == 0:
        window += 1
    smoothed = values.copy()
    deriv = np.full_like(values, np.nan)
    isnan = np.isnan(values)
    seg_starts = np.flatnonzero(~isnan & np.roll(isnan, 1))
    if not isnan[0] and len(values):
        seg_starts = np.concatenate([[0], seg_starts[seg_starts != 0]])
    seg_ends = np.flatnonzero(~isnan & np.roll(isnan, -1)) + 1
    if len(values) and not isnan[-1]:
        seg_ends = np.concatenate([seg_ends[seg_ends != len(values)], [len(values)]])
    for s, e in zip(seg_starts, seg_ends):
        if e - s < window:
            continue
        smoothed[s:e] = savgol_filter(values[s:e], window, SAVGOL_POLY)
        deriv[s:e] = savgol_filter(
            values[s:e], window, SAVGOL_POLY, deriv=1, delta=dt
        )
    return smoothed, deriv


# ── Facing-sign self-calibration ─────────────────────────────────────────────


def helmet_facing_sign(
    facing_axis_world: np.ndarray,
    vel_world: np.ndarray,
) -> tuple[float, float]:
    """Per-run mounting sign of a helmet's facing axis.

    ``facing_axis_world``: (n, 2) planar body-X axis in world coordinates;
    ``vel_world``: (n, 2) planar world velocity. Returns ``(sign, conf)``
    where ``conf = |mean cos(angle)|`` over fast-walking samples. Falls back
    to ``(+1, conf)`` when the participant never walks enough to tell.
    """
    speed = np.hypot(vel_world[:, 0], vel_world[:, 1])
    norm = np.hypot(facing_axis_world[:, 0], facing_axis_world[:, 1])
    ok = (
        (speed > MIN_WALK_SPEED_MS)
        & (norm > MIN_PLANAR_AXIS_NORM)
        & ~np.isnan(speed)
        & ~np.isnan(norm)
    )
    if ok.sum() < 100:
        return 1.0, 0.0
    cosang = (
        np.einsum("ij,ij->i", facing_axis_world[ok], vel_world[ok])
        / (norm[ok] * speed[ok])
    )
    mean = float(np.mean(cosang))
    return (1.0 if mean >= 0 else -1.0), abs(mean)


# ── Vectorized kinematics (row-wise twins of riskam.kinematics) ──────────────

_TRAJ_SAMPLES = 33
_TRAJ_HORIZON_TAUS = 5.0


def trajectory_hazard_vec(
    x: np.ndarray,
    y: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    t_cpa: np.ndarray,
    params: KinematicParams,
) -> np.ndarray:
    """Vectorized ``kinematics.trajectory_hazard`` (NaN velocity → static)."""
    static = np.isnan(vx) | (t_cpa <= 0.0)
    t_end = np.minimum(
        np.where(static, 0.0, t_cpa), _TRAJ_HORIZON_TAUS * params.tau_s
    )
    ts = np.linspace(0.0, 1.0, _TRAJ_SAMPLES)[None, :] * t_end[:, None]
    vx0 = np.where(static, 0.0, vx)[:, None]
    vy0 = np.where(static, 0.0, vy)[:, None]
    dists = np.hypot(x[:, None] + vx0 * ts, y[:, None] + vy0 * ts)
    surface_miss = np.maximum(0.0, dists - params.footprint_radius_m)
    closeness = np.clip(1.0 - surface_miss / params.d_safe_m, 0.0, 1.0)
    return np.max(closeness * np.exp(-ts / params.tau_s), axis=1)


def cpa_vec(
    x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray, v_eps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Row-wise closing speed, tangential speed, t_cpa, d_min — mirrors the
    formulas in ``KinematicTracker._update_person``. Rows with NaN or
    sub-noise velocity degrade to the static case (t_cpa = 0, d_min = r)."""
    r = np.hypot(x, y)
    speed = np.hypot(vx, vy)
    moving = ~np.isnan(speed) & (speed >= v_eps)
    vx0 = np.where(moving, vx, 0.0)
    vy0 = np.where(moving, vy, 0.0)
    p_dot_v = x * vx0 + y * vy0
    with np.errstate(divide="ignore", invalid="ignore"):
        closing = np.where(moving, np.maximum(0.0, -p_dot_v / r), 0.0)
        tan_speed = np.where(moving, np.abs(x * vy0 - y * vx0) / r, 0.0)
        v_sq = vx0 * vx0 + vy0 * vy0
        t_cpa = np.where(moving, np.maximum(0.0, -p_dot_v / np.where(v_sq > 0, v_sq, 1.0)), 0.0)
    d_min = np.hypot(x + vx0 * t_cpa, y + vy0 * t_cpa)
    return closing, tan_speed, t_cpa, d_min


# ── Table builder ────────────────────────────────────────────────────────────


def build_gt_table(
    run: ThorMagniRun,
    params: KinematicParams,
    out_hz: float = OUT_HZ_DEFAULT,
) -> tuple[pd.DataFrame, dict]:
    """One run → long-format GT table (one row per frame × tracked helmet).

    Returns ``(table, meta)``; ``meta`` records the per-helmet facing signs
    and their confidences alongside the parameters used.
    """
    robot = run.robot
    if robot is None:
        raise ValueError(f"{run.file_id}: no {ROBOT_BODY} rigid body")
    robot_valid_frac = float(np.mean(~np.isnan(robot.centroid_m[:, 0])))
    if robot_valid_frac == 0.0:
        raise ValueError(f"{run.file_id}: {ROBOT_BODY} untracked for the whole run")
    t = run.time_s
    dt = float(np.median(np.diff(t)))
    max_gap = int(round(MAX_INTERP_GAP_S / dt))
    stride = max(1, int(round(1.0 / (out_hz * dt))))

    # Robot planar pose and its world-frame twist.
    r_xy = np.stack(
        [_bridge_short_gaps(robot.centroid_m[:, k], max_gap) for k in range(2)],
        axis=1,
    )
    fwd = robot.rot[:, :2, ROBOT_FORWARD_AXIS]
    yaw_raw = np.arctan2(fwd[:, 1], fwd[:, 0])
    # Unwrap over valid samples only — NaNs poison np.unwrap's cumulative
    # correction, turning everything after the first dropout into NaN.
    yaw = np.full_like(yaw_raw, np.nan)
    yaw_valid = ~np.isnan(yaw_raw)
    yaw[yaw_valid] = np.unwrap(yaw_raw[yaw_valid])
    yaw = _bridge_short_gaps(yaw, max_gap)
    _, r_vx_w = smooth_and_differentiate(r_xy[:, 0], dt)
    _, r_vy_w = smooth_and_differentiate(r_xy[:, 1], dt)
    yaw_s, wz = smooth_and_differentiate(yaw, dt)
    cos_y, sin_y = np.cos(yaw_s), np.sin(yaw_s)
    robot_speed = np.hypot(r_vx_w, r_vy_w)
    # Ego twist in the robot frame (REP-103), for the static_odom GT rung.
    robot_vx = cos_y * r_vx_w + sin_y * r_vy_w
    robot_vy = -sin_y * r_vx_w + cos_y * r_vy_w

    frames: list[pd.DataFrame] = []
    signs: dict[str, dict] = {}
    for name, helmet in run.helmets.items():
        h_xy = np.stack(
            [_bridge_short_gaps(helmet.centroid_m[:, k], max_gap) for k in range(2)],
            axis=1,
        )
        # Relative position in the robot frame — transform, THEN differentiate
        # (matches the camera's rotating-frame measurement, ω×p included).
        dx, dy = h_xy[:, 0] - r_xy[:, 0], h_xy[:, 1] - r_xy[:, 1]
        x_rf = cos_y * dx + sin_y * dy
        y_rf = -sin_y * dx + cos_y * dy
        x_s, vx = smooth_and_differentiate(x_rf, dt)
        y_s, vy = smooth_and_differentiate(y_rf, dt)

        # Facing sign from world-frame walking alignment.
        _, h_vx_w = smooth_and_differentiate(h_xy[:, 0], dt)
        _, h_vy_w = smooth_and_differentiate(h_xy[:, 1], dt)
        facing_w = helmet.rot[:, :2, HELMET_FACING_AXIS]
        sign, conf = helmet_facing_sign(facing_w, np.stack([h_vx_w, h_vy_w], axis=1))
        signs[name] = {"sign": sign, "confidence": round(conf, 3)}

        # Angle between the facing direction and the helmet→robot direction.
        to_robot = np.stack([-dx, -dy], axis=1)
        f = sign * facing_w
        norms = np.hypot(f[:, 0], f[:, 1]) * np.hypot(to_robot[:, 0], to_robot[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            cosang = np.einsum("ij,ij->i", f, to_robot) / norms
        facing_angle = np.arccos(np.clip(cosang, -1.0, 1.0))
        facing_angle[np.hypot(f[:, 0], f[:, 1]) < MIN_PLANAR_AXIS_NORM] = np.nan

        closing, tan_speed, t_cpa, d_min = cpa_vec(x_s, y_s, vx, vy, params.v_eps_ms)
        hazard = trajectory_hazard_vec(x_s, y_s, vx, vy, t_cpa, params)

        keep = ~np.isnan(x_s) & (np.arange(len(t)) % stride == 0)
        frames.append(
            pd.DataFrame(
                {
                    "t_s": t[keep],
                    "body": name,
                    "role": helmet.role,
                    "x_m": x_s[keep],
                    "y_m": y_s[keep],
                    "d_m": np.hypot(x_s[keep], y_s[keep]),
                    "bearing_rad": np.arctan2(-y_s[keep], x_s[keep]),
                    "vx_ms": vx[keep],
                    "vy_ms": vy[keep],
                    "closing_ms": closing[keep],
                    "tan_speed_ms": tan_speed[keep],
                    "t_cpa_s": t_cpa[keep],
                    "d_min_m": d_min[keep],
                    "hazard_gt": hazard[keep],
                    "facing_angle_rad": facing_angle[keep],
                    "facing_sign_conf": conf,
                    "robot_speed_ms": robot_speed[keep],
                    "robot_vx_ms": robot_vx[keep],
                    "robot_vy_ms": robot_vy[keep],
                    "robot_wz_rads": wz[keep],
                }
            )
        )

    table = pd.concat(frames, ignore_index=True).sort_values(
        ["t_s", "body"], kind="stable", ignore_index=True
    )
    table.insert(0, "run_id", run.file_id)
    table.insert(1, "scenario", run.scenario)

    meta = {
        "run_id": run.file_id,
        "scenario": run.scenario,
        "n_rows": int(len(table)),
        "robot_valid_frac": round(robot_valid_frac, 3),
        "helmet_valid_frac": {
            n: round(float(np.mean(~np.isnan(h.centroid_m[:, 0]))), 3)
            for n, h in run.helmets.items()
        },
        "out_hz": out_hz,
        "mocap_dt_s": dt,
        "savgol_window_s": SAVGOL_WINDOW_S,
        "savgol_poly": SAVGOL_POLY,
        "max_interp_gap_s": MAX_INTERP_GAP_S,
        "facing_signs": signs,
        "params": {
            "d_safe_m": params.d_safe_m,
            "tau_s": params.tau_s,
            "beta": params.beta,
            "footprint_radius_m": params.footprint_radius_m,
            "v_eps_ms": params.v_eps_ms,
        },
    }
    return table, meta
