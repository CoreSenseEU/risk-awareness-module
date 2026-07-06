"""
riskam.hindsight

The Layer-2 hindsight oracle: per-frame ground truth of the form "a person
ACTUALLY came within r metres within the next T seconds", computed
non-causally from the full recorded trajectory.

Circularity firewall (paper-plan hard constraint): this module reads only
cached detection primitives, camera geometry, and the FUTURE of the
recording. It must never import the causal scoring stack —
``riskam.kinematics.KinematicTracker``, ``riskam.score``, ``riskam.ml`` or
``riskam.ssm`` — and no tracker state may flow in. The oracle reads the
future; every score channel reads only the past; they meet only in the
event table. ``tests/test_hindsight.py`` pins this invariant.

Distances are planar ranges in the robot frame, reconstructed from the
cached bbox centre + median bbox depth via the same pinhole formula the
causal tracker uses (``x = d``, ``y = -d·(u-cx)/fx``) — sharing the
*measurement* is fine (and unavoidable: this is a vision-only recording);
the firewall is about the *inference* direction.

Sentinels (same float-exact conventions as ``riskam.kinematics``):
- ``bbox_depth == 0.0`` — structured-light dead zone: the person is closer
  than the sensor's near clip (~0.6 m). Treated as distance 0.0 (an event
  at every grid radius) and flagged, since intimate-space intrusion is
  precisely what the oracle must not miss.
- ``bbox_depth == d_safe`` — far/invalid fallback: censored (NaN). A person
  known only to be "far or unmeasurable" cannot create a ≤1.5 m event.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskam.kinematics import CameraModel
from riskam.mocap_gt import (
    MAX_INTERP_GAP_S,
    SAVGOL_WINDOW_S,
    _bridge_short_gaps,
    smooth_and_differentiate,
)

# The oracle grid (paper plan §Layer 2).
R_GRID_M = (0.5, 1.0, 1.5)
T_GRID_S = (1.0, 2.0, 3.0)

# Column-name helpers: r=0.5 → "r05", T=1.0 → "T1".
def r_tag(r: float) -> str:
    return f"r{int(round(r * 10)):02d}"


def t_tag(t: float) -> str:
    return f"T{int(round(t))}s"


@dataclass(frozen=True)
class OracleParams:
    """Parameters of the hindsight oracle."""

    r_grid_m: tuple = R_GRID_M
    t_grid_s: tuple = T_GRID_S
    # Same physical referents as the mocap GT toolchain.
    savgol_window_s: float = SAVGOL_WINDOW_S
    max_interp_gap_s: float = MAX_INTERP_GAP_S
    # Episode hysteresis: an event releases only once the scene distance
    # rises back above r + hysteresis (debounces boundary chatter).
    hysteresis_m: float = 0.2


@dataclass
class FrameObservations:
    """Per-person planar observations decoded from one cached frame."""

    t_s: float
    track_ids: list
    x_m: np.ndarray  # forward, NaN = censored (far/invalid sentinel)
    y_m: np.ndarray  # left-positive
    deadzone: np.ndarray  # bool; True → person inside the near-clip dead zone

    @property
    def d_planar_m(self) -> np.ndarray:
        d = np.hypot(self.x_m, self.y_m)
        d[self.deadzone] = 0.0
        return d


def decode_frame(
    prim, camera: CameraModel, d_safe_m: float, t_s: float = 0.0
) -> FrameObservations:
    """Decode one ``CachedFrameFeatures`` into planar observations.

    ``prim`` needs ``human_bboxes``, ``track_ids`` and ``bbox_depths_m``;
    ``t_s`` is the frame timestamp (the RGB stem).
    """
    n = len(prim.human_bboxes)
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    dead = np.zeros(n, dtype=bool)
    for i in range(n):
        d = float(prim.bbox_depths_m[i])
        if d == 0.0:
            dead[i] = True
            x[i] = 0.0
            y[i] = 0.0
        elif d == d_safe_m:
            continue  # censored: stays NaN
        else:
            x1, _, x2, _ = prim.human_bboxes[i]
            u = 0.5 * (float(x1) + float(x2))
            x[i] = d
            y[i] = -d * (u - camera.cx_px) / camera.fx_px
    return FrameObservations(
        t_s=t_s, track_ids=list(prim.track_ids), x_m=x, y_m=y, deadzone=dead
    )


def smoothed_scene_min_d(
    frames: list[FrameObservations], params: OracleParams = OracleParams()
) -> np.ndarray:
    """Per-frame scene minimum distance (metres) with per-track smoothing.

    The event definition is scene-level (identity-free): ByteTrack ID
    switches fragment tracks but cannot hide a scene-level minimum, whereas
    per-track events would drop exactly the fast, close encounters where
    tracking is worst. Identity is used only to give the Savitzky–Golay
    smoother per-track support:

    - per track: (x, y) series on the run's frame grid, short detection
      gaps bridged linearly, savgol-smoothed per contiguous segment
      (segments shorter than the window stay raw);
    - untracked detections (``track_id is None``) contribute raw distance;
    - dead-zone detections contribute exactly 0.0 (never smoothed away);
    - frames with no usable observation are ``+inf``.
    """
    n = len(frames)
    min_d = np.full(n, np.inf)
    if n == 0:
        return min_d

    t = np.array([f.t_s for f in frames])
    dt = float(np.median(np.diff(t))) if n > 1 else 1.0
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0
    max_gap = max(1, int(round(params.max_interp_gap_s / dt)))

    # Per-track series on the frame grid.
    track_x: dict[int, np.ndarray] = {}
    track_y: dict[int, np.ndarray] = {}
    for i, f in enumerate(frames):
        for j, tid in enumerate(f.track_ids):
            if np.isnan(f.x_m[j]):
                continue  # censored
            if f.deadzone[j]:
                min_d[i] = 0.0  # raw, unconditional
                continue
            if tid is None:
                min_d[i] = min(min_d[i], float(np.hypot(f.x_m[j], f.y_m[j])))
                continue
            if tid not in track_x:
                track_x[tid] = np.full(n, np.nan)
                track_y[tid] = np.full(n, np.nan)
            track_x[tid][i] = f.x_m[j]
            track_y[tid][i] = f.y_m[j]

    for tid, xs in track_x.items():
        ys = track_y[tid]
        xs = _bridge_short_gaps(xs, max_gap)
        ys = _bridge_short_gaps(ys, max_gap)
        xs_s, _ = smooth_and_differentiate(xs, dt, params.savgol_window_s)
        ys_s, _ = smooth_and_differentiate(ys, dt, params.savgol_window_s)
        d = np.hypot(xs_s, ys_s)
        valid = ~np.isnan(d)
        min_d[valid] = np.minimum(min_d[valid], d[valid])

    return min_d


def onset_times(
    t: np.ndarray, min_d: np.ndarray, r: float, hysteresis_m: float
) -> np.ndarray:
    """Event-onset timestamps for radius ``r`` with release hysteresis.

    An onset is the crossing of the scene distance to ≤ r from the released
    state; the event releases only once the distance exceeds r + hysteresis.
    A run that STARTS inside r yields an onset at ``t[0]`` (evaluation code
    decides whether such truncated events count toward recall).

    This is the single source of truth for "event" — the event table's
    ``t_to_onset`` columns and the early-warning episode logic both use it.
    """
    onsets = []
    inside = False
    for i in range(len(t)):
        d = min_d[i]
        if not inside and d <= r:
            onsets.append(t[i])
            inside = True
        elif inside and d > r + hysteresis_m:
            inside = False
    return np.array(onsets)


def _future_window_min(t: np.ndarray, values: np.ndarray, horizon_s: float) -> np.ndarray:
    """Min of ``values`` over the window ``(t_i, t_i + horizon]`` per i.

    O(n) two-pointer sweep with a monotonic deque; handles irregular
    timestamps. Empty windows (run end) yield ``+inf``.
    """
    from collections import deque

    n = len(t)
    out = np.full(n, np.inf)
    dq: deque[int] = deque()  # indices, values increasing
    right = 0
    for i in range(n):
        # Grow the window to include all j with t_j <= t_i + horizon.
        while right < n and t[right] <= t[i] + horizon_s:
            while dq and values[dq[-1]] >= values[right]:
                dq.pop()
            dq.append(right)
            right += 1
        # Shrink from the left: window is exclusive of the current frame.
        while dq and t[dq[0]] <= t[i]:
            dq.popleft()
        if dq:
            out[i] = values[dq[0]]
    return out


def _future_window_count(t: np.ndarray, horizon_s: float) -> np.ndarray:
    """Number of frames in ``(t_i, t_i + horizon]`` per i (two-pointer)."""
    n = len(t)
    counts = np.zeros(n, dtype=int)
    right = 0
    for i in range(n):
        if right < i + 1:
            right = i + 1
        while right < n and t[right] <= t[i] + horizon_s:
            right += 1
        counts[i] = right - (i + 1)
    return counts


def hindsight_columns(
    t: np.ndarray, min_d: np.ndarray, params: OracleParams = OracleParams()
) -> dict[str, np.ndarray]:
    """All oracle columns for one run.

    Returns, for each T in the grid: ``future_min_d_<T>`` and
    ``oracle_cov_<T>`` (fraction of the expected lookahead frames actually
    observed — < 1 near run end or across frame drops; evaluation masks
    negatives with low coverage). For each (r, T): ``event_<r>_<T>``
    (uint8). For each r: ``in_event_<r>`` and ``t_to_onset_<r>`` (seconds
    to the next onset, NaN when none follows).
    """
    n = len(t)
    dt = float(np.median(np.diff(t))) if n > 1 else 1.0
    cols: dict[str, np.ndarray] = {}

    for T in params.t_grid_s:
        fmd = _future_window_min(t, min_d, T)
        counts = _future_window_count(t, T)
        expected = max(1.0, T / dt)
        cols[f"future_min_d_{t_tag(T)}"] = fmd
        cols[f"oracle_cov_{t_tag(T)}"] = np.clip(counts / expected, 0.0, 1.0)

    for r in params.r_grid_m:
        cols[f"in_event_{r_tag(r)}"] = (min_d <= r).astype(np.uint8)
        onsets = onset_times(t, min_d, r, params.hysteresis_m)
        # Seconds to the next onset strictly after t_i.
        tto = np.full(n, np.nan)
        if len(onsets):
            idx = np.searchsorted(onsets, t, side="right")
            has_next = idx < len(onsets)
            tto[has_next] = onsets[idx[has_next]] - t[has_next]
        cols[f"t_to_onset_{r_tag(r)}"] = tto
        for T in params.t_grid_s:
            fmd = cols[f"future_min_d_{t_tag(T)}"]
            cols[f"event_{r_tag(r)}_{t_tag(T)}"] = (fmd <= r).astype(np.uint8)

    return cols
