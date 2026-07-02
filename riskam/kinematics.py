"""
riskam.kinematics

Direction A of the metric redesign (``docs/private/paper-plan.md``): a
physically-grounded kinematic hazard with units, modulated by awareness.

Per person and frame the tracker derives, from cached primitives only
(bboxes + per-bbox depths + ByteTrack IDs):

- planar position ``p = (x, y)`` in the robot frame (REP-103: x forward,
  y left) from bbox centre-x, depth, and camera intrinsics;
- relative velocity ``v_rel`` as the time derivative of ``p`` over a short
  per-track window. Because the camera measures *relative* position, this
  derivative already folds in ego-motion — no odometry subtraction is
  needed on the main path (this corrects the paper-plan §2A formulation,
  which composed radial ḋ + bearing rate + an explicit v_human − v_robot);
- closest-point-of-approach extrapolation: ``t_cpa`` (s) and predicted
  miss distance ``d_min`` (m);
- ``hazard ∈ [0, 1]`` from (d_min, t_cpa) and the platform's safety
  distance / footprint / reaction window;
- fused risk ``hazard · (1 + β·(1 − awareness))`` — awareness can only
  scale an existing hazard, never create one.

Odometry (when available) serves three secondary roles: the single-
observation fallback (static-human assumption), a robot-speed severity
feature for the calibrated layer, and a consistency check. Timestamps are
always explicit (frame stems offline, message stamps online) — never
wall-clock, unlike the deployed ``humandet`` velocity history.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from riskam.platforms import BETA_UNAWARE_DEFAULT, TAU_REACTION_S_DEFAULT


# ── Camera geometry ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CameraModel:
    """Minimal horizontal pinhole model: bbox centre-x → bearing.

    ``fx_px`` and ``cx_px`` come from a per-run ``camera_info`` snapshot
    when available (exact), else from the sensor's datasheet HFOV
    (pinhole fallback).
    """

    fx_px: float
    cx_px: float

    @classmethod
    def from_intrinsics(cls, fx: float, cx: float) -> "CameraModel":
        return cls(fx_px=float(fx), cx_px=float(cx))

    @classmethod
    def from_hfov(cls, hfov_deg: float, image_width: int) -> "CameraModel":
        half_w = image_width / 2.0
        fx = half_w / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(fx_px=fx, cx_px=half_w)

    def bearing_rad(self, u_px: float) -> float:
        """Bearing of pixel column ``u_px``; positive = image right."""
        return math.atan2(u_px - self.cx_px, self.fx_px)


# ── Parameters and value types ───────────────────────────────────────────────


@dataclass(frozen=True)
class KinematicParams:
    """Kinematic-hazard knobs; every one has a physical referent.

    ``d_safe_m``: stopping distance + margin (same referent as the
    proximity sub-score). ``tau_s``: reaction window. ``beta``:
    oblivious-human penalty. ``footprint_radius_m``: miss distance is
    measured from the robot surface. ``window``/``max_gap_s``/
    ``min_span_s`` bound the velocity fit; ``v_eps_ms`` is the speed
    below which motion is treated as noise (static case).
    ``ema_tau_s``: time constant of the per-track exponential moving
    average applied to the hazard and awareness channels (the kinematic
    counterpart of the deployed scorer's 5-frame mean); ``0`` disables
    smoothing. ``release_tau_s``: decay constant of the scene-level
    risk release (:class:`SceneRiskSmoother`) — rise is always instant.
    ``hold_max_s``: how far past the last detection-backed value the
    release may extrapolate (same referent as ``max_gap_s``: the track-
    bridging horizon); beyond it an empty scene reports zero.
    """

    d_safe_m: float
    tau_s: float = TAU_REACTION_S_DEFAULT
    beta: float = BETA_UNAWARE_DEFAULT
    footprint_radius_m: float = 0.0
    window: int = 8  # mirrors humandet.VELOCITY_WINDOW
    max_gap_s: float = 1.0
    min_span_s: float = 0.05  # mirrors humandet's dt guard
    v_eps_ms: float = 0.05
    ema_tau_s: float = 0.5
    release_tau_s: float = 0.5
    hold_max_s: float = 1.0


@dataclass(frozen=True)
class PlanarTwist:
    """One ego-motion sample (odometry twist) in the robot base frame."""

    t_s: float
    vx_ms: float
    vy_ms: float
    wz_rads: float = 0.0

    @property
    def speed(self) -> float:
        return math.hypot(self.vx_ms, self.vy_ms)


class KinematicStatus(str, Enum):
    FULL = "full"                # v_rel fit from ≥2 tracked observations
    STATIC_ODOM = "static_odom"  # 1 obs; v_rel = ego-motion (static human)
    STATIC = "static"            # 1 obs, no ego info → d_min = d, t_cpa = 0
    DEAD_ZONE = "dead_zone"      # depth sentinel 0.0 → person too close


@dataclass
class PersonKinematics:
    """Kinematic state and hazard channels for one person in one frame.

    ``hazard`` and ``awareness`` are the EMA-smoothed channels (raw when
    ``ema_tau_s == 0`` or the track is new); ``risk`` is their fusion.
    ``awareness``/``risk`` are ``None`` when the caller did not supply an
    awareness value.
    """

    track_id: int | None
    d_m: float
    bearing_rad: float
    x_m: float
    y_m: float
    vx_ms: float | None            # relative velocity (human w.r.t. robot)
    vy_ms: float | None
    closing_ms: float              # −ṙ; ≥0 = approaching, 0 when unknown
    tan_speed_ms: float            # tangential relative speed; 0 when unknown
    t_cpa_s: float
    d_min_m: float
    hazard: float                  # ∈ [0, 1], separately inspectable channel
    status: KinematicStatus
    n_obs: int
    awareness: float | None = None
    risk: float | None = None


# ── Hazard and fusion (the §2 formulas) ──────────────────────────────────────


def hazard_score(d_min_m: float, t_cpa_s: float, params: KinematicParams) -> float:
    """Pointwise hazard ∈ [0, 1]: distance shortfall at a moment ``t_cpa_s``
    in the future, decayed by how far away that moment is (beyond τ ≈
    reactable)."""
    surface_miss = max(0.0, d_min_m - params.footprint_radius_m)
    closeness = float(np.clip(1.0 - surface_miss / params.d_safe_m, 0.0, 1.0))
    return closeness * math.exp(-max(0.0, t_cpa_s) / params.tau_s)


# Sampling grid for the trajectory hazard. The integrand varies on the τ
# scale, so ~33 samples over ≤ 5τ keep the max within a fraction of a
# percent of the analytic optimum.
_TRAJ_SAMPLES = 33
_TRAJ_HORIZON_TAUS = 5.0


def trajectory_hazard(
    x_m: float, y_m: float,
    vx_ms: float | None, vy_ms: float | None,
    t_cpa_s: float, params: KinematicParams,
) -> float:
    """Hazard of the whole predicted pass: max over t ∈ [0, t_cpa] of the
    pointwise hazard at the extrapolated position.

    Scoring only the closest-approach *moment* is discontinuous at v → 0:
    a person standing at 2 m scores their full proximity shortfall, while
    the same person drifting in at 6 cm/s has a huge t_cpa and scores ~0.
    Taking the max over the predicted trajectory restores continuity —
    with no velocity (or a diverging pass) the max sits at t = 0 and
    reduces exactly to the static proximity shortfall.
    """
    if vx_ms is None or t_cpa_s <= 0.0:
        return hazard_score(math.hypot(x_m, y_m), 0.0, params)
    t_end = min(t_cpa_s, _TRAJ_HORIZON_TAUS * params.tau_s)
    ts = np.linspace(0.0, t_end, _TRAJ_SAMPLES)
    dists = np.hypot(x_m + vx_ms * ts, y_m + vy_ms * ts)
    surface_miss = np.maximum(0.0, dists - params.footprint_radius_m)
    closeness = np.clip(1.0 - surface_miss / params.d_safe_m, 0.0, 1.0)
    return float(np.max(closeness * np.exp(-ts / params.tau_s)))


def fuse_risk(hazard: float, awareness: float, beta: float = BETA_UNAWARE_DEFAULT) -> float:
    """Awareness-modulated risk: scales hazard, never creates it.

    ``hazard = 0 ⇒ risk = 0`` regardless of awareness — the additive
    pathology ("unaware person far away accrues risk") dies here.
    """
    return float(np.clip(hazard * (1.0 + beta * (1.0 - awareness)), 0.0, 1.0))


class SceneRiskSmoother:
    """Asymmetric scene-risk smoother: instant attack, evidence-bounded release.

    Rising risk passes through untouched — an alarm is never delayed.
    A drop decays on the release constant, which bridges short detector
    dropouts (a person YOLO misses for one 33 ms frame is still there).

    The release is **bounded by evidence**: it extrapolates only up to
    ``hold_max_s`` past the last measurement that dominated the output.
    Beyond the grace window the instant value passes through — an empty
    scene reports zero, not a decaying memory of someone who left.
    Without the bound, a departure showed a ghost risk gliding down for
    however long the scene stayed empty. ``release_tau_s = 0`` disables
    the release entirely (pure passthrough).

    ``held`` is True when the last output came from the release rather
    than the instant measurement — callers can (and the video overlay
    does) surface that the value is memory, not measurement.
    """

    def __init__(self, release_tau_s: float, hold_max_s: float = 1.0) -> None:
        self._tau = release_tau_s
        self._hold_max_s = hold_max_s
        # Last evidence-backed point: time and value of the most recent
        # frame whose instant measurement dominated the output.
        self._t_support: float | None = None
        self._v_support = 0.0
        self.held = False

    def update(self, t_s: float, value: float) -> float:
        out = value
        if self._tau > 0.0 and self._t_support is not None:
            dt = max(0.0, t_s - self._t_support)
            if dt <= self._hold_max_s:
                out = max(value, self._v_support * math.exp(-dt / self._tau))
        self.held = out > value
        if not self.held:
            self._t_support = t_s
            self._v_support = value
        return out

    def reset(self) -> None:
        self._t_support = None
        self._v_support = 0.0
        self.held = False


# ── Tracker ──────────────────────────────────────────────────────────────────


class KinematicTracker:
    """Per-track planar kinematics with explicit timestamps.

    Instantiate one per run/session (like ``RiskScorer``); state is a
    per-ByteTrack-ID deque of ``(t, x, y)`` observations. Depth sentinels
    (0.0 dead-zone close-fallback and exactly ``d_safe`` far-fallback,
    see ``riskam/ml/depth.py``) are censored values, not measurements —
    they are never appended to the history.
    """

    def __init__(self, params: KinematicParams, camera: CameraModel) -> None:
        self._params = params
        self._camera = camera
        # track_id → deque[(t_s, x_m, y_m)]
        self._history: dict[int, deque] = {}
        # track_id → (t_last, hazard_ema, awareness_ema | None)
        self._ema: dict[int, tuple] = {}

    def reset(self) -> None:
        self._history.clear()
        self._ema.clear()

    # ── Public API ────────────────────────────────────────────────────────

    def update(
        self,
        t_s: float,
        bboxes: list,
        bbox_depths_m: list,
        track_ids: list,
        ego: PlanarTwist | None = None,
        awareness: list | None = None,
    ) -> list[PersonKinematics]:
        """Ingest one frame's detections; return per-person kinematics.

        ``ego`` is the nearest odometry twist for this frame, if any; it
        is used only for the single-observation fallback (static-human
        assumption) — the ≥2-observation path measures relative velocity
        directly and needs no ego signal. ``awareness`` (per-person, in
        [0, 1], typically the gaze sub-score) enables the smoothed
        awareness channel and the fused ``risk`` output.
        """
        n = len(bboxes)
        if track_ids is None or len(track_ids) != n:
            track_ids = [None] * n

        results: list[PersonKinematics] = []
        for i in range(n):
            aw = float(awareness[i]) if awareness is not None else None
            results.append(
                self._update_person(
                    t_s, bboxes[i], float(bbox_depths_m[i]), track_ids[i],
                    ego, aw,
                )
            )

        self._prune_stale(t_s)
        return results

    # ── Internals ─────────────────────────────────────────────────────────

    def _update_person(
        self,
        t_s: float,
        bbox: list,
        d_m: float,
        track_id: int | None,
        ego: PlanarTwist | None,
        awareness: float | None,
    ) -> PersonKinematics:
        p = self._params
        x1, _, x2, _ = bbox
        u = (float(x1) + float(x2)) / 2.0
        bearing = self._camera.bearing_rad(u)

        # Dead-zone sentinel: person too close to measure. Maximum hazard,
        # and the censored depth must not pollute the velocity fit.
        if d_m == 0.0:
            hazard, aw_s, risk = self._smooth_and_fuse(
                t_s, track_id, 1.0, awareness
            )
            return PersonKinematics(
                track_id=track_id, d_m=0.0, bearing_rad=bearing,
                x_m=0.0, y_m=0.0, vx_ms=None, vy_ms=None,
                closing_ms=0.0, tan_speed_ms=0.0,
                t_cpa_s=0.0, d_min_m=0.0, hazard=hazard,
                status=KinematicStatus.DEAD_ZONE,
                n_obs=self._n_obs(track_id),
                awareness=aw_s, risk=risk,
            )

        # Planar position in the robot frame. d is z-depth along the
        # optical axis; image-right (u > cx) is negative y (REP-103 left+).
        x = d_m
        y = -d_m * (u - self._camera.cx_px) / self._camera.fx_px
        r = math.hypot(x, y)

        # Far-fallback sentinel (no valid pixels → exactly d_safe): treat as
        # censored — usable as a static observation, excluded from history.
        # Float-exact match on a real measurement is measure-zero.
        censored_far = d_m == p.d_safe_m

        hist = None
        if track_id is not None and not censored_far:
            hist = self._history.setdefault(track_id, deque(maxlen=p.window))
            if hist and t_s - hist[-1][0] > p.max_gap_s:
                hist.clear()  # occlusion / track re-use guard
            hist.append((t_s, x, y))

        if hist is not None and len(hist) >= 2 and (hist[-1][0] - hist[0][0]) >= p.min_span_s:
            t_arr = np.array([o[0] for o in hist])
            vx = float(np.polyfit(t_arr - t_arr[0], [o[1] for o in hist], 1)[0])
            vy = float(np.polyfit(t_arr - t_arr[0], [o[2] for o in hist], 1)[0])
            status = KinematicStatus.FULL
        elif ego is not None:
            # Static-human assumption: relative velocity is pure ego-motion,
            # v_rel = −(v_ego + ω × p) with ω × p = wz · (−y, x).
            vx = -(ego.vx_ms + ego.wz_rads * -y)
            vy = -(ego.vy_ms + ego.wz_rads * x)
            status = KinematicStatus.STATIC_ODOM
        else:
            vx = vy = None
            status = KinematicStatus.STATIC

        if vx is not None and math.hypot(vx, vy) >= p.v_eps_ms:
            closing = -(x * vx + y * vy) / r
            tan_speed = abs(x * vy - y * vx) / r
            v_sq = vx * vx + vy * vy
            t_cpa = max(0.0, -(x * vx + y * vy) / v_sq)
            d_min = math.hypot(x + vx * t_cpa, y + vy * t_cpa)
        else:
            # No velocity information (or sub-noise speed): closest approach
            # is now — degrades to exactly today's proximity behaviour.
            closing = tan_speed = 0.0
            t_cpa = 0.0
            d_min = r
            vx = vy = None
            if status is KinematicStatus.FULL:
                status = KinematicStatus.STATIC

        hazard_raw = trajectory_hazard(x, y, vx, vy, t_cpa, p)
        hazard, aw_s, risk = self._smooth_and_fuse(
            t_s, track_id, hazard_raw, awareness
        )
        return PersonKinematics(
            track_id=track_id, d_m=d_m, bearing_rad=bearing,
            x_m=x, y_m=y, vx_ms=vx, vy_ms=vy,
            closing_ms=max(0.0, closing), tan_speed_ms=tan_speed,
            t_cpa_s=t_cpa, d_min_m=d_min,
            hazard=hazard,
            status=status, n_obs=self._n_obs(track_id),
            awareness=aw_s, risk=risk,
        )

    def _smooth_and_fuse(
        self,
        t_s: float,
        track_id: int | None,
        hazard_raw: float,
        awareness_raw: float | None,
    ) -> tuple:
        """Per-track EMA of the two channels, then fusion.

        Untracked detections (``track_id is None``) get no smoothing —
        there is no identity to average over. A gap longer than
        ``max_gap_s`` restarts the EMA (the ID may have been re-used).
        """
        p = self._params
        if track_id is None or p.ema_tau_s <= 0.0:
            hazard, aw = hazard_raw, awareness_raw
        else:
            prev = self._ema.get(track_id)
            if prev is None or t_s - prev[0] > p.max_gap_s:
                hazard, aw = hazard_raw, awareness_raw
            else:
                alpha = 1.0 - math.exp(-(max(0.0, t_s - prev[0])) / p.ema_tau_s)
                hazard = prev[1] + alpha * (hazard_raw - prev[1])
                if awareness_raw is None or prev[2] is None:
                    aw = awareness_raw
                else:
                    aw = prev[2] + alpha * (awareness_raw - prev[2])
            self._ema[track_id] = (t_s, hazard, aw)
        risk = None if aw is None else fuse_risk(hazard, aw, p.beta)
        return hazard, aw, risk

    def _n_obs(self, track_id: int | None) -> int:
        if track_id is None or track_id not in self._history:
            return 0
        return len(self._history[track_id])

    def _prune_stale(self, t_s: float) -> None:
        stale = [
            tid
            for tid, hist in self._history.items()
            if not hist or t_s - hist[-1][0] > self._params.max_gap_s
        ]
        for tid in stale:
            del self._history[tid]
        for tid in [
            t for t, e in self._ema.items()
            if t_s - e[0] > self._params.max_gap_s
        ]:
            del self._ema[tid]
