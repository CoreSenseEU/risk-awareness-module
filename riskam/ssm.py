"""
riskam.ssm

Speed-and-separation monitoring (SSM) in the style of ISO/TS 15066 §5.5.4 /
ISO 13855: the robot must stop before a human can close the protective
separation distance. Used by the Layer-2 evaluation as the industrial
baseline the RiskAM score is compared against — specifically the headline
"worst-case SSM vs awareness-modulated SSM at matched recall".

The protective distance here is the mobile-robot simplification

    S_p(v_r, v_h) = v_h * (t_r + t_s)  +  v_r * t_r  +  C

- ``v_h * (t_r + t_s)``: distance the human covers while the robot senses,
  reacts and stops (ISO 13855 directed-approach term, K = v_h);
- ``v_r * t_r``: distance the robot itself covers before braking begins
  (the braking distance itself is folded into ``t_s`` at TIAGo speeds);
- ``C``: intrusion/measurement margin (ISO 13855 C term).

Every constant is a documented, contestable choice carried in
:class:`SSMParams`; the evaluation sweeps *thresholds on the margin*
``d - S_p`` rather than committing to the margin-zero operating point, so
the matched-recall comparison is insensitive to the exact constants.

The "awareness-modulated" variant replaces the worst-case human approach
speed with an interpolation toward a residual speed for a human the metric
measures as robot-aware — the operational claim of the awareness hook.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SSMParams:
    """Constants of the protective-distance formula (documented choices)."""

    # ISO 13855 walking approach speed (1600 mm/s) — the worst case.
    v_h_ms: float = 1.6
    # Residual approach speed assumed for a fully robot-aware human: aware
    # people slow down / yield; 0.5 m/s ≈ hesitant walking. A modelling
    # choice, stated in the report; the matched-recall framing sweeps
    # thresholds so only the ordering it induces matters.
    v_h_aware_ms: float = 0.5
    # Sensing + processing latency: ~2 frame periods at 30 Hz plus pipeline.
    t_reaction_s: float = 0.3
    # TIAGo stop time from ~1 m/s (brake command to standstill), incl. the
    # braking distance folded in as time at the pre-brake speed.
    t_stop_s: float = 0.6
    # ISO 13855 intrusion margin + depth-measurement uncertainty.
    c_m: float = 0.3


def protective_distance(
    v_r_ms: float, params: SSMParams = SSMParams(), v_h_ms: float | None = None
) -> float:
    """Protective separation distance S_p for robot speed ``v_r_ms``.

    ``v_h_ms`` overrides the human approach speed (worst case when None).
    Robot speed is clamped at 0 (reversing away still needs the human term).
    """
    v_h = params.v_h_ms if v_h_ms is None else v_h_ms
    v_r = max(0.0, v_r_ms)
    return v_h * (params.t_reaction_s + params.t_stop_s) + v_r * params.t_reaction_s + params.c_m


def awareness_human_speed(awareness: float, params: SSMParams = SSMParams()) -> float:
    """Human approach speed under measured awareness ∈ [0, 1].

    Linear interpolation from the worst case (awareness 0) to the aware
    residual speed (awareness 1). NaN awareness degrades to the worst case.
    """
    if awareness is None or np.isnan(awareness):
        return params.v_h_ms
    a = float(np.clip(awareness, 0.0, 1.0))
    return a * params.v_h_aware_ms + (1.0 - a) * params.v_h_ms


def ssm_margins(
    d_m: np.ndarray,
    awareness: np.ndarray,
    v_r_ms: float,
    params: SSMParams = SSMParams(),
) -> tuple[float, float]:
    """Scene SSM margins ``(worst, aware)`` = min over persons of d − S_p.

    ``d_m``: per-person distances (dead-zone persons should be passed as
    0.0 — strongly negative margin, i.e. alarm, which is correct);
    ``awareness``: per-person awareness in [0, 1] (NaN → worst case).
    Empty scenes return ``(+inf, +inf)`` (no alarm). The margins are
    continuous channels: the evaluation thresholds them, with margin ≤ 0
    being the formula's native operating point.
    """
    d = np.asarray(d_m, dtype=float)
    if d.size == 0:
        return np.inf, np.inf
    s_p_worst = protective_distance(v_r_ms, params)
    worst = float(np.min(d - s_p_worst))
    aware_margins = [
        float(di - protective_distance(v_r_ms, params, awareness_human_speed(ai, params)))
        for di, ai in zip(d, np.asarray(awareness, dtype=float))
    ]
    return worst, float(np.min(aware_margins))
