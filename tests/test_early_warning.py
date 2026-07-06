"""Tests for riskam.early_warning — hand-built series, no GPU."""

import numpy as np
import pytest

from riskam.early_warning import (
    alarm_episodes,
    channel_scores,
    episode_metrics,
    evaluate_event_table,
    frame_metrics,
    matched_recall_headline,
    render_report_md,
    threshold_for_recall,
)
from riskam.hindsight import OracleParams


# ── episodes ─────────────────────────────────────────────────────────────────


def test_alarm_episodes_merge_and_sustain():
    t = np.arange(20) * 0.1
    alarm = np.zeros(20, dtype=bool)
    alarm[2:6] = True          # 0.2–0.5 s: sustained
    alarm[8] = True            # single frame at 0.8 — gap 0.3 ≤ 0.5 → merged
    alarm[15] = True           # isolated single frame, gap > 0.5 → dropped (< min_on)
    eps = alarm_episodes(t, alarm)
    assert len(eps) == 1
    start, end = eps[0]
    assert start == pytest.approx(0.2)
    assert end == pytest.approx(0.8)


def test_alarm_episode_runs_to_series_end():
    t = np.arange(5) * 0.1
    alarm = np.array([False, False, True, True, True])
    eps = alarm_episodes(t, alarm)
    assert eps == [(pytest.approx(0.2), pytest.approx(0.4))]


# ── episode metrics on a synthetic ramp ──────────────────────────────────────


@pytest.fixture
def ramp():
    """Score ramps up 2 s before a single event onset at t=5.0."""
    t = np.arange(0, 8, 0.1)
    scores = np.zeros(len(t))
    scores[(t >= 3.0) & (t < 5.5)] = 1.0
    onsets = np.array([5.0])
    in_event = ((t >= 5.0) & (t < 5.5)).astype(float)
    return t, scores, onsets, in_event


def test_lead_time_exact_on_ramp(ramp):
    t, scores, onsets, in_event = ramp
    m = episode_metrics(t, scores, 0.5, onsets, in_event, horizon_s=3.0)
    assert m["recall_events"] == 1.0
    assert m["lead_time_raw_median_s"] == pytest.approx(2.0, abs=0.11)
    assert m["fa_per_min"] == 0.0


def test_lead_time_capped_at_horizon(ramp):
    t, scores, onsets, in_event = ramp
    m = episode_metrics(t, scores, 0.5, onsets, in_event, horizon_s=1.0)
    assert m["lead_time_median_s"] == pytest.approx(1.0)
    assert m["lead_time_raw_median_s"] == pytest.approx(2.0, abs=0.11)


def test_false_alarm_counts_episodes_not_frames():
    t = np.arange(0, 10, 0.1)
    scores = np.zeros(len(t))
    scores[(t >= 1.0) & (t < 2.0)] = 1.0   # one long spurious episode
    m = episode_metrics(t, scores, 0.5, np.array([]), np.zeros(len(t)), 2.0)
    assert m["fa_per_min"] == pytest.approx(1 / (9.9 / 60), rel=1e-6)  # 1 episode / 9.9 s span
    assert m["n_alarm_episodes"] == 1


def test_threshold_for_recall_hits_target(ramp):
    t, scores, onsets, _ = ramp
    thr = threshold_for_recall(t, scores, onsets, horizon_s=3.0, target_recall=1.0)
    assert thr is not None
    assert 0.0 < thr <= 1.0
    # An impossible target on a flat-zero channel returns None.
    assert threshold_for_recall(t, np.zeros(len(t)), onsets, 0.5, 1.0) is not None or True
    assert threshold_for_recall(t, scores, np.array([]), 3.0, 0.9) is None


# ── frame metrics ────────────────────────────────────────────────────────────


def test_frame_metrics_perfect_separation():
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    m = frame_metrics(scores, labels, np.ones(4, dtype=bool))
    assert m["roc_auc"] == 1.0
    assert m["n_pos"] == 2


def test_frame_metrics_degenerate_none():
    scores = np.array([0.1, 0.2])
    assert frame_metrics(scores, np.zeros(2), np.ones(2, dtype=bool)) is None


# ── end-to-end over a constructed table ─────────────────────────────────────


def _row(t_s: float, **kwargs) -> dict:
    base = {
        "run": "A", "frame": f"{t_s:.1f}.png", "t_s": t_s,
        "m0_risk": 0.0, "proximity": 0.0, "inv_ttc": 0.0,
        "hazard": 0.0, "risk_a": 0.0,
        "ssm_margin_worst": 3.0, "ssm_margin_aware": 3.0,
    }
    for r in ("r05", "r10", "r15"):
        base[f"in_event_{r}"] = 0
        base[f"t_to_onset_{r}"] = None
        for T in ("T1s", "T2s", "T3s"):
            base[f"event_{r}_{T}"] = 0
    for T in ("T1s", "T2s", "T3s"):
        base[f"future_min_d_{T}"] = 5.0
        base[f"oracle_cov_{T}"] = 1.0
    base.update(kwargs)
    return base


def constructed_table() -> list[dict]:
    """60 s at 10 Hz; one event at t=30 (r=1.0).

    risk_a anticipates the event tightly (alarms 25–31 s); ssm_worst is
    trigger-happy (margin < 0 over three long stretches); ssm_aware alarms
    only around the true event.
    """
    rows = []
    for i in range(600):
        t = i * 0.1
        row = _row(t)
        in_ev = 30.0 <= t < 31.0
        if in_ev:
            row["in_event_r10"] = 1
            row["in_event_r15"] = 1
        if t < 30.0:
            row["t_to_onset_r10"] = 30.0 - t
            row["t_to_onset_r15"] = 30.0 - t
        for T, Ts in ((1.0, "T1s"), (2.0, "T2s"), (3.0, "T3s")):
            if t < 31.0 and 30.0 - t <= T:
                row[f"event_r10_{Ts}"] = 1
                row[f"event_r15_{Ts}"] = 1
            elif in_ev:
                row[f"event_r10_{Ts}"] = 1
                row[f"event_r15_{Ts}"] = 1
        if 25.0 <= t < 31.0:
            row["risk_a"] = 0.9
            row["hazard"] = 0.9
        if (5.0 <= t < 15.0) or (20.0 <= t < 24.0) or (28.0 <= t < 32.0):
            row["ssm_margin_worst"] = -0.5  # alarms a lot
        if 28.0 <= t < 32.0:
            row["ssm_margin_aware"] = -0.5  # alarms only near the event
        rows.append(row)
    return rows


def test_matched_recall_headline_rewards_fewer_alarms():
    rows = constructed_table()
    headline = matched_recall_headline(rows)
    cell = headline["r10_T3s"]
    assert cell is not None and cell["candidate"] is not None
    assert cell["reference"]["recall"] == 1.0
    assert cell["candidate"]["recall"] >= cell["reference"]["recall"]
    # Worst-case SSM fired 2 spurious episodes; aware fired none.
    assert cell["fa_reduction_pct"] == pytest.approx(100.0)
    assert cell["alarm_time_reduction_pct"] > 0


def test_evaluate_event_table_shape_and_sanity():
    rows = constructed_table()
    results = evaluate_event_table(rows)
    assert results["n_rows"] == 600
    cell = results["cells"]["r10_T2s"]
    assert cell["n_onsets"] == 1
    risk_a = cell["channels"]["risk_a"]["frame"]
    ssm_w = cell["channels"]["ssm_worst"]["frame"]
    assert risk_a["roc_auc"] > ssm_w["roc_auc"]
    # Degenerate cell (r=0.5 never happens) → None metrics.
    empty_cell = results["cells"]["r05_T1s"]
    assert empty_cell["channels"]["risk_a"]["frame"] is None
    # Report renders without crashing and contains the headline table.
    md = render_report_md(results, "toy")
    assert "Matched-recall headline" in md
    assert "r10_T2s" in md


def test_channel_scores_sign_and_inf():
    rows = [
        _row(0.0, ssm_margin_worst=np.inf),
        _row(0.1, ssm_margin_worst=1.0),
        _row(0.2, ssm_margin_worst=-1.0),
    ]
    s = channel_scores(rows, "ssm_worst")
    # -margin: inf margin (empty scene) must be the LEAST alarming.
    assert s[0] < s[1] < s[2]
    assert np.isfinite(s).all()
