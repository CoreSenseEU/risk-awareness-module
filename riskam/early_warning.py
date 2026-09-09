"""
riskam.early_warning

Layer-2 evaluation: score every metric channel as an early-warning signal
against the hindsight oracle (the collision-warning / ADAS framing).

Definitions (single source of truth, matching :mod:`riskam.hindsight`):

- **event onset** — downward crossing of the smoothed oracle scene distance
  through r with release hysteresis (``hindsight.onset_times``);
- **frame label** for (r, T) — ``event_<r>_<T>``: the scene distance
  actually dips ≤ r within the next T seconds;
- **sustained alarm** — channel score ≥ threshold for ≥ ``min_on_s``,
  alarm episodes closer than ``merge_gap_s`` are merged;
- **detected onset** — a sustained alarm episode overlaps ``[t_e − T, t_e]``;
- **lead time** — onset time − start of the earliest covering alarm episode
  (reported both capped at T — "achieved lead within horizon" — and raw);
- **false alarm** — an alarm episode overlapping no event window
  ``[t_e − T, t_e + event duration]``;
- frame-level ROC/PR AUC is computed on frames (all frames, and the
  pre-event-only variant that drops frames already inside r); episode
  metrics (recall / lead / FA-per-minute) use episodes.

Frames whose oracle lookahead coverage is below ``MIN_ORACLE_COV`` are
masked out of the *negatives* (a truncated window cannot prove absence);
positives stay (an observed dip is an observed dip).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from riskam.hindsight import OracleParams, r_tag, t_tag

MIN_ORACLE_COV = 0.5
MIN_ALARM_ON_S = 0.2
ALARM_MERGE_GAP_S = 0.5
TARGET_RECALLS = (0.8, 0.9, 0.95)

# channel name -> (event-table column, sign); score = sign * column, higher
# = more alarming. SSM margins are distances (small/negative = danger).
CHANNELS: dict[str, tuple[str, int]] = {
    "m0": ("m0_risk", +1),
    "proximity": ("proximity", +1),
    "ttc": ("inv_ttc", +1),
    "hazard": ("hazard", +1),
    "risk_a": ("risk_a", +1),
    "ssm_worst": ("ssm_margin_worst", -1),
    "ssm_aware": ("ssm_margin_aware", -1),
}


def _column(rows: list[dict], name: str, default: float = np.nan) -> np.ndarray:
    out = np.empty(len(rows))
    for i, row in enumerate(rows):
        v = row.get(name)
        out[i] = default if v is None else float(v)
    return out


def channel_scores(rows: list[dict], channel: str) -> np.ndarray:
    """Signed channel scores; ±inf (empty scenes) mapped to ∓large-finite."""
    col, sign = CHANNELS[channel]
    x = sign * _column(rows, col)
    finite = np.isfinite(x)
    if finite.any():
        lo, hi = np.min(x[finite]), np.max(x[finite])
    else:
        lo, hi = -1.0, 1.0
    span = max(hi - lo, 1.0)
    x = np.where(np.isneginf(x), lo - span, x)
    x = np.where(np.isposinf(x), hi + span, x)
    return x


# ── frame-level metrics ──────────────────────────────────────────────────────


def frame_metrics(scores: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> dict | None:
    """ROC/PR AUC on the masked frames; None when the truth is degenerate."""
    s, y = scores[mask], labels[mask].astype(int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if len(y) == 0 or y.min() == y.max():
        return None
    return {
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "base_rate": float(y.mean()),
        "n_frames": int(len(y)),
        "n_pos": int(y.sum()),
    }


# ── episode machinery ────────────────────────────────────────────────────────


def alarm_episodes(
    t: np.ndarray,
    alarm: np.ndarray,
    min_on_s: float = MIN_ALARM_ON_S,
    merge_gap_s: float = ALARM_MERGE_GAP_S,
) -> list[tuple[float, float]]:
    """(start, end) of sustained alarm episodes from a boolean frame series."""
    episodes: list[list[float]] = []
    start = None
    for i in range(len(t)):
        if alarm[i] and start is None:
            start = t[i]
        elif not alarm[i] and start is not None:
            episodes.append([start, t[i - 1]])
            start = None
    if start is not None:
        episodes.append([start, t[-1]])
    # Merge episodes separated by short gaps.
    merged: list[list[float]] = []
    for ep in episodes:
        if merged and ep[0] - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = ep[1]
        else:
            merged.append(ep)
    return [
        (s, e) for s, e in merged if e - s >= min_on_s
    ]


def _event_windows(
    t: np.ndarray, onsets: np.ndarray, in_event: np.ndarray, horizon_s: float
) -> list[tuple[float, float]]:
    """Per onset: the no-penalty window [onset − T, onset + duration]."""
    windows = []
    for t_e in onsets:
        i = int(np.searchsorted(t, t_e))
        j = i
        while j < len(t) and in_event[j]:
            j += 1
        end = t[j - 1] if j > i else t_e
        windows.append((t_e - horizon_s, end))
    return windows


def episode_metrics(
    t: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    onsets: np.ndarray,
    in_event: np.ndarray,
    horizon_s: float,
) -> dict:
    """Recall / lead time / FA-per-minute at one operating threshold."""
    episodes = alarm_episodes(t, scores >= threshold)
    duration_min = max((t[-1] - t[0]) / 60.0, 1e-9) if len(t) else 1e-9

    detected, leads_capped, leads_raw = 0, [], []
    for t_e in onsets:
        covering = [ep for ep in episodes if ep[0] <= t_e and ep[1] >= t_e - horizon_s]
        if covering:
            detected += 1
            lead = t_e - min(ep[0] for ep in covering)
            leads_raw.append(lead)
            leads_capped.append(min(lead, horizon_s))

    windows = _event_windows(t, onsets, in_event, horizon_s)
    fa = sum(
        1 for ep in episodes
        if not any(ep[0] <= w_end and ep[1] >= w_start for w_start, w_end in windows)
    )

    alarm_time = sum(e - s for s, e in episodes)
    return {
        "threshold": float(threshold),
        "n_events": int(len(onsets)),
        "recall_events": float(detected / len(onsets)) if len(onsets) else None,
        "lead_time_median_s": float(np.median(leads_capped)) if leads_capped else None,
        "lead_time_raw_median_s": float(np.median(leads_raw)) if leads_raw else None,
        "fa_per_min": float(fa / duration_min),
        "n_alarm_episodes": int(len(episodes)),
        "alarm_time_frac": float(alarm_time / (duration_min * 60.0)),
    }


def threshold_for_recall(
    t: np.ndarray,
    scores: np.ndarray,
    onsets: np.ndarray,
    horizon_s: float,
    target_recall: float,
) -> float | None:
    """Least-alarming threshold whose event recall ≥ target (quantile sweep).

    Returns None when even the most sensitive threshold misses the target.
    """
    if len(onsets) == 0:
        return None
    finite = scores[np.isfinite(scores)]
    if len(finite) == 0:
        return None
    candidates = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 201)))
    best = None
    for thr in candidates:  # ascending: less → more alarming
        episodes = alarm_episodes(t, scores >= thr)
        detected = sum(
            1 for t_e in onsets
            if any(ep[0] <= t_e and ep[1] >= t_e - horizon_s for ep in episodes)
        )
        if detected / len(onsets) >= target_recall:
            best = float(thr)  # keep raising: fewer alarms, recall still met
        else:
            # recall is monotone non-increasing in thr; once lost, stop.
            if best is not None:
                break
    return best


# ── run-level assembly ───────────────────────────────────────────────────────


def _per_run(rows: list[dict]) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}
    for row in rows:
        runs.setdefault(row["run"], []).append(row)
    return runs


def _run_onsets(run_rows: list[dict], r: float) -> np.ndarray:
    """Onset times from the precomputed t_to_onset column (0-crossings)."""
    t = np.array([row["t_s"] for row in run_rows])
    tto = _column(run_rows, f"t_to_onset_{r_tag(r)}")
    onset_t = t + tto
    onsets = np.unique(onset_t[np.isfinite(onset_t)].round(6))
    return onsets


def evaluate_event_table(
    rows: list[dict],
    oracle_params: OracleParams = OracleParams(),
    channels: dict = CHANNELS,
    target_recalls: tuple = TARGET_RECALLS,
) -> dict:
    """The full Layer-2 evaluation over one dataset's event table."""
    runs = _per_run(rows)
    results: dict = {"cells": {}, "headline": {}, "n_rows": len(rows),
                     "runs": sorted(runs)}

    for r in oracle_params.r_grid_m:
        for T in oracle_params.t_grid_s:
            cell_key = f"{r_tag(r)}_{t_tag(T)}"
            label_col = f"event_{r_tag(r)}_{t_tag(T)}"
            cov_col = f"oracle_cov_{t_tag(T)}"
            in_event_col = f"in_event_{r_tag(r)}"

            labels = _column(rows, label_col)
            cov = _column(rows, cov_col)
            in_event = _column(rows, in_event_col)
            # Mask: negatives need adequate lookahead coverage.
            base_mask = (labels == 1) | (cov >= MIN_ORACLE_COV)
            pre_event_mask = base_mask & (in_event == 0)

            n_onsets = sum(len(_run_onsets(rr, r)) for rr in runs.values())
            cell: dict = {"n_onsets": int(n_onsets),
                          "n_event_frames": int((labels == 1).sum()),
                          "channels": {}}

            for name in channels:
                scores = channel_scores(rows, name)
                ch: dict = {
                    "frame": frame_metrics(scores, labels, base_mask),
                    "frame_pre_event": frame_metrics(scores, labels, pre_event_mask),
                    "operating_points": {},
                }
                # Episode metrics per run, pooled at matched thresholds.
                for target in target_recalls:
                    pooled = _pooled_operating_point(
                        runs, name, r, T, target
                    )
                    ch["operating_points"][f"recall_{target}"] = pooled
                cell["channels"][name] = ch

            results["cells"][cell_key] = cell

    results["headline"] = matched_recall_headline(rows, oracle_params)
    results["ssm_decomposition"] = ssm_decomposition(rows, oracle_params)
    return results


def _pooled_operating_point(
    runs: dict[str, list[dict]], channel: str, r: float, T: float,
    target_recall: float,
) -> dict | None:
    """Threshold on the pooled score distribution; metrics summed over runs."""
    all_rows = [row for rr in runs.values() for row in rr]
    scores_all = channel_scores(all_rows, channel)
    # Pool onsets across runs for the threshold sweep target.
    per_run_data = []
    for rr in runs.values():
        t = np.array([row["t_s"] for row in rr])
        scores = channel_scores(rr, channel)
        onsets = _run_onsets(rr, r)
        in_ev = _column(rr, f"in_event_{r_tag(r)}")
        per_run_data.append((t, scores, onsets, in_ev))

    n_onsets_total = sum(len(o) for _, _, o, _ in per_run_data)
    if n_onsets_total == 0:
        return None

    def pooled_recall(thr: float) -> float:
        detected = 0
        for t, scores, onsets, _ in per_run_data:
            episodes = alarm_episodes(t, scores >= thr)
            detected += sum(
                1 for t_e in onsets
                if any(ep[0] <= t_e and ep[1] >= t_e - T for ep in episodes)
            )
        return detected / n_onsets_total

    finite = scores_all[np.isfinite(scores_all)]
    if len(finite) == 0:
        return None
    candidates = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 201)))
    thr = None
    for cand in candidates:
        if pooled_recall(cand) >= target_recall:
            thr = float(cand)
        elif thr is not None:
            break
    if thr is None:
        return None

    agg = {"threshold": thr, "n_events": n_onsets_total, "recall_events": 0.0,
           "fa_per_min": 0.0, "n_alarm_episodes": 0, "alarm_time_frac": 0.0}
    leads = []
    total_minutes = 0.0
    detected_total = 0
    fa_total = 0
    alarm_time_total = 0.0
    for t, scores, onsets, in_ev in per_run_data:
        m = episode_metrics(t, scores, thr, onsets, in_ev, T)
        minutes = (t[-1] - t[0]) / 60.0 if len(t) > 1 else 0.0
        total_minutes += minutes
        fa_total += m["fa_per_min"] * minutes
        alarm_time_total += m["alarm_time_frac"] * minutes * 60.0
        if m["recall_events"] is not None:
            detected_total += round(m["recall_events"] * m["n_events"])
        if m["lead_time_median_s"] is not None:
            leads.append(m["lead_time_median_s"])
        agg["n_alarm_episodes"] += m["n_alarm_episodes"]
    agg["recall_events"] = detected_total / n_onsets_total
    agg["fa_per_min"] = fa_total / max(total_minutes, 1e-9)
    agg["alarm_time_frac"] = alarm_time_total / max(total_minutes * 60.0, 1e-9)
    agg["lead_time_median_s"] = float(np.median(leads)) if leads else None
    return agg


def matched_recall_headline(
    rows: list[dict],
    oracle_params: OracleParams = OracleParams(),
    ref: str = "ssm_worst",
    cand: str = "ssm_aware",
) -> dict:
    """The paper headline: awareness-modulated SSM vs worst-case SSM.

    Reference operating point = worst-case SSM at its native margin ≤ 0
    threshold (score = −margin ≥ 0). The candidate's threshold is matched
    to the reference's pooled event recall; the deliverable is the
    false-alarm and alarm-time reduction at that equal-safety point,
    per (r, T) cell.
    """
    runs = _per_run(rows)
    headline: dict = {}
    for r in oracle_params.r_grid_m:
        for T in oracle_params.t_grid_s:
            key = f"{r_tag(r)}_{t_tag(T)}"
            per_run_data = {}
            for run, rr in runs.items():
                t = np.array([row["t_s"] for row in rr])
                onsets = _run_onsets(rr, r)
                in_ev = _column(rr, f"in_event_{r_tag(r)}")
                per_run_data[run] = (t, rr, onsets, in_ev)
            n_onsets = sum(len(o) for _, _, o, _ in per_run_data.values())
            if n_onsets == 0:
                headline[key] = None
                continue

            def pooled(channel: str, thr: float) -> dict:
                detected, fa_min_num, minutes_total, alarm_s = 0, 0.0, 0.0, 0.0
                episodes_n = 0
                for t, rr, onsets, in_ev in per_run_data.values():
                    scores = channel_scores(rr, channel)
                    m = episode_metrics(t, scores, thr, onsets, in_ev, T)
                    minutes = (t[-1] - t[0]) / 60.0 if len(t) > 1 else 0.0
                    minutes_total += minutes
                    fa_min_num += m["fa_per_min"] * minutes
                    alarm_s += m["alarm_time_frac"] * minutes * 60.0
                    episodes_n += m["n_alarm_episodes"]
                    if m["recall_events"] is not None:
                        detected += round(m["recall_events"] * m["n_events"])
                return {
                    "recall": detected / n_onsets,
                    "fa_per_min": fa_min_num / max(minutes_total, 1e-9),
                    "alarm_time_frac": alarm_s / max(minutes_total * 60.0, 1e-9),
                    "n_alarm_episodes": episodes_n,
                }

            # Reference: worst-case SSM at margin <= 0  <=>  -margin >= 0.
            ref_stats = pooled(ref, 0.0)
            target = ref_stats["recall"]
            if target == 0.0:
                headline[key] = {"reference": ref_stats, "candidate": None,
                                 "note": "reference detects nothing at native threshold"}
                continue

            # Candidate threshold matched to the reference recall.
            all_rows = [row for _, rr, _, _ in per_run_data.values() for row in rr]
            cand_scores_all = channel_scores(all_rows, cand)
            finite = cand_scores_all[np.isfinite(cand_scores_all)]
            candidates = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 201)))
            thr = None
            for c in candidates:
                if pooled(cand, c)["recall"] >= target:
                    thr = float(c)
                elif thr is not None:
                    break
            if thr is None:
                headline[key] = {"reference": ref_stats, "candidate": None,
                                 "note": "candidate cannot match reference recall"}
                continue
            cand_stats = pooled(cand, thr)
            fa_red = (
                100.0 * (1.0 - cand_stats["fa_per_min"] / ref_stats["fa_per_min"])
                if ref_stats["fa_per_min"] > 0 else None
            )
            at_red = (
                100.0 * (1.0 - cand_stats["alarm_time_frac"] / ref_stats["alarm_time_frac"])
                if ref_stats["alarm_time_frac"] > 0 else None
            )
            headline[key] = {
                "n_events": n_onsets,
                "reference": ref_stats,
                "candidate": {**cand_stats, "threshold": thr},
                "fa_reduction_pct": fa_red,
                "alarm_time_reduction_pct": at_red,
            }
    return headline


def ssm_decomposition(
    rows: list[dict],
    oracle_params: OracleParams = OracleParams(),
    ref: str = "ssm_worst",
    cand: str = "ssm_aware",
    min_onsets_per_run: int = 10,
) -> dict:
    """Decompose the SSM headline into threshold tuning + awareness increment.

    The matched-recall headline conflates two effects: part of the savings is
    available to anyone who simply *re-calibrates the stock worst-case
    formula* to the same recall (the reference's native margin ≤ 0 operating
    point is deliberately conservative), and only the remainder is bought by
    measuring awareness. This function makes that split explicit — the honest
    control for the headline (first computed in the 2026-07-07 investigation,
    promoted here into the standard report).

    Per qualifying run (≥ ``min_onsets_per_run`` onsets in the cell):

    - stock reference: ``ref`` at its native threshold (margin ≤ 0) → recall
      ``r0`` and alarm-time ``at0``;
    - re-tuned reference: ``ref`` threshold matched per run to ``r0`` →
      ``at_w`` (calibration-only savings);
    - candidate: ``cand`` threshold matched per run to ``r0`` → ``at_a``.

    ``headline = 1 − at_a/at0``, ``tuning = 1 − at_w/at0``, ``awareness =
    (at_w − at_a)/at0``; headline = tuning + awareness by construction.
    Aggregation across runs is duration-weighted. Thresholds are matched per
    run (the per-deployment setting), so these numbers are the citable form;
    the pooled table above them remains for continuity.
    """
    runs = _per_run(rows)
    out: dict = {}
    for r in oracle_params.r_grid_m:
        for T in oracle_params.t_grid_s:
            key = f"{r_tag(r)}_{t_tag(T)}"
            per_run: dict = {}
            for run, rr in sorted(runs.items()):
                t = np.array([row["t_s"] for row in rr])
                onsets = _run_onsets(rr, r)
                if len(onsets) < min_onsets_per_run:
                    continue
                in_ev = _column(rr, f"in_event_{r_tag(r)}")
                w = channel_scores(rr, ref)
                a = channel_scores(rr, cand)
                stock = episode_metrics(t, w, 0.0, onsets, in_ev, T)
                r0, at0 = stock["recall_events"], stock["alarm_time_frac"]
                if not r0 or at0 <= 0:
                    continue
                thr_w = threshold_for_recall(t, w, onsets, T, r0)
                thr_a = threshold_for_recall(t, a, onsets, T, r0)
                if thr_w is None or thr_a is None:
                    continue
                at_w = episode_metrics(t, w, thr_w, onsets, in_ev, T)[
                    "alarm_time_frac"
                ]
                at_a = episode_metrics(t, a, thr_a, onsets, in_ev, T)[
                    "alarm_time_frac"
                ]
                per_run[run] = {
                    "n_events": int(len(onsets)),
                    "recall_stock": float(r0),
                    "alarm_time_stock": float(at0),
                    "alarm_time_tuned_ref": float(at_w),
                    "alarm_time_candidate": float(at_a),
                    "headline_pct": 100.0 * (1.0 - at_a / at0),
                    "tuning_pct": 100.0 * (1.0 - at_w / at0),
                    "awareness_pct": 100.0 * (at_w - at_a) / at0,
                    "duration_s": float(t[-1] - t[0]),
                }
            if not per_run:
                out[key] = None
                continue
            wts = np.array([v["duration_s"] for v in per_run.values()])
            wmean = lambda field: float(np.average(  # noqa: E731
                [v[field] for v in per_run.values()], weights=wts
            ))
            out[key] = {
                "n_runs": len(per_run),
                "n_events": int(sum(v["n_events"] for v in per_run.values())),
                "per_run": per_run,
                "weighted": {
                    "headline_pct": wmean("headline_pct"),
                    "tuning_pct": wmean("tuning_pct"),
                    "awareness_pct": wmean("awareness_pct"),
                },
            }
    return out


# ── report rendering ─────────────────────────────────────────────────────────


def render_report_md(results: dict, dataset: str) -> str:
    """Markdown report over `evaluate_event_table` output."""
    lines = [f"# Layer 2 early-warning report — {dataset}", ""]
    lines.append(f"Rows: {results['n_rows']}, runs: {', '.join(results['runs'])}")
    lines.append("")
    lines.append("## Frame-level AUC per (r, T) cell")
    lines.append("")
    header = "| cell | onsets | base rate | " + " | ".join(CHANNELS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (3 + len(CHANNELS)))
    for cell_key, cell in results["cells"].items():
        aucs = []
        base = None
        for name in CHANNELS:
            fm = cell["channels"][name]["frame"]
            aucs.append(f"{fm['roc_auc']:.3f}" if fm else "—")
            if fm:
                base = fm["base_rate"]
        base_str = f"{base:.3f}" if base is not None else "—"
        lines.append(
            f"| {cell_key} | {cell['n_onsets']} | {base_str} | "
            + " | ".join(aucs) + " |"
        )
    lines.append("")
    lines.append("## Pre-event-only ROC AUC (pure anticipation)")
    lines.append("")
    lines.append(header)
    lines.append("|" + "---|" * (3 + len(CHANNELS)))
    for cell_key, cell in results["cells"].items():
        aucs = []
        for name in CHANNELS:
            fm = cell["channels"][name]["frame_pre_event"]
            aucs.append(f"{fm['roc_auc']:.3f}" if fm else "—")
        lines.append(f"| {cell_key} | {cell['n_onsets']} | — | " + " | ".join(aucs) + " |")
    lines.append("")
    lines.append("## Matched-recall headline: worst-case SSM vs awareness-modulated SSM")
    lines.append("")
    lines.append(
        "In high-event-density recordings the worst-case reference is in alarm "
        "for a large fraction of total time, so its episode-count FA/min is "
        "deceptively low (few, enormous episodes that always overlap some "
        "event). **Alarm-time reduction at matched recall is the primary "
        "equal-safety comparison**; episode FA/min is reported for "
        "completeness (fragmenting one always-on alarm into shorter ones "
        "raises it mechanically)."
    )
    lines.append("")
    lines.append("| cell | events | ref recall | ref alarm-time | cand alarm-time | alarm-time reduction | FA/min ref→cand |")
    lines.append("|---|---|---|---|---|---|---|")
    for cell_key, h in results["headline"].items():
        if h is None or h.get("candidate") is None:
            note = (h or {}).get("note", "no events")
            lines.append(f"| {cell_key} | — | — | — | — | {note} | — |")
            continue
        ref, cand = h["reference"], h["candidate"]
        at_red = (f"**{h['alarm_time_reduction_pct']:.0f}%**"
                  if h["alarm_time_reduction_pct"] is not None else "—")
        lines.append(
            f"| {cell_key} | {h['n_events']} | {ref['recall']:.2f} | "
            f"{ref['alarm_time_frac']:.2f} | {cand['alarm_time_frac']:.2f} | {at_red} | "
            f"{ref['fa_per_min']:.2f}→{cand['fa_per_min']:.2f} |"
        )
    lines.append("")
    lines.append(
        "Cells with fewer than 10 onsets are unreliable; the headline cell is "
        "the largest (r, T) with ≥10 pooled onsets (pre-registered rule)."
    )
    decomp = results.get("ssm_decomposition")
    if decomp:
        lines.append("")
        lines.append(
            "## Decomposition: threshold tuning vs awareness increment "
            "(per-run matched recall)"
        )
        lines.append("")
        lines.append(
            "The headline above conflates two effects: savings available by "
            "simply **re-calibrating the stock worst-case formula** to the "
            "same recall (its native margin ≤ 0 operating point is "
            "deliberately conservative), and the increment **only measured "
            "awareness buys**. Here each run's thresholds are matched to the "
            "stock reference's own recall on that run (the per-deployment "
            "setting); runs with fewer than 10 onsets are excluded; "
            "aggregation is duration-weighted. By construction "
            "headline = tuning + awareness."
        )
        lines.append("")
        lines.append(
            "| cell | runs | events | headline | = tuning | + awareness |"
        )
        lines.append("|---|---|---|---|---|---|")
        for cell_key, d in decomp.items():
            if d is None:
                lines.append(
                    f"| {cell_key} | — | — | — | — | no qualifying runs |"
                )
                continue
            wtd = d["weighted"]
            lines.append(
                f"| {cell_key} | {d['n_runs']} | {d['n_events']} | "
                f"{wtd['headline_pct']:.0f}% | {wtd['tuning_pct']:.0f}% | "
                f"**{wtd['awareness_pct']:.0f}%** |"
            )
    lines.append("")
    return "\n".join(lines)
