"""Why the SSM headline (awareness-modulated vs worst-case at matched recall)
does not replicate on cs_robocup_2024 — investigation of 2026-07-07.

All diagnostics run off the persisted Layer-2 event tables; cell r05_T3s
(2023's strongest headline cell) unless noted. Findings in the paper plan's
checkpoint log (docs/private/paper-plan.md).

  A. dataset profile — crowding, robot speed, awareness signal, margin gap
  B. per-run matched-recall headline breakdown
  C. pooled savings vs matched-recall target (global threshold)
  D. alarm-episode structure at the native worst-case threshold
  E. pooled headline with PER-RUN candidate thresholds (pooling artifact test)
  F. saturation/density correlates of the per-run reduction
  G. per-run savings at fixed matched recall (worst-case also re-tuned)
  H. oracle event-window coverage (alarm-time floor at recall 1.0)
  K. decomposition: headline = threshold tuning + awareness increment
  I. P(encounter within 3 s | binding person's awareness) at matched geometry
  J. closing speed vs awareness at matched distance
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
from riskam.early_warning import (
    _column,
    _per_run,
    _pooled_operating_point,
    _run_onsets,
    alarm_episodes,
    channel_scores,
    episode_metrics,
    matched_recall_headline,
    threshold_for_recall,
)
from riskam.event_table import read_event_table_csv
from riskam.hindsight import OracleParams, r_tag

ROOT = Path(__file__).resolve().parents[1] / "exp_results"
DATASETS = ["cs_robocup_2023", "cs_robocup_2024"]
CELL_R, CELL_T, CELL_KEY = 0.5, 3.0, "r05_T3s"
GAP_FULL = 1.1 * 0.9  # (v_h - v_h_aware) * (t_r + t_s): margin gap at awareness 1


def q(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    x = x[np.isfinite(x)]
    return " ".join(f"p{int(100*p)}={v:.2f}" for p, v in zip(qs, np.quantile(x, qs)))


def profile(name, rows):
    print(f"\n=== A. profile: {name} ===")
    n_h = _column(rows, "n_humans")
    aware = _column(rows, "awareness")
    v_r = _column(rows, "v_robot_speed_ms")
    mw = _column(rows, "ssm_margin_worst")
    ma = _column(rows, "ssm_margin_aware")
    dz = _column(rows, "deadzone_any")
    t_min = sum(
        (max(rr, key=lambda r: r["t_s"])["t_s"] - min(rr, key=lambda r: r["t_s"])["t_s"])
        for rr in _per_run(rows).values()
    ) / 60.0
    occ = n_h > 0
    print(f"rows={len(rows)} runs={len(_per_run(rows))} minutes={t_min:.1f}")
    print(f"n_humans: mean={n_h.mean():.2f} {q(n_h)} | frames w/ humans: {occ.mean():.0%}")
    print(f"deadzone_any frac={np.nanmean(dz):.2%}")
    print(f"robot speed: mean={np.nanmean(v_r):.3f} {q(v_r)} m/s")
    print(f"awareness (scene person, occupied): {q(aware[occ])} | frac>0.5={np.mean(aware[occ]>0.5):.0%}")
    with np.errstate(invalid="ignore"):
        gap = ma - mw  # how much awareness relaxed the binding margin (0..~1 m)
    print(f"margin gap aware-worst (occupied): {q(gap[occ])} | frac<0.05m={np.mean(gap[occ]<0.05):.0%}")
    near = occ & (mw > -0.5) & (mw < 0.5)
    print(f"  near-boundary frames ({near.sum()}): gap {q(gap[near])} | frac<0.05m={np.mean(gap[near]<0.05):.0%}")


def per_run_headline(name, rows):
    print(f"\n=== B. per-run headline, {CELL_KEY}: {name} ===")
    op = OracleParams(r_grid_m=(CELL_R,), t_grid_s=(CELL_T,))
    print(f"{'run':16s} {'events':>6s} {'ev/min':>6s} {'ref rec':>7s} {'ref AT':>6s} {'cand AT':>7s} {'AT red':>7s}")
    for run, rr in sorted(_per_run(rows).items()):
        h = matched_recall_headline(rr, op)[CELL_KEY]
        if h is None or h.get("candidate") is None:
            print(f"{run:16s} {'—':>6s}  ({(h or {}).get('note', 'no events')})")
            continue
        minutes = (max(x["t_s"] for x in rr) - min(x["t_s"] for x in rr)) / 60.0
        print(f"{run:16s} {h['n_events']:6d} {h['n_events']/minutes:6.1f} "
              f"{h['reference']['recall']:7.2f} {h['reference']['alarm_time_frac']:6.2f} "
              f"{h['candidate']['alarm_time_frac']:7.2f} {h['alarm_time_reduction_pct']:6.0f}%")


def savings_curve(name, rows):
    print(f"\n=== C. pooled savings vs matched recall target (global threshold), {CELL_KEY}: {name} ===")
    runs = _per_run(rows)
    print(f"{'target':>6s} {'worst AT':>8s} {'aware AT':>8s} {'saving':>7s}")
    for target in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        w = _pooled_operating_point(runs, "ssm_worst", CELL_R, CELL_T, target)
        a = _pooled_operating_point(runs, "ssm_aware", CELL_R, CELL_T, target)
        if w is None or a is None:
            print(f"{target:6.2f}  unreachable")
            continue
        sav = 100.0 * (1.0 - a["alarm_time_frac"] / w["alarm_time_frac"])
        print(f"{target:6.2f} {w['alarm_time_frac']:8.2f} {a['alarm_time_frac']:8.2f} {sav:6.0f}%"
              f"   (recalls {w['recall_events']:.2f}/{a['recall_events']:.2f})")


def fragmentation(name, rows):
    print(f"\n=== D. episode structure at native worst threshold (margin<=0): {name} ===")
    for ch in ("ssm_worst", "ssm_aware"):
        n_ep, lens = 0, []
        for rr in _per_run(rows).values():
            t = np.array([r["t_s"] for r in rr])
            eps = alarm_episodes(t, channel_scores(rr, ch) >= 0.0)
            n_ep += len(eps)
            lens += [e - s for s, e in eps]
        med = np.median(lens) if lens else float("nan")
        print(f"{ch}: episodes={n_ep} median_len={med:.1f}s total={sum(lens)/60:.1f}min")


def per_run_threshold_pooled(name, rows):
    print(f"\n=== E/F. pooled headline with per-run thresholds, {CELL_KEY}: {name} ===")
    op = OracleParams(r_grid_m=(CELL_R,), t_grid_s=(CELL_T,))
    tot_min = ref_alarm_s = cand_alarm_s = 0.0
    n_events = 0
    reds, weights, ats, dens = [], [], [], []
    for run, rr in sorted(_per_run(rows).items()):
        h = matched_recall_headline(rr, op)[CELL_KEY]
        if h is None:
            continue
        minutes = (max(x["t_s"] for x in rr) - min(x["t_s"] for x in rr)) / 60.0
        ref, cand, ne = h["reference"], h.get("candidate"), h.get("n_events", 0)
        n_events += ne
        tot_min += minutes
        ref_alarm_s += ref["alarm_time_frac"] * minutes * 60.0
        at = (cand or ref)["alarm_time_frac"]  # no-match -> no savings
        cand_alarm_s += at * minutes * 60.0
        if cand is not None:
            reds.append(h["alarm_time_reduction_pct"])
            weights.append(minutes)
            ats.append(ref["alarm_time_frac"])
            dens.append(ne / minutes)
    pooled_red = 100.0 * (1.0 - cand_alarm_s / ref_alarm_s) if ref_alarm_s > 0 else float("nan")
    print(f"POOLED: alarm-time reduction {pooled_red:.0f}% "
          f"({ref_alarm_s/60:.1f} -> {cand_alarm_s/60:.1f} min over {tot_min:.1f} min)")
    print(f"per-run reductions: median {np.median(reds):.0f}%, "
          f"duration-weighted mean {np.average(reds, weights=weights):.0f}%, "
          f"positive in {sum(1 for x in reds if x > 0)}/{len(reds)} runs")
    if len(reds) >= 3:
        print(f"corr(AT red, ref alarm-time) = {np.corrcoef(ats, reds)[0,1]:.2f}, "
              f"corr(AT red, events/min) = {np.corrcoef(dens, reds)[0,1]:.2f}")


def fixed_recall(name, rows):
    print(f"\n=== G. per-run savings at fixed matched recall (worst also re-tuned), {CELL_KEY}: {name} ===")
    for target in (0.80, 0.85, 0.90):
        savs, weights = [], []
        for run, rr in sorted(_per_run(rows).items()):
            t = np.array([row["t_s"] for row in rr])
            onsets = _run_onsets(rr, CELL_R)
            if len(onsets) < 10:
                continue
            in_ev = _column(rr, f"in_event_{r_tag(CELL_R)}")
            w = channel_scores(rr, "ssm_worst")
            a = channel_scores(rr, "ssm_aware")
            thr_w = threshold_for_recall(t, w, onsets, CELL_T, target)
            if thr_w is None:
                continue
            m_w = episode_metrics(t, w, thr_w, onsets, in_ev, CELL_T)
            thr_a = threshold_for_recall(t, a, onsets, CELL_T, m_w["recall_events"])
            if thr_a is None or m_w["alarm_time_frac"] <= 0:
                continue
            m_a = episode_metrics(t, a, thr_a, onsets, in_ev, CELL_T)
            savs.append(100.0 * (1.0 - m_a["alarm_time_frac"] / m_w["alarm_time_frac"]))
            weights.append(t[-1] - t[0])
        if savs:
            print(f"target {target:.2f}: median {np.median(savs):5.0f}%, "
                  f"duration-weighted mean {np.average(savs, weights=weights):5.0f}%, "
                  f"positive in {sum(1 for s in savs if s > 0)}/{len(savs)} runs")


def window_coverage(name, rows):
    print(f"\n=== H. oracle event-window coverage, {CELL_KEY}: {name} ===")
    tot_s = cov_s = 0.0
    for run, rr in sorted(_per_run(rows).items()):
        t = np.array([row["t_s"] for row in rr])
        onsets = _run_onsets(rr, CELL_R)
        in_ev = _column(rr, f"in_event_{r_tag(CELL_R)}")
        windows = []
        for t_e in onsets:
            i = int(np.searchsorted(t, t_e))
            j = i
            while j < len(t) and in_ev[j]:
                j += 1
            end = t[j - 1] if j > i else t_e
            windows.append((t_e - CELL_T, end))
        windows.sort()
        merged = []
        for s, e in windows:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        run_len = t[-1] - t[0]
        cov_s += sum(min(e, t[-1]) - max(s, t[0]) for s, e in merged)
        tot_s += run_len
    print(f"DATASET: {cov_s/tot_s:.0%} of the timeline is inside an event window "
          f"-> alarm-time floor at recall 1.0")


def decompose(name, rows):
    print(f"\n=== K. decomposition headline = tuning + awareness, {CELL_KEY}: {name} ===")
    print(f"{'run':16s} {'R0':>5s} {'AT0':>5s} {'AT_w':>5s} {'AT_a':>5s} "
          f"{'headline':>8s} {'tuning':>7s} {'awareness':>9s}")
    heads, tuns, awas, weights = [], [], [], []
    for run, rr in sorted(_per_run(rows).items()):
        t = np.array([row["t_s"] for row in rr])
        onsets = _run_onsets(rr, CELL_R)
        if len(onsets) < 10:
            continue
        in_ev = _column(rr, f"in_event_{r_tag(CELL_R)}")
        w = channel_scores(rr, "ssm_worst")
        a = channel_scores(rr, "ssm_aware")
        m0 = episode_metrics(t, w, 0.0, onsets, in_ev, CELL_T)  # stock reference
        r0, at0 = m0["recall_events"], m0["alarm_time_frac"]
        if not r0 or at0 <= 0:
            continue
        thr_w = threshold_for_recall(t, w, onsets, CELL_T, r0)
        thr_a = threshold_for_recall(t, a, onsets, CELL_T, r0)
        if thr_w is None or thr_a is None:
            continue
        at_w = episode_metrics(t, w, thr_w, onsets, in_ev, CELL_T)["alarm_time_frac"]
        at_a = episode_metrics(t, a, thr_a, onsets, in_ev, CELL_T)["alarm_time_frac"]
        head = 100.0 * (1.0 - at_a / at0)
        tun = 100.0 * (1.0 - at_w / at0)
        awa = 100.0 * (at_w - at_a) / at0
        heads.append(head); tuns.append(tun); awas.append(awa)
        weights.append(t[-1] - t[0])
        print(f"{run:16s} {r0:5.2f} {at0:5.2f} {at_w:5.2f} {at_a:5.2f} "
              f"{head:7.0f}% {tun:6.0f}% {awa:8.0f}%")
    if heads:
        wm = lambda x: np.average(x, weights=weights)
        print(f"{'WEIGHTED MEAN':16s} {'':23s} {wm(heads):7.0f}% {wm(tuns):6.0f}% {wm(awas):8.0f}%")


def behaviour(name, rows):
    print(f"\n=== I/J. does awareness predict yielding? {name} ===")
    mw = _column(rows, "ssm_margin_worst")
    ma = _column(rows, "ssm_margin_aware")
    ev = _column(rows, "event_r05_T3s")
    in_ev = _column(rows, "in_event_r05")
    n_h = _column(rows, "n_humans")
    d = _column(rows, "d_m")
    closing = _column(rows, "closing_ms")
    aware_p = _column(rows, "awareness")
    kin = np.array([r["kin_status"] for r in rows])

    with np.errstate(invalid="ignore"):
        a_bind = np.clip((ma - mw) / GAP_FULL, 0.0, 1.0)  # binding person's awareness
    ok = (n_h > 0) & np.isfinite(mw) & np.isfinite(ev) & (in_ev == 0) & (kin != "dead_zone")
    print("I. event rate within 3 s (pre-event frames), by margin band x binding awareness")
    print(f"{'margin band':>14s} {'n(aw<0.3)':>9s} {'P(ev|unaware)':>13s} {'n(aw>0.7)':>9s} "
          f"{'P(ev|aware)':>11s} {'ratio':>6s}")
    for lo, hi in ((-0.5, 0.0), (0.0, 0.5), (0.5, 1.0)):
        band = ok & (mw >= lo) & (mw < hi)
        un = band & (a_bind < 0.3)
        aw = band & (a_bind > 0.7)
        if un.sum() < 50 or aw.sum() < 50:
            continue
        p_un, p_aw = ev[un].mean(), ev[aw].mean()
        print(f"[{lo:+.1f},{hi:+.1f}) m {un.sum():9d} {p_un:13.2f} {aw.sum():9d} "
              f"{p_aw:11.2f} {p_aw/max(p_un,1e-9):6.2f}")

    okj = (n_h > 0) & np.isfinite(closing) & np.isfinite(aware_p) & (kin == "full")
    print("J. closing speed (m/s) of scene-risk person, by distance band x awareness")
    print(f"{'distance band':>14s} {'n(aw<0.3)':>9s} {'v(unaware)':>10s} {'n(aw>0.7)':>9s} {'v(aware)':>9s}")
    for lo, hi in ((0.5, 1.5), (1.5, 2.5), (2.5, 4.0)):
        band = okj & (d >= lo) & (d < hi)
        un = band & (aware_p < 0.3)
        aw = band & (aware_p > 0.7)
        if un.sum() < 50 or aw.sum() < 50:
            continue
        print(f"[{lo:.1f},{hi:.1f}) m {un.sum():9d} {closing[un].mean():10.3f} "
              f"{aw.sum():9d} {closing[aw].mean():9.3f}")


for ds in DATASETS:
    rows = read_event_table_csv(ROOT / ds / "layer2" / "event_table.csv")
    profile(ds, rows)
    per_run_headline(ds, rows)
    savings_curve(ds, rows)
    fragmentation(ds, rows)
    per_run_threshold_pooled(ds, rows)
    fixed_recall(ds, rows)
    window_coverage(ds, rows)
    decompose(ds, rows)
    behaviour(ds, rows)
