"""
riskam.eval_summary

Cross-run aggregation and ranked reports for RiskAM experiment sweeps.

Reads per-experiment ``results.json`` files produced by
:func:`riskam.experiments.run_experiment` and aggregates them into:
  - ``summary.json`` — machine-readable per-config aggregate across all runs
  - ``summary.md``   — human-readable ranked top-K table + per-run breakdown

Aggregation semantics
---------------------
Two experiments share a *config* iff they have the same ``params_slug``. For
each config we aggregate across runs:

* Confusion matrices are summed cell-wise; per-class / macro / micro metrics
  are **recomputed** from the summed matrix (statistically correct pooling,
  rather than averaging ratios).
* ``correct`` / ``underestimate`` / ``overestimate`` / ``total`` are summed.
* ``mae`` / ``rmse`` are sample-weighted across runs — ``mae_pooled =
  Σ(mae_i · n_i) / Σ n_i``, and analogously ``rmse_pooled = sqrt(Σ(rmse_i² · n_i)
  / Σ n_i)``.
* ``avg_time`` is the unweighted mean of per-run ``avg_time`` values.
"""

from __future__ import annotations

import datetime
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from riskam.eval_metrics import (
    N_CLASSES,
    macro_prf,
    micro_prf,
    per_class_prf,
)


RESULTS_JSON = "results.json"
SUMMARY_JSON = "summary.json"
SUMMARY_MD = "summary.md"

# Metrics keys permitted as the ranking key; all are "higher is better" except
# MAE/RMSE which the ranker negates internally.
RANK_METRIC_CHOICES = ("macro_f1", "accuracy", "mae", "rmse")


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class ExperimentRecord:
    """One experiment's worth of results + identifying metadata."""

    run: str                # e.g. "RB_01"
    params_slug: str        # e.g. "w_prox_0.7-w_gaze_0.25-..."
    path: Path              # directory containing results.json
    results: dict           # parsed results.json


# ── Collection ───────────────────────────────────────────────────────────────


def collect_results(dataset_root: Path) -> list[ExperimentRecord]:
    """Walk ``dataset_root`` and return all per-experiment records.

    Expected layout (as emitted by ``run_experiment``)::
        <dataset_root>/<run>/<params_slug>/results.json
    """
    records: list[ExperimentRecord] = []
    if not dataset_root.is_dir():
        return records

    for run_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        for exp_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            results_path = exp_dir / RESULTS_JSON
            if not results_path.is_file():
                continue
            with results_path.open("r", encoding="utf-8") as f:
                results = json.load(f)
            records.append(
                ExperimentRecord(
                    run=run_dir.name,
                    params_slug=exp_dir.name,
                    path=exp_dir,
                    results=results,
                )
            )
    return records


# ── Aggregation ──────────────────────────────────────────────────────────────


def aggregate_by_config(records: list[ExperimentRecord]) -> dict[str, dict]:
    """Group records by ``params_slug`` and pool metrics across runs.

    Returns ``{params_slug: {"params": {...}, "runs": [...], "aggregate": {...}}}``.
    """
    by_slug: dict[str, list[ExperimentRecord]] = {}
    for r in records:
        by_slug.setdefault(r.params_slug, []).append(r)

    out: dict[str, dict] = {}
    for slug, recs in by_slug.items():
        out[slug] = _aggregate_one(recs)
    return out


def _aggregate_one(records: list[ExperimentRecord]) -> dict:
    """Pool a set of per-run results that share the same config."""
    # Sum confusion matrices; recompute per-class/macro/micro from the sum.
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    total = 0
    correct = 0
    under = 0
    over = 0
    skipped = 0

    mae_sum_weighted = 0.0
    mse_sum_weighted = 0.0  # sum(rmse_i^2 * n_i) — not sum of squared errors
    mae_weight = 0

    avg_times: list[float] = []

    for rec in records:
        res = rec.results
        classification = res.get("classification")
        if classification is None:
            raise ValueError(
                f"Record {rec.path} predates T3.3.4 and has no "
                "'classification' block; re-run the experiment."
            )
        cm += np.asarray(classification["confusion_matrix"], dtype=int)
        n = int(res.get("total", 0))
        total += n
        correct += int(res.get("correct", 0))
        under += int(res.get("underestimate", 0))
        over += int(res.get("overestimate", 0))
        skipped += int(res.get("skipped_no_depth", 0))

        if n > 0:
            mae = float(classification["regression"]["mae"])
            rmse = float(classification["regression"]["rmse"])
            mae_sum_weighted += mae * n
            mse_sum_weighted += (rmse ** 2) * n
            mae_weight += n

        avg_times.append(float(res.get("avg_time", 0.0)))

    per_class = per_class_prf(cm)
    pooled_mae = mae_sum_weighted / mae_weight if mae_weight > 0 else 0.0
    pooled_rmse = (
        math.sqrt(mse_sum_weighted / mae_weight) if mae_weight > 0 else 0.0
    )

    # The params dict is recorded per-experiment (T3.3.9). Runs with the same
    # slug share the same dict; pick the first record's copy.
    params = records[0].results.get("params", {})

    return {
        "params": params,
        "runs": sorted({r.run for r in records}),
        "n_runs": len({r.run for r in records}),
        "aggregate": {
            "total": total,
            "correct": correct,
            "underestimate": under,
            "overestimate": over,
            "skipped_no_depth": skipped,
            "confusion_matrix": cm.tolist(),
            "per_class": per_class,
            "macro": macro_prf(per_class),
            "micro": micro_prf(cm),
            "regression": {"mae": pooled_mae, "rmse": pooled_rmse},
            "avg_time_mean": (sum(avg_times) / len(avg_times)) if avg_times else 0.0,
        },
        "per_run": _per_run_breakdown(records),
    }


def _per_run_breakdown(records: list[ExperimentRecord]) -> dict:
    """One-row-per-run summary of the headline numbers."""
    out: dict = {}
    for rec in records:
        res = rec.results
        c = res.get("classification", {})
        out[rec.run] = {
            "total": int(res.get("total", 0)),
            "accuracy": float(c.get("micro", {}).get("f1", 0.0)),
            "macro_f1": float(c.get("macro", {}).get("f1", 0.0)),
            "mae": float(c.get("regression", {}).get("mae", 0.0)),
        }
    return out


# ── Ranking ──────────────────────────────────────────────────────────────────


def rank_configs(
    aggregated: dict[str, dict],
    metric: str = "macro_f1",
    top_k: int | None = None,
) -> list[tuple[str, dict]]:
    """Sort configs by the chosen metric and return at most ``top_k`` of them.

    ``metric`` ∈ ``RANK_METRIC_CHOICES``. For MAE/RMSE lower is better, so the
    sort is ascending; for the others it is descending.
    """
    if metric not in RANK_METRIC_CHOICES:
        raise ValueError(f"metric must be one of {RANK_METRIC_CHOICES}, got {metric!r}")

    def _key(item: tuple[str, dict]) -> float:
        agg = item[1]["aggregate"]
        if metric == "macro_f1":
            return -agg["macro"]["f1"]
        if metric == "accuracy":
            return -agg["micro"]["f1"]
        return agg["regression"][metric]  # mae | rmse, ascending

    ranked = sorted(aggregated.items(), key=_key)
    if top_k is not None:
        ranked = ranked[:top_k]
    return ranked


# ── Writers ──────────────────────────────────────────────────────────────────


def write_summary(
    dataset_root: Path,
    aggregated: dict[str, dict],
    rank_metric: str = "macro_f1",
    top_k: int = 10,
) -> tuple[Path, Path]:
    """Write ``summary.json`` and ``summary.md`` into ``dataset_root``.

    Returns the two written paths.
    """
    dataset_root.mkdir(parents=True, exist_ok=True)
    ranked = rank_configs(aggregated, metric=rank_metric, top_k=top_k)

    summary = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "n_configs": len(aggregated),
        "rank_metric": rank_metric,
        "ranked": [
            {"params_slug": slug, **info}
            for slug, info in rank_configs(aggregated, metric=rank_metric)
        ],
    }
    summary_json = dataset_root / SUMMARY_JSON
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary_md = dataset_root / SUMMARY_MD
    summary_md.write_text(_render_markdown(aggregated, ranked, rank_metric))

    return summary_json, summary_md


def _render_markdown(
    aggregated: dict[str, dict],
    ranked: list[tuple[str, dict]],
    rank_metric: str,
) -> str:
    if not ranked:
        return "# Summary\n\n_No experiments found._\n"

    lines: list[str] = []
    first_info = ranked[0][1]
    runs = first_info["runs"]
    lines.append("# Summary")
    lines.append("")
    lines.append(
        f"Aggregated across **{len(runs)} run(s)**: "
        + ", ".join(f"`{r}`" for r in runs) + "."
    )
    lines.append(
        f"Total configs: **{len(aggregated)}**. "
        f"Ranked by **{rank_metric}**."
    )
    lines.append("")

    # Top-K table
    lines.append(f"## Top {len(ranked)} configs")
    lines.append("")
    lines.append(
        "| Rank | Params slug | Macro F1 | Accuracy | MAE | RMSE | Frames |"
    )
    lines.append("|------|-------------|----------|----------|-----|------|--------|")
    for i, (slug, info) in enumerate(ranked, start=1):
        agg = info["aggregate"]
        lines.append(
            f"| {i} | `{slug}` | "
            f"{agg['macro']['f1']:.3f} | "
            f"{agg['micro']['f1']:.3f} | "
            f"{agg['regression']['mae']:.3f} | "
            f"{agg['regression']['rmse']:.3f} | "
            f"{agg['total']} |"
        )
    lines.append("")

    # Per-run breakdown of the best config
    best_slug, best_info = ranked[0]
    lines.append(f"## Per-run breakdown — best config (`{best_slug}`)")
    lines.append("")
    lines.append("| Run | Frames | Accuracy | Macro F1 | MAE |")
    lines.append("|-----|--------|----------|----------|-----|")
    for run in sorted(best_info["per_run"].keys()):
        r = best_info["per_run"][run]
        lines.append(
            f"| {run} | {r['total']} | "
            f"{r['accuracy']:.3f} | {r['macro_f1']:.3f} | {r['mae']:.3f} |"
        )
    lines.append("")

    return "\n".join(lines) + "\n"


# ── One-shot entry point ─────────────────────────────────────────────────────


def summarize(
    dataset_root: Path,
    rank_metric: str = "macro_f1",
    top_k: int = 10,
) -> dict:
    """Collect → aggregate → write summary. Returns the aggregated dict.

    Suitable for scripting: ``summarize(Path("exp_results/cs_robocup_2023"))``.
    """
    records = collect_results(dataset_root)
    if not records:
        raise FileNotFoundError(
            f"No results.json found under {dataset_root}. "
            "Run experiments first."
        )
    aggregated = aggregate_by_config(records)
    summary_json, summary_md = write_summary(
        dataset_root, aggregated, rank_metric=rank_metric, top_k=top_k
    )
    print(
        f"Wrote {summary_json.relative_to(dataset_root.parent)} "
        f"and {summary_md.relative_to(dataset_root.parent)} "
        f"({len(aggregated)} configs × "
        f"{len(set(r.run for r in records))} runs)."
    )
    return aggregated
