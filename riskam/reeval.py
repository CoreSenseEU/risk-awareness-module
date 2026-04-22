"""
riskam.reeval

Eval-only mode: recompute metrics for one or more experiments without
re-running YOLO / featextr / the scorer.

Each ``run_experiment`` call writes a ``raw_predictions.json`` mapping
``{image_name: continuous_risk_score}``; given that file plus the ground
truth, every classification and regression metric in the framework can be
recomputed in milliseconds. This is useful when:

  * Risk-score breakpoints are tuned (``riskam.score.RISK_SCORE_BREAKPOINTS``).
  * New metrics are added to ``riskam.eval_metrics``.
  * A post-hoc diagnostic is wanted on historical runs.

Behaviour
---------
By default, ``results.json`` and ``predictions.json`` are overwritten
in-place and stamped with a fresh provenance block so the re-evaluation is
auditable. The original ``provenance.git_sha`` is preserved under
``reeval_of`` so the audit chain back to the inference run survives. A
different ``output_suffix`` writes to ``results{suffix}.json`` /
``predictions{suffix}.json`` alongside the originals.
"""

from __future__ import annotations

import json
from pathlib import Path

from riskam.eval_metrics import classification_report
from riskam.provenance import reproducibility_metadata


RAW_PREDICTIONS_JSON_FNAME = "raw_predictions.json"
PREDICTIONS_JSON_FNAME = "predictions.json"
RESULTS_JSON_FNAME = "results.json"


def _eval_prediction(gt: int, pred: float) -> str:
    """Mirror of ``experiments._eval_prediction`` — kept local so this module
    has no cyclic dependency on ``riskam.experiments``."""
    from riskam.score import RISK_SCORE_BREAKPOINTS, VERY_SMALL_RISK_VALUE

    if gt == 0:
        return "correct" if pred == 0.0 else "overestimate"
    lower = VERY_SMALL_RISK_VALUE if gt == 1 else RISK_SCORE_BREAKPOINTS[gt - 1]
    upper = RISK_SCORE_BREAKPOINTS[gt]
    if lower <= pred < upper:
        return "correct"
    if pred < lower:
        return "underestimate"
    return "overestimate"


def reevaluate_experiment(
    experiment_dir: Path,
    ground_truth: dict[str, int],
    output_suffix: str = "",
) -> dict:
    """Recompute metrics for one experiment from its raw predictions.

    Parameters
    ----------
    experiment_dir
        The ``exp_results/<dataset>/<bucket>/<run>/<slug>/`` directory.
    ground_truth
        ``{image_name: class_label}`` for this run.
    output_suffix
        Appended to the output file stems. ``""`` overwrites the originals;
        e.g. ``"_v2"`` writes ``results_v2.json`` alongside.

    Returns the new results dict.
    """
    raw_path = experiment_dir / RAW_PREDICTIONS_JSON_FNAME
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw predictions not found: {raw_path}")

    raw = json.loads(raw_path.read_text(encoding="utf-8"))

    metrics = {
        "correct": 0,
        "underestimate": 0,
        "overestimate": 0,
        "total": 0,
    }
    predictions: dict[str, list[str]] = {
        "correct": [],
        "underestimate": [],
        "overestimate": [],
    }
    y_true: list[int] = []
    y_pred: list[float] = []

    for img_name, score in raw.items():
        gt = ground_truth.get(img_name)
        if gt is None:
            continue
        eval_result = _eval_prediction(int(gt), float(score))
        metrics[eval_result] += 1
        metrics["total"] += 1
        predictions[eval_result].append(img_name)
        y_true.append(int(gt))
        y_pred.append(float(score))

    # Preserve split / params / avg_time / original provenance from the
    # inference-time results.json so the re-evaluation is traceable.
    prev_results_path = experiment_dir / RESULTS_JSON_FNAME
    prev = (
        json.loads(prev_results_path.read_text(encoding="utf-8"))
        if prev_results_path.is_file()
        else {}
    )

    metrics["avg_time"] = prev.get("avg_time", 0.0)
    if "skipped_no_depth" in prev:
        metrics["skipped_no_depth"] = prev["skipped_no_depth"]
    metrics["classification"] = classification_report(y_true, y_pred)
    metrics["params"] = prev.get("params")
    metrics["split"] = prev.get("split")
    if prev.get("split_meta") is not None:
        metrics["split_meta"] = prev["split_meta"]

    new_provenance = reproducibility_metadata()
    original_sha = (prev.get("provenance") or {}).get("git_sha")
    if original_sha is not None:
        new_provenance["reeval_of"] = original_sha
    metrics["provenance"] = new_provenance

    results_out = experiment_dir / f"results{output_suffix}.json"
    predictions_out = experiment_dir / f"predictions{output_suffix}.json"
    with results_out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    with predictions_out.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4)

    return metrics


def reevaluate_tree(
    dataset_root: Path,
    ground_truth_by_run: dict[str, dict[str, int]],
    output_suffix: str = "",
) -> int:
    """Walk a full ``exp_results/<dataset>/<bucket>/`` tree and re-evaluate
    every experiment with a ``raw_predictions.json``.

    Returns the number of experiments re-evaluated.
    """
    if not dataset_root.is_dir():
        return 0

    count = 0
    for run_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        gt = ground_truth_by_run.get(run_dir.name, {})
        if not gt:
            continue
        for exp_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            if (exp_dir / RAW_PREDICTIONS_JSON_FNAME).is_file():
                reevaluate_experiment(exp_dir, gt, output_suffix=output_suffix)
                count += 1
    return count
