"""
reeval_experiments.py

Recompute metrics for existing experiments from their cached
``raw_predictions.json`` files. Useful after tweaking
``RISK_SCORE_BREAKPOINTS`` or adding new metrics to ``riskam.eval_metrics``.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
from riskam.data.ml_datasets import DATASETS
from riskam.data.paths import CS_ROBOCUP_2023_GROUND_TRUTH_PATH
from riskam.experiments import EXP_ROOT_DIR
from riskam.reeval import reevaluate_tree


def _load_ground_truth_by_run(dataset: str) -> dict[str, dict[str, int]]:
    if dataset != "cs_robocup_2023":
        raise ValueError(f"Ground truth loader not implemented for {dataset!r}")
    with CS_ROBOCUP_2023_GROUND_TRUTH_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-evaluate experiments from their cached raw predictions."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=DATASETS.keys(),
        help="Dataset whose experiments to re-evaluate.",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="all",
        choices=["all", "val", "test"],
        help="Which split bucket's results to re-evaluate. Default 'all'.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help=(
            "Appended to output file stems. '' overwrites originals (default); "
            "'_v2' writes results_v2.json alongside."
        ),
    )
    args = parser.parse_args()

    dataset_root = EXP_ROOT_DIR / args.dataset / args.bucket
    gt = _load_ground_truth_by_run(args.dataset)
    n = reevaluate_tree(dataset_root, gt, output_suffix=args.output_suffix)
    print(f"Re-evaluated {n} experiment(s) under {dataset_root}.")
