"""
summarize_experiments.py

Aggregate all per-experiment results.json files under
``exp_results/<dataset>/`` into a single ``summary.json`` + ``summary.md``.
"""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
from riskam.data.ml_datasets import DATASETS
from riskam.eval_summary import RANK_METRIC_CHOICES, summarize
from riskam.experiments import EXP_ROOT_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate experiment results across runs and configs."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=DATASETS.keys(),
        help="The dataset whose results to summarize.",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="all",
        choices=["all", "val", "test"],
        help=(
            "Which split bucket's results to aggregate. 'all' = experiments "
            "run with no split filter. Default 'all'."
        ),
    )
    parser.add_argument(
        "--rank-metric",
        type=str,
        default="macro_f1",
        choices=RANK_METRIC_CHOICES,
        help="Metric used to rank configs (default: macro_f1).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Show the top-K configs in the markdown table (default: 10).",
    )
    args = parser.parse_args()

    dataset_root = EXP_ROOT_DIR / args.dataset / args.bucket
    summarize(dataset_root, rank_metric=args.rank_metric, top_k=args.top_k)
