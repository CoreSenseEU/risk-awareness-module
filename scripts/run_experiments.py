"""
run_experiments.py

Runs experiments on a dataset (and run if applicable) and evaluates the results.
"""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
from riskam.data.ml_datasets import DATASETS
from riskam.experiments import inspect_predictions, run_experiments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments on a dataset")
    parser.add_argument(
        "action", type=str, choices=["run", "inspect"], help="The dataset to process."
    )
    parser.add_argument(
        "dataset", type=str, choices=DATASETS.keys(), help="The dataset to process."
    )
    parser.add_argument(
        "--run",
        type=str,
        choices=[f"RB_0{i}" for i in range(1, 9)],
        help="The run to process",
    )
    parser.add_argument(
        "--pred",
        type=str,
        choices=["correct", "underestimate", "overestimate"],
        help="The prediction type to inspect.",
    )
    args = parser.parse_args()

    if args.action == "inspect":
        inspect_predictions(args.dataset, args.pred, args.run)
    elif args.action == "run":
        run_experiments(args.dataset, args.run)
