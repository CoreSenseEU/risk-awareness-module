"""
visualize_exp_results.py

Visualizes experimental results
"""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
from riskam.data.ml_datasets import DATASETS
from riskam.visualization import visualize_exp_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments on a dataset")
    parser.add_argument(
        "dataset", type=str, choices=DATASETS.keys(), help="The dataset to process."
    )
    args = parser.parse_args()

    visualize_exp_results(args.dataset)
