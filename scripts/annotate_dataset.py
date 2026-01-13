""" """

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
from riskam.data.ml_datasets import DATASETS
from riskam.data.annotator import annotate_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate the dataset")
    parser.add_argument(
        "dataset", type=str, choices=DATASETS.keys(), help="The dataset to annotate"
    )
    parser.add_argument(
        "--run",
        type=str,
        choices=[f"RB_0{i}" for i in range(1, 9)],
        help="The run to annotate",
    )

    args = parser.parse_args()
    annotate_dataset(args.dataset, args.run)
