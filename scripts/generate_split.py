"""
generate_split.py

Generate (or regenerate) the canonical val/test split for cs_robocup_2023.
Stratified within-run by class — see riskam/data/splits.py for rationale.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
from riskam.data.paths import CS_ROBOCUP_2023_GROUND_TRUTH_PATH
from riskam.data.splits import (
    CS_ROBOCUP_2023_SPLIT_PATH,
    DEFAULT_SEED,
    DEFAULT_TEST_FRACTION,
    make_stratified_split,
    save_split,
    split_summary,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the cs_robocup_2023 val/test split."
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help=f"Fraction of frames per (run, class) sent to test. Default {DEFAULT_TEST_FRACTION}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for the within-class shuffle. Default {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Free-form notes recorded inside split.json.",
    )
    args = parser.parse_args()

    if not CS_ROBOCUP_2023_GROUND_TRUTH_PATH.exists():
        sys.exit(
            f"[error] ground truth not found at {CS_ROBOCUP_2023_GROUND_TRUTH_PATH}"
        )

    with CS_ROBOCUP_2023_GROUND_TRUTH_PATH.open("r", encoding="utf-8") as f:
        gt = json.load(f)

    split = make_stratified_split(
        gt,
        test_fraction=args.test_fraction,
        seed=args.seed,
        notes=args.notes,
    )
    save_split(CS_ROBOCUP_2023_SPLIT_PATH, split)
    print(f"Wrote split → {CS_ROBOCUP_2023_SPLIT_PATH}")
    print(json.dumps(split_summary(gt, split), indent=2))
