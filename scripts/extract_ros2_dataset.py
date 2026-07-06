"""
extract_ros2_dataset.py

Extracts RGB and depth images from ROS2 datasets.
"""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
# Note: we deliberately do NOT import riskam.data.ml_datasets here, since it
# pulls torch transitively. Extraction runs in a ROS-only environment (e.g.
# the macOS Docker wrapper) where torch is not installed. Keep this script's
# imports torch-free.
from riskam.data.extract_cs_robocup import (
    extract_cs_robocup2023,
    extract_cs_robocup2023_aux,
    extract_cs_robocup2024,
    extract_cs_robocup2024_aux,
)

EXTRACTORS = {
    "cs_robocup_2023": extract_cs_robocup2023,
    "cs_robocup_2023_aux": extract_cs_robocup2023_aux,
    "cs_robocup_2024": extract_cs_robocup2024,
    "cs_robocup_2024_aux": extract_cs_robocup2024_aux,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from ROS2 datasets.")

    parser.add_argument(
        "dataset",
        type=str,
        choices=sorted(EXTRACTORS),
        help=(
            "The dataset to extract. The '_aux' variant extracts only "
            "odometry twists + camera intrinsics (fast, no image decoding) "
            "for the kinematic metric."
        ),
        default="cs_robocup_2023",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help=(
            "Extract a single run (e.g. RB_02 or storing_2) instead of all "
            "runs present on disk. Enables run-at-a-time processing when "
            "disk space is tight."
        ),
    )

    args = parser.parse_args()

    EXTRACTORS[args.dataset](run=args.run)
