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
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from ROS2 datasets.")

    parser.add_argument(
        "dataset",
        type=str,
        choices=["cs_robocup_2023", "cs_robocup_2023_aux"],
        help=(
            "The dataset to extract. The '_aux' variant extracts only "
            "odometry twists + camera intrinsics (fast, no image decoding) "
            "for the kinematic metric."
        ),
        default="cs_robocup_2023",
    )

    args = parser.parse_args()

    if args.dataset == "cs_robocup_2023":
        extract_cs_robocup2023()
    elif args.dataset == "cs_robocup_2023_aux":
        extract_cs_robocup2023_aux()
