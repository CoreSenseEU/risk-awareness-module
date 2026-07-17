"""
extract_crowdbot_v2.py

Extracts RGB, aligned depth, odometry twists and camera intrinsics from the
CrowdBot v2 ROS 1 bags staged in ``ros_datasets/crowdbot_v2/``. Pure Python
(``rosbags``) — runs in the normal uv environment, no ROS/Docker needed.

Usage:
    uv run python scripts/extract_crowdbot_v2.py [--run rds_1120]

See scripts/prepare_crowdbot_v2.sh for the disk-aware stage/extract/delete
loop over all seven runs.
"""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
from riskam.data.extract_crowdbot import RUNS, extract_crowdbot_v2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract the CrowdBot v2 bags.")
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        choices=sorted(RUNS),
        help="Extract a single run instead of all staged bags.",
    )
    args = parser.parse_args()
    extract_crowdbot_v2(run=args.run)
