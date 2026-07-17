#!/usr/bin/env bash
# prepare_crowdbot_v2.sh
#
# Disk-aware, run-at-a-time preparation of the CrowdBot v2 dataset (EPFL
# Qolo, Lausanne 2021-04-24 RDS session). The source archive (~32 GB zip)
# stays in SRC_DIR; each run's ROS 1 bag (2–7 GB) is unzipped into
# ros_datasets/crowdbot_v2/, extracted to
# ml_datasets/crowdbot_v2/raw_dataset/<run>/ with the pure-Python rosbags
# extractor (no Docker), and the staged bag is deleted before the next one.
#
# Usage (from the repo root):
#   scripts/prepare_crowdbot_v2.sh [run ...]
# Default: all 7 runs. Runs whose output already looks complete are
# skipped, so the script is resumable.

set -euo pipefail

SRC_ZIP="${CROWDBOT_V2_SRC:-$HOME/data/crowdbot_v2/rosbags_0424_rds_defaced.zip}"
ROS_DIR="ros_datasets/crowdbot_v2"
ML_DIR="ml_datasets/crowdbot_v2/raw_dataset"
ALL_RUNS=(rds_1120 rds_1123 rds_1135 rds_1140 rds_1143 rds_1148 rds_1155)

bag_for() {
  case "$1" in
    rds_1120) echo "defaced_2021-04-24-11-20-18_filtered_lidar_odom.bag" ;;
    rds_1123) echo "defaced_2021-04-24-11-23-43_filtered_lidar_odom.bag" ;;
    rds_1135) echo "defaced_2021-04-24-11-35-54_filtered_lidar_odom.bag" ;;
    rds_1140) echo "defaced_2021-04-24-11-40-33_filtered_lidar_odom.bag" ;;
    rds_1143) echo "defaced_2021-04-24-11-43-54_filtered_lidar_odom.bag" ;;
    rds_1148) echo "defaced_2021-04-24-11-48-21_filtered_lidar_odom.bag" ;;
    rds_1155) echo "defaced_2021-04-24-11-55-30_filtered_lidar_odom.bag" ;;
    *) echo "error: unknown run '$1'" >&2; return 1 ;;
  esac
}

run_complete() {
  local run="$1"
  [[ -f "$ML_DIR/$run/camera_info.json" && -f "$ML_DIR/$run/odom.csv" ]] \
    && [[ -n "$(ls -A "$ML_DIR/$run/rgb" 2>&1 | head -1)" ]] \
    && [[ -n "$(ls -A "$ML_DIR/$run/depth" 2>&1 | head -1)" ]]
}

if [[ ! -f pyproject.toml ]]; then
  echo "error: run this script from the repository root." >&2
  exit 1
fi
if [[ ! -f "$SRC_ZIP" ]]; then
  echo "error: source archive not found: $SRC_ZIP" >&2
  exit 1
fi

mkdir -p "$ROS_DIR"
RUNS=("${@:-${ALL_RUNS[@]}}")

for run in "${RUNS[@]}"; do
  echo ""
  echo "=== $run ==="

  if run_complete "$run"; then
    echo "output already complete, skipping."
    continue
  fi

  bag="$(bag_for "$run")"
  if [[ ! -f "$ROS_DIR/$bag" ]]; then
    echo "staging $bag..."
    unzip -o -q "$SRC_ZIP" "$bag" -d "$ROS_DIR"
  fi

  echo "extracting..."
  uv run python scripts/extract_crowdbot_v2.py --run "$run"

  if ! run_complete "$run"; then
    echo "error: extraction output incomplete for $run — bag kept for inspection." >&2
    exit 1
  fi
  rm -f "$ROS_DIR/$bag"

  n_rgb=$(ls "$ML_DIR/$run/rgb" | wc -l | tr -d ' ')
  n_depth=$(ls "$ML_DIR/$run/depth" | wc -l | tr -d ' ')
  size=$(du -sh "$ML_DIR/$run" | cut -f1)
  echo "$run done: $n_rgb rgb, $n_depth depth, $size"
done

echo ""
echo "=== all requested runs processed ==="
du -sh "$ML_DIR"/* 2>&1 || true
