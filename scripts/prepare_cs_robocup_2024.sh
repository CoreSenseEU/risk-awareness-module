#!/usr/bin/env bash
# prepare_cs_robocup_2024.sh
#
# Disk-aware, run-at-a-time preparation of the CoreSense RoboCup 2024
# dataset. The source archives (~53 GB zipped, ~146 GB of rosbags) stay in
# SRC_DIR; each run's bag is staged into ros_datasets/cs_robocup_2024/<run>/
# on its own, extracted to ml_datasets/cs_robocup_2024/raw_dataset/<run>/
# via the ROS 2 Docker wrapper, and the staged .db3 is deleted before the
# next run is touched. Small sidecar files (metadata.yaml, logs, demo
# videos) are kept in the run dir.
#
# The receptionist runs have no zip — their bags live extracted in SRC_DIR
# (the only copy). They are MOVED into the run dir for extraction and MOVED
# BACK afterwards, never deleted.
#
# Usage (from the repo root):
#   scripts/prepare_cs_robocup_2024.sh [run ...]
# Default: all 10 runs. Runs whose output already looks complete are
# skipped, so the script is resumable.

set -euo pipefail

SRC_DIR="${CS_ROBOCUP_2024_SRC:-$HOME/data/coresense_robocup_2024}"
ROS_DIR="ros_datasets/cs_robocup_2024"
ML_DIR="ml_datasets/cs_robocup_2024/raw_dataset"
ALL_RUNS=(storing_2 restaurant_1 receptionist_1 receptionist_2
          stickler_1 stickler_2 carry_1 carry_2 gpsr_1 gpsr_2)

# run -> "zipfile|inner_dir" ("-" zipfile = pre-extracted receptionist)
archive_for() {
  case "$1" in
    gpsr_1)         echo "GPSR.zip|GPSR_TRY_1" ;;
    gpsr_2)         echo "GPSR.zip|GPSR_TRY_2" ;;
    carry_1)        echo "carry_my_luggage.zip|carry_try_1" ;;
    carry_2)        echo "carry_my_luggage.zip|carry_try_2" ;;
    stickler_1)     echo "stickler for the rules.zip|stickler_try_1" ;;
    stickler_2)     echo "stickler for the rules.zip|stickler_try_2" ;;
    restaurant_1)   echo "restaurant.zip|restaurant" ;;
    storing_2)      echo "storing groceries.zip|storing_try_2" ;;
    receptionist_1) echo "-|rosbag2_2024_07_18-13_15_28_0.db3|metadata_recepcionist_try_1.yaml" ;;
    receptionist_2) echo "-|rosbag2_2024_07_18-13_56_14_0.db3|metadata_recepcionist_try_2.yaml" ;;
    *) echo "error: unknown run '$1'" >&2; return 1 ;;
  esac
}

free_gb() {
  df -g . | awk 'NR==2 {print $4}'
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
if ! docker info >/dev/null 2>&1; then
  echo "error: Docker is not running. Start Docker Desktop and retry." >&2
  exit 1
fi

RUNS=("${@:-${ALL_RUNS[@]}}")

for run in "${RUNS[@]}"; do
  echo ""
  echo "=== $run ==="

  if run_complete "$run"; then
    echo "output already complete, skipping."
    continue
  fi

  spec="$(archive_for "$run")"
  zipfile="${spec%%|*}"
  rest="${spec#*|}"
  run_dir="$ROS_DIR/$run"
  mkdir -p "$run_dir"

  moved_back_needed=""

  if [[ "$zipfile" == "-" ]]; then
    # Receptionist: move the bag (only copy!) in, restore afterwards.
    bag_name="${rest%%|*}"
    meta_name="${rest#*|}"
    src_bag="$SRC_DIR/receptionist/$bag_name"
    if [[ ! -f "$src_bag" && ! -f "$run_dir/$bag_name" ]]; then
      echo "error: $src_bag not found." >&2
      exit 1
    fi
    cp "$SRC_DIR/receptionist/$meta_name" "$run_dir/metadata.yaml"
    if [[ -f "$src_bag" ]]; then
      mv "$src_bag" "$run_dir/$bag_name"
    fi
    moved_back_needed="$run_dir/$bag_name|$src_bag"
  else
    inner="$rest"
    # Disk guard: need the bag itself plus ~60% for extracted frames.
    bag_bytes=$(unzip -l "$SRC_DIR/$zipfile" "$inner/*.db3" \
                | awk '/\.db3$/ {s+=$1} END {print s}')
    need_gb=$(( bag_bytes * 16 / 10 / 1024 / 1024 / 1024 + 2 ))
    # APFS reclaims space from just-deleted bags lazily; give it a few
    # minutes before concluding the disk is genuinely full.
    tries=0
    while (( $(free_gb) < need_gb )); do
      if (( ++tries > 10 )); then
        echo "error: need ~${need_gb} GB free for $run, have $(free_gb) GB." >&2
        exit 1
      fi
      echo "waiting for ${need_gb} GB free (have $(free_gb) GB, APFS reclaim lag)..."
      sleep 60
    done
    if ! ls "$run_dir"/*.db3 >/dev/null 2>&1; then
      echo "unzipping $inner from $zipfile ($((bag_bytes / 1024 / 1024 / 1024)) GB bag)..."
      unzip -o -j -q "$SRC_DIR/$zipfile" "$inner/*" -d "$run_dir" -x "$inner/log/*"
    fi
  fi

  echo "extracting images..."
  scripts/extract_ros2_dataset_macos.sh cs_robocup_2024 --run "$run"
  echo "extracting aux (odom + camera_info)..."
  scripts/extract_ros2_dataset_macos.sh cs_robocup_2024_aux --run "$run"

  if ! run_complete "$run"; then
    echo "error: extraction output incomplete for $run — bag kept for inspection." >&2
    exit 1
  fi

  # Cleanup: restore receptionist bags, delete staged zip-sourced bags.
  if [[ -n "$moved_back_needed" ]]; then
    mv "${moved_back_needed%%|*}" "${moved_back_needed#*|}"
  else
    rm -f "$run_dir"/*.db3
  fi

  n_rgb=$(ls "$ML_DIR/$run/rgb" | wc -l | tr -d ' ')
  n_depth=$(ls "$ML_DIR/$run/depth" | wc -l | tr -d ' ')
  n_odom=$(($(wc -l < "$ML_DIR/$run/odom.csv") - 1))
  size=$(du -sh "$ML_DIR/$run" | cut -f1)
  echo "$run done: $n_rgb rgb, $n_depth depth, $n_odom odom rows, $size, $(free_gb) GB free"
done

echo ""
echo "=== all requested runs processed ==="
du -sh "$ML_DIR"/* 2>/dev/null || true
