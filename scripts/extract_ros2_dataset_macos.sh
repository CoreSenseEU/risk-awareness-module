#!/usr/bin/env bash
# extract_ros2_dataset_macos.sh
#
# macOS wrapper for scripts/extract_ros2_dataset.py. Native ROS 2 isn't
# available on macOS, so extraction runs inside a ROS 2 rolling Docker
# container with the repo bind-mounted at /workspace; output lands under
# ml_datasets/<dataset>/raw_dataset/ on the host.
#
# Usage (from the repo root):
#   scripts/extract_ros2_dataset_macos.sh [dataset]
# Default dataset: cs_robocup_2023.
#
# Linux users with ROS 2 rolling installed should run extract_ros2_dataset.py
# directly via uv; this wrapper exists for the macOS dev workflow only.

set -euo pipefail

DATASET="${1:-cs_robocup_2023}"
IMAGE="ros:rolling-perception"

if [[ ! -f pyproject.toml ]]; then
  echo "error: run this script from the repository root." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "error: Docker is not running. Start Docker Desktop and retry." >&2
  exit 1
fi

docker run --rm -it \
  -v "$(pwd)":/workspace -w /workspace \
  "$IMAGE" \
  bash -lc "apt-get update -qq && \
            DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends python3-pil >/dev/null && \
            source /opt/ros/rolling/setup.bash && \
            python3 scripts/extract_ros2_dataset.py ${DATASET}"
