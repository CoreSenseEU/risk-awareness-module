#!/usr/bin/env bash
# smoke_test_node_macos.sh
#
# End-to-end smoke test of riskam_node inside a ROS 2 rolling Docker
# container: colcon-builds the ROS packages, imports the riskam library from
# source (so ml_models/ resolves), starts the node, feeds it real extracted
# frames via scripts/smoke_feed_frames.py, and asserts that both the weighted
# score and the kinematic companion channel publish. Exercises the
# camera_info intrinsics path with the run's real fx/cx.
#
# Usage (from the repo root):
#   scripts/smoke_test_node_macos.sh
#
# Prints SMOKE_PASS on success. Sibling of extract_ros2_dataset_macos.sh;
# exists for the macOS dev workflow — Linux users with ROS 2 run the node
# directly.

set -euo pipefail

IMAGE="ros:rolling-perception"

if [[ ! -f pyproject.toml ]]; then
  echo "error: run this script from the repository root." >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "error: Docker is not running. Start Docker Desktop and retry." >&2
  exit 1
fi

TTY_FLAGS=""
if [ -t 0 ]; then
  TTY_FLAGS="-it"
fi

docker run --rm ${TTY_FLAGS} \
  -v "$(pwd)":/workspace \
  "$IMAGE" \
  bash -lc '
    set -e
    echo "── installing python deps (the riskam setup.py closure, CPU) ──"
    apt-get update -qq >/dev/null
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      python3-pip >/dev/null
    # The Debian-owned typing_extensions cannot be uninstalled by pip; replace
    # it explicitly first so the ultralytics/torch install does not abort.
    pip3 install --break-system-packages -q --ignore-installed typing_extensions
    # The node imports riskam.visualization -> riskam.experiments, so the
    # full runtime closure from riskam/setup.py is required, not just YOLO.
    # numpy is pinned to the Debian-owned version so pip never attempts the
    # (impossible) uninstall; riskam supports numpy 1.x and 2.x alike.
    # lap is the ByteTrack linear-assignment dependency; ultralytics tries
    # to auto-install it on first use, which PEP 668 blocks in the
    # container, so it is preinstalled explicitly here.
    NUMPY_PIN=$(python3 -c "import numpy; print(numpy.__version__)")
    pip3 install --break-system-packages -q "numpy==${NUMPY_PIN}" \
      ultralytics lap matplotlib seaborn scikit-learn joblib pyyaml tqdm
    source /opt/ros/rolling/setup.bash

    # riskam_bringup is excluded: its CMake requires realsense2_camera (a
    # camera-launch runtime dependency) that the smoke test does not need —
    # the node is started directly via ros2 run.
    echo "── colcon build (ROS packages; riskam library used from source) ──"
    cd /tmp
    colcon build --base-paths /workspace \
      --packages-select riskam_msgs riskam_ros \
      > /tmp/build.log 2>&1 || { tail -30 /tmp/build.log; exit 1; }
    source /tmp/install/setup.bash
    export PYTHONPATH="/workspace:${PYTHONPATH:-}"

    echo "── starting riskam_node ──"
    ros2 run riskam_ros riskam_node.py --ros-args \
      -p camera_topic:=/smoke/rgb \
      -p depth_topic:=/smoke/depth \
      -p camera_info_topic:=/smoke/camera_info \
      -p visualize_image:=false \
      > /tmp/node.log 2>&1 &
    NODE_PID=$!
    for i in $(seq 1 18); do
      sleep 5
      if ! kill -0 $NODE_PID 2>/dev/null; then
        echo "── node died during startup ──"; cat /tmp/node.log; exit 1
      fi
      grep -q "RiskAM node started" /tmp/node.log && break
    done
    if ! grep -q "RiskAM node started" /tmp/node.log; then
      echo "── node did not report startup in 90 s ──"; cat /tmp/node.log; exit 1
    fi

    echo "── feeding frames ──"
    python3 /workspace/scripts/smoke_feed_frames.py --n 80 --hz 2 \
      > /tmp/feed.log 2>&1 &

    SCORE=$(timeout 90 ros2 topic echo /riskam/risk_score --once 2>/dev/null || true)
    KIN=$(timeout 90 ros2 topic echo /riskam/risk_kinematic --once 2>/dev/null || true)
    DIAG=$(timeout 30 ros2 topic echo /riskam/diagnostics --once 2>/dev/null || true)

    kill $NODE_PID 2>/dev/null || true
    echo "── node log (tail) ──"; tail -12 /tmp/node.log || true
    echo "── feeder log (tail) ──"; tail -3 /tmp/feed.log || true
    echo "── /riskam/risk_score ──"; echo "$SCORE" | grep -E "score:" || echo "(none)"
    echo "── /riskam/risk_kinematic ──"; echo "$KIN" | grep -E "score:" || echo "(none)"
    echo "── diagnostics kinematic keys ──"
    echo "$DIAG" | grep -B1 -A1 kinematic || echo "(none)"

    # Assert on actual message payloads — ros2 topic echo prints discovery
    # warnings to stdout, so non-emptiness alone is not evidence.
    if echo "$SCORE" | grep -q "score:" && echo "$KIN" | grep -q "score:"; then
      echo SMOKE_PASS
    else
      echo SMOKE_FAIL
      exit 1
    fi
  '
