# CoreSense Risk Awareness Module (RiskAM)

[CoreSense](https://coresense.eu/) is a Horizon Europe-funded project that aims to develop a theory and a derived cognitive architecture for understanding in autonomous robots. A key ingredient needed to move towards true open-world autonomy in robotics is *risk awareness*: instead of modelling all fathomable risks beforehand (which is infeasible in open-world scenarios), we make the robot itself aware of risks. This is the aim of the **CoreSense Risk Awareness Module (RiskAM)**. You can read the full motivation, description, and prototype documentation in [CoreSense deliverable D3.5](http://zahalka.net/wp-content/uploads/2025/04/CoreSense___CS_067_D3_5__RiskAM_deliverable.pdf).

RiskAM operates on *visual* navigation and considers *risks to humans*. A live deployment with the SamXL robotics partner exposed several limitations of the original prototype; the current code base reflects an extensive set of corrections and additions. A summary of the current architecture:

| Sub-score | Inputs | Algorithm |
|-----------|--------|-----------|
| **proximity** | RGB + absolute depth (in metres) | linear falloff `1 - d/d_safe` |
| **gaze** | RGB → YOLO11-Pose keypoints | 2-D head pose (yaw + pitch Gaussian) |
| **x_offset** | RGB + (optional) `cmd_vel` | path-aware projection when moving; centre-offset fallback when stationary |
| **approach** | tracked depth history per ByteTrack ID | per-track linear fit of depth(t) |

Scene risk is `max(per-person risk) × (1 + α · log(1 + n_extra_persons))`. See [`docs/scoring.md`](docs/scoring.md) for the full algorithm.

[Full demo videos from the CoreSense RoboCup @ Home 2023 dataset](https://drive.google.com/drive/folders/1y_I-fNZk9aPJJtIgrDVrzYYbc89Gha_P?usp=sharing).

## Documentation

Full reference docs live in [`docs/`](docs/) ([index](docs/README.md)):

- [architecture.md](docs/architecture.md) — pipeline + sub-score input contract
- [scoring.md](docs/scoring.md) — the sub-score math
- [sensors-and-platforms.md](docs/sensors-and-platforms.md) — depth handling + platform presets
- [ros-deployment.md](docs/ros-deployment.md) — **full parameter / topic reference**
- [evaluation-framework.md](docs/evaluation-framework.md) — offline benchmarking toolchain
- [visualization.md](docs/visualization.md) — overlay semantics
- [design-rationale.md](docs/design-rationale.md) — why the redesign happened
- [improvement-plan.md](docs/improvement-plan.md) — outstanding work

---

## Hard requirements

- **RGB + absolute depth in metres.** The pipeline raises `ValueError` if either is missing — silent zeroing of the dominant proximity sub-score is a safety hazard, not a degradation mode. Depth is near-commodity in mobile robotics (RealSense, Azure Kinect, ZED, Orbbec).
- **Supported depth sensors.** Defaults are calibrated for active-stereo IR sensors with clean close-range data (RealSense D4xx). Sensors with a hard near-clip dead zone (PrimeSense Xtion / Carmine) are supported via `depth_near_clip_m` — see [`docs/sensors-and-platforms.md`](docs/sensors-and-platforms.md).
- **Optional inputs.** `cmd_vel` (`geometry_msgs/Twist`) enables path-aware x_offset; ByteTrack track continuity enables the approach sub-score. Absent inputs report `FALLBACK` / `UNAVAILABLE` on the diagnostics topic.
- **ROS 2.** Rolling-tested; distro-agnostic in principle.
- **Models.** Ultralytics YOLO11n-Pose (`ml_models/yolo11n-pose.pt`); ByteTrack bundled, no extra dependency.

---

## Install and build

Clone into the source folder of your colcon workspace and build:

```bash
cd src   # workspace root/src
git clone git@github.com:CoreSenseEU/risk-awareness-module.git
cd ..
colcon build --packages-select riskam riskam_ros riskam_msgs riskam_bringup
```

The ROS build path (colcon → `riskam/setup.py`) is independent of the Python-only path used for offline experiments (see [Python environment](#python-environment-offline-tooling)).

## Run the node

```bash
ros2 run riskam_ros riskam_node.py --ros-args -p camera_topic:=/your/color/topic
```

Or launch node + bagger:

```bash
ros2 launch riskam_bringup riskam.launch.py run_logger:=true
```

Parameters live in `riskam_bringup/config/riskam_config.yml`; override at launch via `--ros-args -p name:=value`.

## Essential parameters

The defaults below are enough to get a working score; the **complete** parameter, bagger, and topic reference is in [`docs/ros-deployment.md`](docs/ros-deployment.md).

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `camera_topic` | `/camera/camera/color/image_raw` | RGB input (`sensor_msgs/Image`, BGR8) |
| `depth_topic` | `/camera/camera/depth/image_rect_raw` | Depth input (uint16 mm → float32 m) |
| `cmd_vel_topic` | `/cmd_vel` | Robot velocity (optional; enables path-aware x_offset) |
| `w_proximity` / `w_gaze` / `w_position` / `w_approach` | `0.7` / `0.25` / `0.05` / `0.0` | Sub-score weights (must sum to 1.0) |
| `d_safe` | `1.5` m | Safety distance: proximity is 0 at/beyond `d_safe`, 1 at the camera. Tune to stopping distance + margin |
| `depth_near_clip_m` | `0.0` m | Sensor near-clip dead zone; `0` = RealSense. Set per sensor for structured-light cameras |

## Published topics

| Topic | Type | Notes |
|-------|------|-------|
| `/riskam/risk_score` | `riskam_msgs/FloatStamped` | Aggregated scene risk in [0, 1] |
| `/riskam/depth` · `/gaze` · `/x_pose` · `/approach` | `riskam_msgs/FloatStamped` | Per-frame max of each sub-score |
| `/riskam/annotated_image` | `sensor_msgs/Image` | Visualisation overlay ([read it here](docs/visualization.md)) |
| `/riskam/diagnostics` | `diagnostic_msgs/DiagnosticArray` | Timing, person/track count, per-sub-score status (`active`/`fallback`/`unavailable`); level escalates to `WARN` on any non-`active` sub-score |

---

## For deployers

The short version (full checklist in [`docs/ros-deployment.md`](docs/ros-deployment.md)):

1. **Wire RGB + depth** to `camera_topic` / `depth_topic`.
2. **Pick or derive your platform calibration** (`d_safe`, `depth_near_clip_m`, `near_clip_valid_frac_max`). Use a shipped preset from [`riskam/platforms.py`](riskam/platforms.py) if your hardware matches (`RIDGEBACK_D435`, `TIAGO_XTION`), else derive from the sensor spec sheet + stopping distance — see [`docs/sensors-and-platforms.md`](docs/sensors-and-platforms.md).
3. **Optionally wire `/cmd_vel`** to make x_offset path-aware.
4. **Watch `/riskam/diagnostics`** for unexpected `fallback` / `unavailable` statuses.
5. **No hyperparameter sweep needed** — shipped weights are calibrated; the sweep tooling is for research, not deployment.

## For researchers

RiskAM ships an offline evaluation framework (sweep, metrics, cross-run summary, eval-only re-scoring, feature cache, val/test split, provenance, gaze ablation). The full toolchain and dataset preparation are documented in [`docs/evaluation-framework.md`](docs/evaluation-framework.md); the methodology and decisions in [`docs/design-rationale.md`](docs/design-rationale.md).

Quick demo video for one run:

```bash
uv run python scripts/test_run_with_video.py cs_robocup_2023 RB_##
```

Parameter sweep:

```bash
uv run python scripts/run_experiments.py run cs_robocup_2023 --run RB_##
# or all populated runs:
scripts/run_experiments_cs_robocup_2023_all.sh
```

Outputs land under `exp_results/<dataset>/{all,val,test}/<run>/<params_slug>/`.

Experimental metrics (kinematic hazard × awareness + ordinal calibration —
see [`docs/experimental-metrics.md`](docs/experimental-metrics.md)):

```bash
uv run python scripts/metric_lab.py populate --device mps   # one-time YOLO pass
uv run --group experiments python scripts/metric_lab.py all # table → fit → evaluate
```

---

## Using RiskAM in a Python file

For non-ROS use (offline analysis, notebooks):

```python
import cv2
import numpy as np

from riskam.ml import featextr
from riskam.ml.subscores import FrameInputs, RobotVelocity
from riskam.score import RiskScorer
from riskam import visualization as vis

scorer = RiskScorer()

cv_image = cv2.imread("frame_001.png")              # BGR
depth_m = np.load("frame_001_depth_m.npy")          # float32, metres

result = featextr.extract(
    FrameInputs(
        rgb=cv_image,
        depth_m=depth_m,
        cmd_vel=RobotVelocity(linear_x=0.3, linear_y=0.0),  # or None
    ),
)

risk_score, max_risk_idx, per_person = scorer.score(
    result.features,
    track_ids=result.track_ids,
)

annotated = vis.visualize_risk(
    cv_image, result.human_bboxes, result.features, risk_score, max_risk_idx,
)
```

`result.subscore_status` / `result.subscore_reasons` carry the same per-sub-score status the ROS node publishes on `/riskam/diagnostics`. See [`docs/architecture.md`](docs/architecture.md) for the contract.

---

## Python environment (offline tooling)

Tested on Ubuntu 24.04 with ROS 2 rolling. The Python-only workflow (offline experiments, evaluation framework, tests) is managed with [uv](https://docs.astral.sh/uv/); dependencies are declared in `pyproject.toml` and locked in `uv.lock`. The ROS build path is independent.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is not present
cd <riskam_root_dir>
uv sync                                            # creates .venv/ from the lockfile
uv run pytest                                      # run anything via `uv run`
```

For ROS-bag extraction (`scripts/extract_ros2_dataset.py`), link the system `rosbag2_py` into the venv and source ROS 2 first:

```bash
echo "/opt/ros/<ros2_distro>/lib/python3.x/site-packages" \
  > <riskam_root_dir>/.venv/lib/python3.x/site-packages/ros2.pth
source /opt/ros/<ros2_distro>/setup.bash
```

Dataset download, extraction, and the full research workflow are documented in [`docs/evaluation-framework.md`](docs/evaluation-framework.md).

---

## Acknowledgment

This work has received funding from the European Union's Horizon Europe research and innovation programme under grant agreement No. 101070254 CORESENSE. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the Horizon Europe programme. Neither the European Union nor the granting authority can be held responsible for them.
