# CoreSense Risk Awareness Module (RiskAM)

[CoreSense](https://coresense.eu/) is a Horizon Europe-funded project that aims to develop a theory and a derived cognitive architecture for understanding in autonomous robots. A key ingredient needed to move towards true open-world autonomy in robotics is *risk awareness*: instead of modelling all fathomable risks beforehand (which is infeasible in open-world scenarios), we make the robot itself aware of risks. This is the aim of the **CoreSense Risk Awareness Module (RiskAM)**. You can read the full motivation, description, and prototype documentation in [CoreSense deliverable D3.5](http://zahalka.net/wp-content/uploads/2025/04/CoreSense___CS_067_D3_5__RiskAM_deliverable.pdf).

RiskAM operates on *visual* navigation and considers *risks to humans*. A live deployment with the SamXL robotics partner exposed several limitations of the original prototype; the current code base reflects an extensive set of corrections and additions tracked in [`docs/improvement_plan.md`](docs/improvement_plan.md). A summary of the current architecture:

| Sub-score | Inputs | Algorithm |
|-----------|--------|-----------|
| **proximity** | RGB + absolute depth (RealSense in metres) | linear falloff from `1 - d/d_safe` |
| **gaze** | RGB → YOLO11-Pose keypoints | 2-D head pose (yaw + pitch Gaussian); pre-T1.3 eye-symmetry baseline retained for the ablation study |
| **x_offset** | RGB + (optional) `cmd_vel` | path-aware projection when the robot is moving; centre-offset fallback when stationary |
| **approach** | tracked depth history per ByteTrack ID | per-track linear fit of depth(t) → ±1 m/s clamped to [0, 1] |

The aggregated scene risk is `max(per-person risk) × (1 + α · log(1 + n_extra_persons))` with a configurable crowd-penalty coefficient.

You can find [full demo videos from the CoreSense RoboCup @ Home 2023 dataset here](https://drive.google.com/drive/folders/1y_I-fNZk9aPJJtIgrDVrzYYbc89Gha_P?usp=sharing).

---

## Hard requirements

- **Inputs.** RGB + absolute depth in metres. The pipeline raises a `ValueError` at `FrameInputs.validate()` if either is missing — silent zeroing of the dominant proximity sub-score is a safety hazard, not a degradation mode. Depth is near-commodity in mobile robotics (RealSense, Azure Kinect, ZED, Orbbec) so this is not a meaningful platform restriction.
- **Supported depth sensors.** Defaults are calibrated for active-stereo IR sensors with clean close-range data (RealSense D4xx family). Sensors with a hard near-clip dead zone (PrimeSense Xtion / Carmine) are also supported via `depth_near_clip_m` — see the parameter table below and `docs/improvement_plan.md` §3.4 for the rationale.
- **Optional inputs.** `cmd_vel` (`geometry_msgs/Twist`) enables path-aware x_offset; absent → centre-offset fallback (status reported as `FALLBACK` on the diagnostics topic). ByteTrack-tracked frame continuity enables the approach sub-score; absent → status `UNAVAILABLE`.
- **ROS 2.** Currently rolling-tested. The module is ROS 2-distro-agnostic but we reserve the right to change this in line with CoreSense project specs.
- **Models.** Ultralytics YOLO11n-Pose (`ml_models/yolo11n-pose.pt`). ByteTrack is bundled with Ultralytics; no extra dependency.

---

## Installation and build

Clone into the source folder of your colcon workspace:

```bash
cd src   # assuming you're at the workspace root
git clone git@github.com:CoreSenseEU/risk-awareness-module.git
```

Build with colcon:

```bash
cd ..
colcon build --packages-select riskam riskam_ros riskam_msgs riskam_bringup
```

For the Python-only side (offline experiments, evaluation framework), set up the uv-managed environment as described in [Installation & prerequisites](#installation--prerequisites) below. The two paths are independent: ROS deployment uses `riskam/setup.py` via colcon and is unaffected by `uv`.

---

## Running the node

Run the node directly:

```bash
ros2 run riskam_ros riskam_node.py --ros-args -p camera_topic:=/your/color/topic
```

Or launch the node + bagger:

```bash
ros2 launch riskam_bringup riskam.launch.py run_logger:=true
```

Parameters live in `riskam_bringup/config/riskam_config.yml`; override at launch time via `--ros-args -p name:=value`.

---

## Parameters

The values below are the source-of-truth defaults from `riskam_bringup/config/riskam_config.yml`.

### `riskam_node`

#### Topics

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `camera_topic` | `/camera/camera/color/image_raw` | RGB input (`sensor_msgs/Image`, BGR8) |
| `depth_topic` | `/camera/camera/depth/image_rect_raw` | Depth input (`sensor_msgs/Image`, uint16 mm); converted to float32 metres internally |
| `cmd_vel_topic` | `/cmd_vel` | Robot instantaneous velocity (`geometry_msgs/Twist`); optional |

#### Risk-score weights (must sum to 1.0)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `w_proximity` | `0.7` | Weight of the depth-calibrated proximity sub-score |
| `w_gaze` | `0.25` | Weight of the 2-D head-pose gaze sub-score (inverted: higher gaze = lower risk) |
| `w_position` | `0.05` | Weight of the x_offset sub-score (path-aware when `cmd_vel` is available, else centre-offset) |
| `w_approach` | `0.0` | Weight of the per-track approach sub-score. Default `0.0` keeps live behaviour identical to the pre-T3.3.2 calibration; the eval framework's ablation sweep exercises non-zero values, and a calibrated default will land once that produces evidence |

#### Proximity / depth

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `d_safe` | `1.5` (metres) | Safety distance. Persons at or beyond `d_safe` score 0 on proximity; the score rises linearly to 1 at the camera. Tune to your robot's stopping distance + a margin |
| `depth_near_clip_m` | `0.0` (metres) | Sensor near-clip dead zone (T2.7). `0` → off — RealSense behaviour: a bbox with all-zero depth falls back to "far" (proximity 0). For sensors with a hard near-clip (e.g. Xtion/Carmine ≈ 0.6 m) set this to the spec-sheet value; RiskAM will then treat zero-depth in a *large* bbox as "person too close to measure" and emit max proximity instead of inverting the safety direction |
| `near_clip_bbox_min_frac` | `0.05` | Minimum bbox area as a fraction of the full frame for the close-fallback to fire. Ignored when `depth_near_clip_m == 0`. Guards against tiny noise bboxes triggering max risk |
| `near_clip_valid_frac_max` | `0.0` | Maximum valid-pixel fraction within a bbox for the close-fallback to fire. `0.0` → strict zero (RealSense pre-T2.7 contract). Set to a small positive value (e.g. `0.05` for Xtion) to also fire on "mostly empty" bboxes whose few valid pixels are likely background bleed-through, not the person |

#### Gaze (2-D head pose)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `gaze_sigma_yaw` | `0.3` | Gaussian σ for yaw deviation, normalised by inter-eye distance |
| `gaze_sigma_pitch` | `0.5` | Gaussian σ for pitch deviation from the expected frontal nose-eye ratio |
| `gaze_frontal_pitch_ratio` | `0.7` | Expected `(nose.y - eye_mid.y) / inter_eye_dist` for a frontal face |

#### Temporal aggregation

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `n_frames_aggregate` | `5` | Number of frames in each track's risk-history window |
| `crowd_alpha` | `0.1` | Crowd-penalty coefficient α: `scene_risk = max(per_person) × (1 + α·log(1 + n_extra))` |

#### Other

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `visualize_image` | `true` | Publish annotated visualisation on `/riskam/annotated_image` |
| `sync_slop` | `0.1` (seconds) | Tolerance for the color/depth message-filter sync |

### `riskam_bagger`

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `bag_folder` | `bags` | Output directory |
| `bag_name` | `riskam_test` | Output filename stem |
| `tag_with_time` | `true` | Append timestamp to filename |
| `log_images` | `true` | Record `topic_camera_image` |
| `log_annotated_images` | `true` | Record `topic_annotated_image` |
| `log_subscores` | `true` | Record gaze/depth/x_pose/approach sub-score topics |
| `topic_camera_image` | `/camera/camera/color/image_raw` | RGB input topic (must match the node) |
| `topic_annotated_image` | `/riskam/annotated_image` | Annotated-image topic published by the node |
| `topic_risk_score` | `/riskam/risk_score` | Always logged regardless of `log_subscores` |
| `topic_gaze` | `/riskam/gaze` | Per-frame max gaze sub-score |
| `topic_depth` | `/riskam/depth` | Per-frame max proximity sub-score |
| `topic_x_pose` | `/riskam/x_pose` | Per-frame max x_offset sub-score |
| `topic_approach` | `/riskam/approach` | Per-frame max approach sub-score |

---

## Published topics

| Topic | Type | Notes |
|-------|------|-------|
| `/riskam/risk_score` | `riskam_msgs/FloatStamped` | Aggregated scene risk, in [0, 1] |
| `/riskam/depth` | `riskam_msgs/FloatStamped` | Per-frame max proximity sub-score |
| `/riskam/gaze` | `riskam_msgs/FloatStamped` | Per-frame max gaze sub-score |
| `/riskam/x_pose` | `riskam_msgs/FloatStamped` | Per-frame max x_offset sub-score |
| `/riskam/approach` | `riskam_msgs/FloatStamped` | Per-frame max approach sub-score |
| `/riskam/annotated_image` | `sensor_msgs/Image` | Visualisation overlay (when `visualize_image: true`) |
| `/riskam/diagnostics` | `diagnostic_msgs/DiagnosticArray` | Per-frame timing, person/track count, **per-sub-score status** (`active` / `fallback` / `unavailable`) and reason. Diagnostic level escalates to `WARN` whenever any sub-score is non-`active`, so degraded modes are visible on the bus |

---

## For deployers

If you are integrating RiskAM onto a robot and want a working risk score on Day 1, this is the short version:

1. **You need RGB + depth.** Wire up your sensor and point `camera_topic` / `depth_topic` at it.
2. **Pick or derive your platform calibration.** RiskAM ships with named factory presets for the hardware it has been validated on, defined in [`riskam/platforms.py`](riskam/platforms.py). If yours matches, copy the values straight into `riskam_config.yml`; otherwise derive them from the sensor spec sheet and your robot's stopping distance.

   | Platform preset | `d_safe` | `depth_near_clip_m` | `near_clip_valid_frac_max` | Hardware |
   |-----------------|----------|---------------------|----------------------------|----------|
   | `RIDGEBACK_D435` (SamXL — the shipped defaults) | `1.5` | `0.0` | `0.0` | Clearpath Ridgeback (~70 kg, ~1 m/s indoor) + Intel RealSense D4xx |
   | `TIAGO_XTION` (cs_robocup_2023) | `2.5` | `0.6` | `0.05` | PAL TIAGo (~70 kg, ~1 m/s indoor) + PAL Xtion / PrimeSense Carmine |

   **Custom platform** — derive from physics:
   - `d_safe` ≈ stopping distance at max speed + safety margin. ~1.5 m for slow indoor mobile (~1 m/s); larger for faster or heavier robots.
   - `depth_near_clip_m` from the sensor spec sheet. `0` for active-stereo (RealSense, Azure Kinect, ZED) — clean close-range data, "no measurement in bbox = person far away" is correct. The published near-clip for structured-light sensors (Xtion/Carmine ≈ 0.6 m, Astra ≈ 0.6 m) — RiskAM will then treat "all-zero depth in a person-sized bbox" as "person too close to measure" instead of inverting the safety direction.
   - `near_clip_valid_frac_max`: `0` if your sensor delivers clean data; ~`0.05` for structured-light sensors that produce noisy zero-fill plus background bleed-through. If unsure, leave at `0` and bump only if you see "person clearly close, but proximity stays zero" patterns on `/riskam/diagnostics`.

   If you bring up a new platform you'd like RiskAM to ship a preset for, contributions to `riskam/platforms.py` are welcome — keep the dataclass, document the values, and add a sanity test in `tests/test_platforms.py`.

3. **Optional: wire `/cmd_vel`.** If your robot publishes velocity, the x_offset sub-score becomes path-aware (Gaussian around the robot's lane of motion). Without it, x_offset uses a centre-offset heuristic and reports `FALLBACK` on diagnostics.
4. **Watch `/riskam/diagnostics`.** Any sub-score reporting `unavailable` or `fallback` should be expected (e.g. `cmd_vel` not subscribed) — if it isn't expected, something upstream is misconfigured.
5. **You should not need to run a hyperparameter sweep.** The shipped weights are calibrated against the cs_robocup_2023 dataset; the research-methodology tooling in the next section is for the RiskAM authors when publishing, not for you.

---

## For researchers: evaluation framework

RiskAM ships with an evaluation framework for offline benchmarking against annotated datasets. The pieces:

| Tool | Purpose |
|------|---------|
| `scripts/extract_ros2_dataset.py cs_robocup_2023` | Extract RGB + depth `.npy` frames from the source ROS 2 bags |
| `scripts/generate_split.py` | Write the canonical val/test split JSON (stratified within-run by class) |
| `scripts/run_experiments.py run cs_robocup_2023 --run RB_XX --split val [--sweep-config configs/sweeps/foo.yaml] [--no-cache]` | Run a parameter sweep on one run; sweep grid is YAML-defined; default at `configs/sweeps/default.yaml`, ablation at `configs/sweeps/ablation_gaze.yaml` |
| `scripts/summarize_experiments.py cs_robocup_2023 --bucket val --rank-metric macro_f1` | Aggregate per-experiment results across runs and rank configs |
| `scripts/reeval_experiments.py cs_robocup_2023 --bucket val` | Re-score cached `raw_predictions.json` under updated metric definitions without re-running inference |

Each experiment's `results.json` is self-describing: it stamps git SHA, hostname, package versions, sweep params, split metadata, and feature-cache hit/miss stats alongside the classification report (per-class P/R/F1, confusion matrix, MAE/RMSE).

The full design rationale — sub-score input contract, why no train split, why stratified-within-run was chosen over whole-run holdout, why MiDaS was ruled out as a baseline — lives in [`docs/improvement_plan.md`](docs/improvement_plan.md) §4.9 / §4.10 / §3 and the per-item entries.

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
    cv_image,
    result.human_bboxes,
    result.depth_viz,
    result.features,
    risk_score,
    max_risk_idx,
)
```

`result.subscore_status` and `result.subscore_reasons` carry the same per-sub-score status info that the ROS node publishes on `/riskam/diagnostics`.

---

## Installation & prerequisites

Tested on Ubuntu 24.04 with ROS 2 rolling. [Installation guide for `rolling`](https://docs.ros.org/en/rolling/Installation.html).

The Python-only workflow (offline experiments, evaluation framework, tests) is managed with [uv](https://docs.astral.sh/uv/). Dependencies are declared in `pyproject.toml` and locked in `uv.lock`. The ROS build path is independent: `colcon build` uses `riskam/setup.py` and ignores `pyproject.toml` entirely.

Install uv (if not already present):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Sync the environment (creates `.venv/`, installs runtime + `dev` deps from the lockfile):

```bash
cd <riskam_root_dir>
uv sync
```

Run any command in the env via `uv run` (no manual `activate` needed):

```bash
uv run pytest
uv run python scripts/run_experiments.py run cs_robocup_2023 --run RB_02
```

Link the system's `rosbag2_py` into the uv venv (needed by `scripts/extract_ros2_dataset.py`):

```bash
echo "/opt/ros/<ros2_distro>/lib/python3.x/site-packages" \
  > <riskam_root_dir>/.venv/lib/python3.x/site-packages/ros2.pth
```

Source ROS 2 before extracting data from ROS 2 bags:

```bash
source /opt/ros/<ros2_distro>/setup.bash
```

### Demo dataset download

The usage walkthroughs use the CoreSense RoboCup @ Home 2023 dataset ([download](https://zenodo.org/records/13902513)). Download `2023_Dataset.zip` and unzip it.

### Image + depth extraction from ROS 2 bags

Both RGB and depth are extracted (depth is required by the supported pipeline). Create the dataset directory:

```bash
cd <riskam_root_dir>
mkdir -p ros_datasets/cs_robocup_2023
```

For each populated run (`RB_01`–`RB_03`, `RB_05`, `RB_06`, `RB_08`; `RB_04` and `RB_07` are empty), unzip the respective ZIP file:

```bash
unzip <download_dir>/2023_Dataset/Dataset_v2/RB_01/RB_01.zip \
      -d <riskam_root_dir>/ros_datasets/cs_robocup_2023/RB_01
```

For `RB_01` specifically, manually decompress the `.zst` bags first:

```bash
cd <riskam_root_dir>/ros_datasets/cs_robocup_2023/RB_01/mapeo1
unzstd *.zst
cd <riskam_root_dir>
```

Then extract:

```bash
uv run python scripts/extract_ros2_dataset.py cs_robocup_2023
```

On macOS (no native ROS 2), use the Docker wrapper instead:

```bash
scripts/extract_ros2_dataset_macos.sh cs_robocup_2023
```

It runs extraction inside a `ros:rolling-perception` container with the repo bind-mounted; output lands on the host at `ml_datasets/cs_robocup_2023/raw_dataset/`. First invocation pulls the image (~1.5 GB).

You can run extraction per `RB_##` if disk space is tight; the script processes whatever runs are present.

---

## Usage

### Demo video for a specific run

```bash
uv run python scripts/test_run_with_video.py cs_robocup_2023 RB_##
```

Outputs to `test_results/videos/`:
- `cs_robocup_2023_RB_##_raw.avi` — the actual footage the robot sees
- `cs_robocup_2023_RB_##_risk.avi` — the RiskAM output overlay, with:
  1. Color-coded scene risk in the top-right corner
  2. Bounding boxes for all detected humans
  3. Each bbox color-coded by gaze along a continuous red↔white gradient:
     - Red (gaze ≈ 0): pose undetectable, or person likely *not* aware of the robot
     - White (gaze ≈ 1): person clearly aware (looking at the camera)
     - Pinks in between for ambiguous gazes
     - A black outline is drawn underneath each bbox so it stays visible on red walls / shirts and on white / overexposed backgrounds
  4. The bbox driving the scene-risk maximum is marked with an asterisk
  5. Each bbox interior is tinted red with opacity proportional to its proximity sub-score (`α = proximity · 0.7`): red = close = danger, transparent = far = safe. Per-bbox uniform — driven by the proximity sub-score, not raw per-pixel depth — so Xtion-class sensor noise does not splotch the overlay. T2.7 close-fallback bboxes ("person too close to measure") render as full red panels. Outside bboxes, the original scene is untouched

### Experiments

Run a sweep on one run:

```bash
uv run python scripts/run_experiments.py run cs_robocup_2023 --run RB_##
```

Or all populated runs sequentially:

```bash
scripts/run_experiments_cs_robocup_2023_all.sh
```

Outputs land under `exp_results/<dataset>/{all,val,test}/<run>/<params_slug>/`. See [For researchers: evaluation framework](#for-researchers-evaluation-framework) for the full toolchain.

---

## Acknowledgment

This work has received funding from the European Union's Horizon Europe research and innovation programme under grant agreement No. 101070254 CORESENSE. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the Horizon Europe programme. Neither the European Union nor the granting authority can be held responsible for them.
