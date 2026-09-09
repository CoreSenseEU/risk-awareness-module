# ROS deployment

The deployer reference: building, running, and configuring the RiskAM node on a
robot. This is the canonical, exhaustive parameter/topic reference; the root
`README.md` carries a leaner quickstart subset.

See [`architecture.md`](architecture.md) for the node's threading model,
[`scoring.md`](scoring.md) for what the weights control, and
[`sensors-and-platforms.md`](sensors-and-platforms.md) for the depth knobs and
platform presets.

---

## Stack

Conservative stance: RiskAM must work with the rest of the CoreSense stack. No
ROS version changes are proposed.

- **MiDaS removed** — `timm` / `torch.hub` MiDaS gone; one fewer inference model.
- **`numpy<2.0.0` pin lifted** — was imposed only by MiDaS.
- **MediaPipe removed** — old face-mesh gaze module deleted.
- **Ultralytics YOLO11n-Pose + ByteTrack** — single inference model for the full
  pipeline (`ml_models/yolo11n-pose.pt`); ByteTrack bundled, no extra dependency.
- **ROS 2 Rolling** — rolling-tested; distro-agnostic in principle.
- **Depth sensors** — RealSense D4xx default-calibrated; structured-light
  sensors supported via the near-clip fallback.
- **Known dead dep** — `nav_msgs` is declared in `riskam_ros/package.xml` but the
  path-aware `x_offset` actually uses `geometry_msgs/Twist` from `/cmd_vel`; drop
  `nav_msgs` when convenient.

---

## Build

Clone into the `src/` of your colcon workspace and build:

```bash
cd src
git clone git@github.com:CoreSenseEU/risk-awareness-module.git
cd ..
colcon build --packages-select riskam riskam_ros riskam_msgs riskam_bringup
```

The ROS build path uses `riskam/setup.py` via colcon and is independent of the
`uv`/`pyproject.toml` Python-only path used by the offline tooling.

---

## Run

Node directly:

```bash
ros2 run riskam_ros riskam_node.py --ros-args -p camera_topic:=/your/color/topic
```

Node + bagger via launch:

```bash
ros2 launch riskam_bringup riskam.launch.py run_logger:=true
```

Parameters live in `riskam_bringup/config/riskam_config.yml`; override at launch
with `--ros-args -p name:=value`. The values below are the source-of-truth
defaults from that file.

---

## `riskam_node` parameters

### Topics

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `camera_topic` | `/camera/camera/color/image_raw` | RGB input (`sensor_msgs/Image`, BGR8) |
| `depth_topic` | `/camera/camera/depth/image_rect_raw` | Depth input (`sensor_msgs/Image`, uint16 mm); converted to float32 metres internally |
| `cmd_vel_topic` | `/cmd_vel` | Robot instantaneous velocity (`geometry_msgs/Twist`); optional |

### Risk-score weights (must sum to 1.0)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `w_proximity` | `0.7` | Depth-calibrated proximity sub-score |
| `w_gaze` | `0.25` | 2-D head-pose gaze sub-score (inverted: higher gaze = lower risk) |
| `w_position` | `0.05` | x_offset sub-score (path-aware with `cmd_vel`, else centre-offset) |
| `w_approach` | `0.0` | Per-track approach sub-score. `0.0` keeps live behaviour identical to the pre-calibration baseline; the ablation sweep exercises non-zero values |

### Proximity / depth

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `d_safe` | `1.5` m | Safety distance. Persons at/beyond `d_safe` score 0; rises linearly to 1 at the camera. Tune to stopping distance + margin |
| `depth_near_clip_m` | `0.0` m | Sensor near-clip dead zone. `0` → RealSense behaviour. See [`sensors-and-platforms.md`](sensors-and-platforms.md) |
| `near_clip_bbox_min_frac` | `0.05` | Minimum bbox area fraction for the close-fallback to fire |
| `near_clip_valid_frac_max` | `0.0` | Maximum valid-pixel fraction for the close-fallback to fire |

### Gaze (2-D head pose)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `gaze_sigma_yaw` | `0.3` | Gaussian σ for yaw deviation, normalised by inter-eye distance |
| `gaze_sigma_pitch` | `0.5` | Gaussian σ for pitch deviation from the frontal nose-eye ratio |
| `gaze_frontal_pitch_ratio` | `0.7` | Expected `(nose.y − eye_mid.y) / inter_eye_dist` for a frontal face |

### Temporal aggregation

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `n_frames_aggregate` | `5` | Frames in each track's risk-history window |
| `crowd_alpha` | `0.1` | Crowd-penalty α: `scene_risk = max(per_person)·(1 + α·log(1 + n_extra))` |

### Kinematic companion channel

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `publish_kinematic` | `true` | Compute and publish the awareness-modulated kinematic formulation alongside the weighted score (`/riskam/risk_kinematic`). The weighted path is unaffected |
| `camera_info_topic` | `""` | Intrinsics source; empty → the `camera_info` sibling of `camera_topic` (first message wins) |
| `camera_hfov_deg` | `0.0` | Pinhole fallback when no `camera_info` is available (e.g. `69.0` D4xx, `58.0` Xtion); `0` → wait for `camera_info` |
| `kinematic_tau_s` | `2.0` s | Reaction window τ of the trajectory hazard |
| `kinematic_beta` | `1.0` | Oblivious-human penalty β in `hazard · (1 + β(1 − awareness))` |
| `footprint_radius_m` | `0.0` m | Robot footprint radius; miss distance is measured from the robot surface |

### Other

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `visualize_image` | `true` | Publish annotated overlay on `/riskam/annotated_image` |
| `sync_slop` | `0.1` s | Tolerance for the color/depth message-filter sync |

---

## `riskam_bagger` parameters

All topic strings are node parameters (no longer hardcoded) and are mirrored in
`riskam_config.yml`.

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `bag_folder` | `bags` | Output directory |
| `bag_name` | `riskam_test` | Output filename stem |
| `tag_with_time` | `true` | Append timestamp to filename |
| `log_images` | `true` | Record `topic_camera_image` |
| `log_annotated_images` | `true` | Record `topic_annotated_image` |
| `log_subscores` | `true` | Record gaze/depth/x_pose/approach sub-score topics |
| `topic_camera_image` | `/camera/camera/color/image_raw` | RGB input (must match the node) |
| `topic_annotated_image` | `/riskam/annotated_image` | Annotated image from the node |
| `topic_risk_score` | `/riskam/risk_score` | Always logged regardless of `log_subscores` |
| `topic_gaze` | `/riskam/gaze` | Per-frame max gaze sub-score |
| `topic_depth` | `/riskam/depth` | Per-frame max proximity sub-score |
| `topic_x_pose` | `/riskam/x_pose` | Per-frame max x_offset sub-score |
| `topic_approach` | `/riskam/approach` | Per-frame max approach sub-score |

---

## Published topics

| Topic | Type | Notes |
|-------|------|-------|
| `/riskam/risk_score` | `riskam_msgs/FloatStamped` | Aggregated scene risk in [0, 1] (weighted formulation, the field-validated default) |
| `/riskam/risk_kinematic` | `riskam_msgs/FloatStamped` | Awareness-modulated kinematic scene risk in [0, 1]; silent until camera intrinsics are available |
| `/riskam/depth` | `riskam_msgs/FloatStamped` | Per-frame max proximity sub-score |
| `/riskam/gaze` | `riskam_msgs/FloatStamped` | Per-frame max gaze sub-score |
| `/riskam/x_pose` | `riskam_msgs/FloatStamped` | Per-frame max x_offset sub-score |
| `/riskam/approach` | `riskam_msgs/FloatStamped` | Per-frame max approach sub-score |
| `/riskam/annotated_image` | `sensor_msgs/Image` | Visualisation overlay (when `visualize_image: true`) |
| `/riskam/diagnostics` | `diagnostic_msgs/DiagnosticArray` | Per-frame timing, person/track count, **per-sub-score status** (`active`/`fallback`/`unavailable`) + reason. Level escalates to `WARN` whenever any sub-score is non-`active` |

---

## Deployment checklist

1. **You need RGB + depth.** Point `camera_topic` / `depth_topic` at your sensor.
   The pipeline raises `ValueError` if depth is missing — this is intentional
   (see [`design-rationale.md`](design-rationale.md), T3.2).
2. **Pick or derive your platform calibration** — `d_safe`, `depth_near_clip_m`,
   `near_clip_valid_frac_max`. Use a shipped preset if your hardware matches,
   else derive from the sensor spec sheet + robot stopping distance. See
   [`sensors-and-platforms.md`](sensors-and-platforms.md).
3. **Optional: wire `/cmd_vel`** to make `x_offset` path-aware. Without it,
   x_offset uses the centre-offset fallback and reports `FALLBACK` on diagnostics.
4. **Watch `/riskam/diagnostics`.** Expected `fallback`/`unavailable` statuses
   (e.g. no `cmd_vel`) are fine; unexpected ones mean something upstream is
   misconfigured.
5. **No hyperparameter sweep needed for deployment.** Shipped weights are
   calibrated against cs_robocup_2023; the sweep tooling is research methodology
   (see [`evaluation-framework.md`](evaluation-framework.md)), not a deployment step.
