# Evaluation framework

The researcher reference: offline benchmarking of RiskAM against annotated
datasets. This is a **research-methodology toolchain for the RiskAM authors when
publishing**, not a deployment workflow — deployed RiskAM works out of the box
with calibrated defaults (see [`ros-deployment.md`](ros-deployment.md)).

The framework shares the live node's pipeline exactly: both go through
`featextr.extract(FrameInputs(...))` (see [`architecture.md`](architecture.md)).

---

## Toolchain

| Tool | Purpose |
|------|---------|
| `scripts/extract_ros2_dataset.py cs_robocup_2023` | Extract RGB + depth `.npy` frames from the source ROS 2 bags |
| `scripts/generate_split.py` | Write the canonical val/test split JSON (stratified within-run by class) |
| `scripts/run_experiments.py run <dataset> --run RB_XX --split val [--sweep-config …] [--no-cache]` | Run a parameter sweep on one run |
| `scripts/run_experiments_cs_robocup_2023_all.sh` | All populated runs sequentially |
| `scripts/summarize_experiments.py <dataset> --bucket val --rank-metric macro_f1 [--top-k N]` | Aggregate across runs and rank configs |
| `scripts/reeval_experiments.py <dataset> --bucket val` | Re-score cached predictions under updated metrics without re-running inference |

Output layout: `exp_results/<dataset>/{all,val,test}/<run>/<params_slug>/`.

Each experiment's `results.json` is self-describing — it stamps git SHA,
hostname, package versions, sweep params, split metadata, and feature-cache
hit/miss stats alongside the classification report.

---

## Components

### Offline depth — `riskam/data/cs_robocup_2023.py`

Per-frame absolute-depth `.npy` files (raw uint16 mm) are saved by
`extract_cs_robocup.py`. `CSRobocup2023DepthIndex` does nearest-neighbour
RGB↔depth lookup via binary search (0.2 s tolerance); `experiments.run_experiment`
passes `depth_image_m` into `featextr`. This re-enabled the proximity sub-score
offline — previously the sweep ran with `depth_image_m=None`, silently zeroing
the single most important sub-score.

### Config-driven sweep — `riskam/sweep_config.py`, `configs/sweeps/`

`SweepConfig.iter_experiments()` yields the Cartesian product of:

- **weight tuples** (coupled — each must sum to 1.0 ± 0.01)
- **independent axes**: `gaze_sigma_yaw`, `gaze_sigma_pitch`, `gaze_algorithm`

The hardcoded grid was removed from `experiments.py`. Two configs ship:
`configs/sweeps/default.yaml` (36 cells; `gaze_algorithm` defaults to
`head_pose` only) and `configs/sweeps/ablation_gaze.yaml` (72 cells; crosses both
gaze algorithms). Schema validation raises with the source path in the message on
missing keys, unknown keys, empty lists, or non-normalised weight tuples.
Extension axes (`d_safe`, `crowd_alpha`) are a trivial YAML addition once
ablation design dictates them.

### Metrics — `riskam/eval_metrics.py`

`classification_report(y_true, y_pred_continuous)` produces per-class
precision/recall/F1/support, a 4×4 confusion matrix, macro/micro averages (micro
≡ accuracy), and MAE/RMSE against class-centre targets. Merged into each
`results.json` under `classification`.

### Cross-run aggregation — `riskam/eval_summary.py`

Walks the per-experiment `results.json` files and pools metrics across runs per
config: confusion matrices summed cell-wise (per-class/macro/micro recomputed
from the pooled CM — statistically correct), MAE/RMSE sample-weighted. Emits
`summary.json` + a ranked top-K `summary.md` with a per-run breakdown of the best
config. Plots are deferred (not a blocker for the paper numbers).

### Eval-only mode — `riskam/reeval.py`

Given a cached `raw_predictions.json`, recomputes all metrics without re-running
YOLO/featextr. Stamps a fresh provenance block with `reeval_of` pointing at the
original inference-run git SHA so the audit chain is preserved. `--output-suffix`
writes alongside the original instead of overwriting.

### Feature cache — `riskam/feature_cache.py`

Caches `CachedFrameFeatures` (bboxes, keypoints, track_ids, per-bbox depths,
depth_viz) under `feature_cache/<dataset>/<run>/<model_sha>/<frame_stem>.npz`.
`run_experiment(use_cache=True)` by default; `--no-cache` disables. Turns the
inference-bound sweep from O(hours) to O(minutes).

> **Caveat.** ByteTrack's temporal coupling means the cache is most useful
> *after* a full uncached pass populates it — mixed hit/miss within a single run
> can desynchronise track IDs. Populate, then sweep.

### Provenance — `riskam/provenance.py`

`reproducibility_metadata()` stamps git SHA + dirty flag, hostname, ISO-8601 UTC
timestamp, Python version, and tracked package versions (torch, ultralytics,
numpy, opencv-python). Merged into `results.json` under `provenance`; params are
mirrored under `params` so each result is self-describing.

### Val/test split — `riskam/data/splits.py`

**Note: there is no train split.** RiskAM has no trainable parameters, so the
sweep does *hyperparameter selection* on val and the final report comes from a
single pass on test.

Stratified **within-run** by class (seed 42, default `test_fraction=0.3`).
Whole-run holdout was rejected for cs_robocup_2023 (RB_07 empty, RB_05
single-class, classes 0/1 concentrated in RB_01–RB_03 — any whole-run partition
leaves classes missing from a bucket). The canonical split lives at
`ml_datasets/cs_robocup_2023/split.json` (22 055 val / 9 452 test).
`run_experiment(split=...)` filters per bucket.

### Gaze ablation — `humandet`, `configs/sweeps/ablation_gaze.yaml`

The `eye_symmetry` (pre-T1.3, yaw-only) gaze algorithm is retained as a named
baseline; `gaze_algorithm` is a sweep axis. Because the algorithm name is part of
`params_slug`, the summariser ranks `head_pose` vs `eye_symmetry` side by side —
no separate comparison tool. See [`scoring.md`](scoring.md) for both algorithms
and [`design-rationale.md`](design-rationale.md) for why MiDaS was ruled out as a
baseline.

---

## Dataset preparation (cs_robocup_2023)

The CoreSense RoboCup @ Home 2023 dataset
([download](https://zenodo.org/records/13902513)). Both RGB **and** depth are
extracted (depth is required).

```bash
mkdir -p ros_datasets/cs_robocup_2023
# unzip each populated run (RB_01–RB_03, RB_05, RB_06, RB_08; RB_04/RB_07 empty)
unzip <dl>/2023_Dataset/Dataset_v2/RB_01/RB_01.zip -d ros_datasets/cs_robocup_2023/RB_01
# RB_01 only: decompress .zst bags first
( cd ros_datasets/cs_robocup_2023/RB_01/mapeo1 && unzstd *.zst )
uv run python scripts/extract_ros2_dataset.py cs_robocup_2023
```

On macOS (no native ROS 2), use the Docker wrapper, which runs extraction inside
a `ros:rolling-perception` container with the repo bind-mounted:

```bash
scripts/extract_ros2_dataset_macos.sh cs_robocup_2023
```

The Python-only environment is managed with [uv](https://docs.astral.sh/uv/)
(`uv sync`; run commands via `uv run`). For `extract_ros2_dataset.py`, link the
system `rosbag2_py` into the venv via a `ros2.pth` file and source ROS 2 first —
see the root README's "Installation & prerequisites".
