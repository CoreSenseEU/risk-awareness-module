# Changelog

Distilled implementation history of the RiskAM redesign. Append new entries at
the top; do not rewrite history. For the rationale behind these changes see
[`design-rationale.md`](design-rationale.md); for what is still outstanding see
[`improvement-plan.md`](improvement-plan.md).

Item codes (T1.x / T2.x / T3.x) refer to the original improvement-plan roadmap.

---

## 2026-07-06 — cs_robocup_2024 extraction + Layer-2 hindsight-oracle evaluation

Two milestones: the 2024 dataset is pipeline-ready, and the annotation-free
Layer-2 evaluation (paper plan §Layer 2) runs end-to-end on both RoboCup
datasets. See [`evaluation-framework.md`](evaluation-framework.md)
§"Dataset preparation (cs_robocup_2024)" and §"Layer 2".

**Dataset (10 runs, 96,033 frames, 46 GB):**
- `riskam/data/extract_cs_robocup.py` generalized to a per-dataset spec:
  2024 topics (`/head_front_camera/*`), depth normalized to uint16 mm and
  stored as 16-bit PNG (~3× smaller), odometry derived from `/tf`
  odom→base transforms (the 2024 bags carry no odom topic; verified 50 Hz,
  plausible TIAGo speeds). `riskam/data/cs_robocup_indices.py` — shared
  year-agnostic depth/odom indices. Disk-aware run-at-a-time orchestrator
  `scripts/prepare_cs_robocup_2024.sh`. Every run's frame counts match the
  bag message counts exactly; depth-index hit rate 100 %.
- No annotations, by design — 2024 is evaluated via Layer 2.

**Layer 2 (hindsight oracle + early-warning metrics):**
- New `riskam/hindsight.py` (non-causal oracle; scene-level future-min
  distance, per-track savgol via the `mocap_gt` helpers, import firewall
  against the causal stack), `riskam/ssm.py` (ISO/TS 15066-style margin
  channels, worst-case + awareness-modulated), `riskam/event_table.py`
  (per-frame backbone: causal channels + oracle outcomes; sibling of the
  frozen `scene_table.py`), `riskam/early_warning.py` (ROC/PR AUC per
  (r, T) cell, sustained-alarm episodes, lead time, FA/min,
  matched-recall headline), driver `scripts/layer2.py`,
  `riskam/data/run_datasets.py` (torch-free dataset wiring registry).
- 40 new tests (315 total green).

**First results** (`exp_results/<dataset>/layer2/report.md`):
- Frame-level AUCs 0.71–0.94 across the (r, T) grid on both datasets;
  distance-dominated channels lead where events are near-trivial.
- **Pre-event-only (pure anticipation): `risk_a` is the best channel in
  all 18 cells across both datasets** (2023: 0.57–0.65; 2024: 0.62–0.69)
  — the kinematic awareness-fused metric anticipates encounters better
  than m0, proximity, TTC and both SSM variants. This is the replicated
  headline candidate.
- The awareness-modulated-SSM alarm-time reduction at matched recall is
  8–23 % on 2023 but **does not replicate on 2024** (−3 to +2 %) — an
  honest negative worth investigating (denser crowds, higher reference
  recall) before it appears in the paper.
- Episode-count FA/min is misleading in high-event-density recordings
  (the worst-case reference is in alarm >50 % of total time); alarm-time
  at matched recall is the primary equal-safety comparison.

Also: `scripts/test_run_with_video.py` generalized to any registered
dataset; 2024 showcase videos rendered
(`test_results/videos/cs_robocup_2024_receptionist_1_{risk,raw}.avi`).

---

## 2026-07-03 — THÖR-MAGNI mocap ground truth (evaluation plan Layer 1)

First dataset work under the committed eval revamp (`private/paper-plan.md`
§6): mocap-derived ground-truth kinematics for measurement validity. See
[`evaluation-framework.md`](evaluation-framework.md) §THÖR-MAGNI.

- **`riskam/data/thor_magni.py`** — parser for the Zenodo mocap CSVs
  (metadata header, 100 Hz centroid + body→world rotation tracks per rigid
  body, roles). 52 runs across 5 scenarios.
- **`scripts/calibrate_thor_magni_frames.py`** — empirical frame-convention
  evidence: DARKO forward = body +X (velocity alignment +0.999 on the
  differential-drive SC3A runs — also confirms body→world); helmet facing =
  body ±X with per-helmet mounting sign (Helmet_6 mirrored); footprint scale
  from marker planar spread (0.35–0.57 m).
- **`riskam/mocap_gt.py`** — GT tables per (frame, participant): robot-frame
  planar position, relative velocity (differentiated *after* the robot-frame
  transform so the camera's ω×p transport term is included), closing/
  tangential speed, `t_cpa`, `d_min`, trajectory hazard via vectorized twins
  of the `riskam/kinematics.py` formulas (equality pinned by tests), and a
  head-facing awareness reference with self-calibrated facing sign. NaN-aware
  Savitzky–Golay smoothing per tracked segment; occlusion gaps ≤ 0.3 s
  bridged, never extrapolated. Fixed en route: `np.unwrap` over NaN-holed
  yaw poisoned everything after the first robot-track dropout.
- **`riskam/platforms.py`** — `AZURE_KINECT` sensor + `DARKO_KINECT`
  platform presets.
- Sanity aggregates match the scenario design (robot static in SC1/SC2,
  0.31–0.45 m/s in SC3–5; closest approaches in the HRI scenarios).
- **Blocker:** onboard RGB-D is not in the public record (GDPR,
  upon-request) — the vision-vs-mocap comparison waits on a data request to
  the authors; the GT tables meanwhile serve Layers 2 and 4a.

Offline implementation of both paper-plan directions as a research track that
leaves the deployed pipeline untouched. See
[`experimental-metrics.md`](experimental-metrics.md) for the full reference.

- **`riskam/kinematics.py`** — Direction A: bearing from bbox centre-x +
  camera intrinsics, planar per-track relative-velocity fits (explicit frame
  timestamps, unlike the wall-clock `humandet` history), CPA extrapolation
  (`t_cpa`, `d_min`), trajectory hazard (max of `closeness(t)·exp(−t/τ)`
  over the predicted pass — continuous at v→0, unlike scoring only the
  CPA moment), awareness-modulated fusion `hazard · (1 + β(1−awareness))`.
  Degradation ladder full → static_odom → static (≡ deployed proximity) →
  dead_zone; depth sentinels censored from velocity fits. Two smoothing
  stages against detector/depth noise: per-track channel EMA
  (`ema_tau_s` 0.5 s) and an instant-attack/exponential-release scene
  smoother (`release_tau_s` 0.5 s) that bridges single-frame detection
  dropouts. The release is evidence-bounded (`hold_max_s` 1.0 s, the
  track-bridging horizon): it never extrapolates further than that past
  the last detection, so departures leave no ghost risk in empty frames
  (the video overlay tags held values with `hold`). Result: 2–4× *less*
  frame-to-frame flicker than the deployed metric on every run, and the
  unfitted physics metric's Spearman ρ edged past the hand-tuned
  baseline (+0.758 vs +0.750 on test).
- **`riskam/ordinal.py` + `riskam/proba_metrics.py`** — Direction B1:
  proportional-odds logit (statsmodels, `experiments` dep group) persisted as
  numpy-only JSON fits, LR ablation tests; Brier/log-loss/RPS/cumulative-AUC/
  reliability+ECE/Spearman.
- **`riskam/scene_table.py` + `scripts/metric_lab.py`** — cache → per-frame
  scene feature table; models m0/m0c/a_raw/a_cal/b1_sub/b1_kin fit on val,
  frozen report on test; artifacts under
  `exp_results/cs_robocup_2023/metric_lab/`. Absorbs and replaces the
  exploratory `scripts/metric_probe.py`. A `video` subcommand renders
  per-run MP4s annotated with the Direction-A channels only (no ground
  truth, no deployed metric) for unbiased qualitative review.
- **Aux bag extraction** — `extract_ros2_dataset.py cs_robocup_2023_aux`
  writes per-run `odom.csv` (`/mobile_base_controller/odom`; `/cmd_vel` is
  empty in RB_01/06/07) and `camera_info.json` (fx = 530.2 @ 640 px);
  `CSRobocup2023OdomIndex` + `load_camera_model` degrade gracefully when the
  files are absent. Unblocks the data half of T3.3.3.
- **`riskam/platforms.py`** — additive referent constants:
  `DepthSensor.rgb_hfov_deg` (Xtion 58°, D435 69°),
  `RobotPlatform.footprint_radius_m` (TIAGo 0.27 m, Ridgeback 0.48 m),
  `TAU_REACTION_S_DEFAULT` (2.0 s), `BETA_UNAWARE_DEFAULT` (1.0).

## 2026-05-06 — Sensor compatibility (T2.7) and visualisation

- **T2.7 — RealSense + Xtion support.** Added the near-clip dead-zone fallback in
  `riskam/ml/depth.py`: `depth_near_clip_m` (default `0.0` keeps RealSense
  behaviour bit-identical) and `near_clip_bbox_min_frac`. When both fire,
  `extract_bbox_depths` returns `0 m` (max proximity) for a person in the dead
  zone instead of `d_safe`. `depth_to_visualization` upper bound expanded to
  `max(d_safe, p95)`. Surfaced from soundness testing the offline pipeline
  against cs_robocup_2023 (Xtion). See [`sensors-and-platforms.md`](sensors-and-platforms.md).
- **T2.7 robustness.** Generalised the close-fallback condition from "exactly
  zero valid pixels" to "valid fraction ≤ `near_clip_valid_frac_max`" (default
  `0.0`) to absorb background bleed-through on structured-light sensors.
- **T2.7 platform abstraction.** Introduced `riskam/platforms.py`
  (`DepthSensor` → `RobotPlatform`) with presets `RIDGEBACK_D435` (SamXL) and
  `TIAGO_XTION` (cs_robocup_2023, `d_safe` 1.5 → 2.5 m). The live ROS path still
  reads scalars from `riskam_config.yml`.
- **Visualisation.** Bbox interior switched to red-by-proximity (`α = proximity ·
  0.7`); gaze coloured by a continuous red↔white gradient with a black outline;
  depth blend scoped to bbox interiors only. See [`visualization.md`](visualization.md).

## 2026-04-22 — Evaluation framework refactor (T3.3.*) and docs

A single large session brought the offline evaluation framework to scientific
standard and rewrote the README. By item:

- **T3.3.1** — Offline depth re-enabled. `CSRobocup2023DepthIndex` does
  nearest-neighbour RGB↔depth lookup; `experiments` passes `depth_image_m` into
  `featextr`. (Previously the sweep ran with `depth=None`, zeroing proximity.)
- **T3.3.2** — `w_approach` wired into `RiskScorer.score()` (default `0.0`);
  declared as a ROS param, validated in the weight sum, and added as a sweep axis.
  ByteTrack enabled offline + `reset_velocity_history()` per run.
- **T3.3.3 (refactor half)** — Sub-score input-availability contract
  (`riskam/ml/subscores.py`): `FrameInputs`, `RobotVelocity`, `SubScoreStatus`,
  `SubScoreResult`, `FrameExtraction`, `compute_*`. Live + offline unified on
  `featextr.extract(FrameInputs(...))`. Diagnostics escalate to WARN on any
  non-ACTIVE sub-score. Fixed a latent unit bug (velocity tracker was fed
  proximity scores instead of depth metres). *Remaining: bag-backed `/cmd_vel`
  replay — blocked on data.*
- **T3.3.4 + T3.3.9** — `eval_metrics.py` (per-class P/R/F1, confusion matrix,
  macro/micro, MAE/RMSE) and `provenance.py` (git SHA, hostname, versions),
  merged into each `results.json`.
- **T3.3.5** — `eval_summary.py` cross-run aggregation; pools confusion matrices
  cell-wise, emits ranked `summary.md` + `summary.json`.
- **T3.3.6** — Config-driven sweep (`sweep_config.py` + `configs/sweeps/`);
  hardcoded grid removed.
- **T3.3.7** — Feature cache (`feature_cache.py`); `featextr` split into
  `extract_primitives` (cacheable) + `extract_features`.
- **T3.3.8** — Eval-only re-scoring (`reeval.py`) from cached predictions, with a
  `reeval_of` provenance link.
- **T3.3.10** — Val/test split (`data/splits.py`), stratified within-run; no
  train split (RiskAM has no trainable parameters). Canonical split at
  `ml_datasets/cs_robocup_2023/split.json`.
- **T3.3.11** — Gaze ablation: `eye_symmetry` baseline retained;
  `gaze_algorithm` sweep axis; `configs/sweeps/ablation_gaze.yaml`.
- **T3.6 (partial)** — README rewritten against `riskam_config.yml` (removed
  stale `depth_gamma` / `gaze_bounds`; corrected the Python example to the
  `featextr.extract(FrameInputs(...))` API).
- **Plan/audit bookkeeping** — Pre-T3.3 audit identified the framework gaps;
  end-of-session cleanup aligned §5 / T2.4 / T2.6 / T3.2 with reality (T3.2 moved
  to Done: the depth-unavailable fallback was deliberately rejected).

## Tier 1 + Tier 2 (verified at commit `4fc6711`)

Core redesign landed across Tier 1 and Tier 2:

- **T1.1** MiDaS → RealSense absolute depth; `numpy<2.0.0` pin lifted.
- **T1.2** Depth-calibrated proximity (`1 − d/d_safe`).
- **T1.3** 2-D head-pose gaze (yaw + pitch).
- **T1.4** Custom tracker → ByteTrack (`model.track(persist=True)`).
- **T1.5** Async frame processing (worker thread, latest-only buffer).
- **T1.6** Module-global `prev_risks` → stateful `RiskScorer` class.
- **T2.1** Path-aware `x_offset` from `/cmd_vel`.
- **T2.2** Per-ByteTrack-ID approach sub-score.
- **T2.3** Multi-person aggregation with crowd penalty.
- **T2.4** `/riskam/diagnostics` topic (later expanded with per-sub-score status).
- **T2.5** Dead code removed (old `gaze.py`, `llava.py`, `autoencoder.py`).
- **T2.6** Unit/integration test suite (grew to ~190 tests through Tier 3).
- **T3.5** `riskam_bagger.py` topics made configurable (landed incidentally).
