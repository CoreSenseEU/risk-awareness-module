# Changelog

Distilled implementation history of the RiskAM redesign. Append new entries at
the top; do not rewrite history. For the rationale behind these changes see
[`design-rationale.md`](design-rationale.md); for what is still outstanding see
[`improvement-plan.md`](improvement-plan.md).

Item codes (T1.x / T2.x / T3.x) refer to the original improvement-plan roadmap.

---

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
