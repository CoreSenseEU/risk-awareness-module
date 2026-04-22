# RiskAM Improvement Plan

*Author: Jan Zahálka | Date: April 2026*

---

## 1. Overview

This document synthesises the issues identified by the SamXL robotics partner during real-world deployment testing (reported in the internal `RiskAM_Report_Section.pdf`) with an independent code review, and produces a prioritised roadmap of improvements. The goal is a module that (i) functions robustly in real deployment, (ii) is a credible Horizon EU software deliverable, (iii) supports a written technical deliverable, and (iv) is competitive at a reputable scientific venue.

---

## 2. Analysis of SamXL Findings

SamXL tested RiskAM live on the SAMXL Ridgeback (RealSense D345i + NVIDIA Jetson Orin AGX, ARM64). They identified six core problems and ultimately decided to move on to a different module. This section evaluates their findings and proposed remedies.

### 2.1 Accepted findings

#### F1: Relative depth is not a stable proximity proxy
SamXL identified a fundamental issue with MiDaS-based relative depth: the depth value of a human changes when other objects move in or out of frame, even if the human stays put. This makes the proximity sub-score inconsistent across scenes and across robot runs — a serious problem for a safety-critical module.

**Verdict: Fully accepted.** This is a principal correctness issue, not a tuning issue.

**Remedy:** The RealSense D345i already provides dense absolute depth data in metres via a ROS topic. Replacing MiDaS with the depth sensor is the highest-priority change in this plan. As a bonus, this removes one inference model from the pipeline entirely, which is a major performance win, and lifts the `numpy<2.0.0` constraint imposed by MiDaS.

---

#### F2: Gaze calculation has a logical error
The current gaze score is computed from the horizontal symmetry of the eyes around the nose-to-neck midpoint. Eye symmetry is indeed maximised when the person faces the camera — but it is equally maximised when the person looks directly up, directly down, or directly behind (i.e., anytime the head is in a frontoparallel pose with respect to the *vertical* axis, regardless of pitch). The score therefore produces false "aware" readings for people who are looking elsewhere.

**Verdict: Fully accepted.** This is a real algorithmic error.

**Remedy:** Replace the 1D horizontal-offset heuristic with full 3D head pose estimation. YOLO11-Pose already provides the 17-keypoint skeleton including both eyes, nose, and ear landmarks. These are sufficient to fit a coarse head orientation (yaw + pitch) without an additional model. A human is considered "aware" only when both yaw and pitch are within thresholds of the camera-facing direction, not when either alone is small.

---

#### F3: Tracker fails at low frame rates (3–7 Hz)
The current IoU-based `BboxTracker` assumes an object is near its previous position. At 3–7 Hz — measured in live deployment — walking humans can travel far enough between frames that IoU falls below threshold and continuity is lost, causing flickering detections and dropped tracks.

**Verdict: Fully accepted.** The frame rate issue has two components: the tracker is genuinely too simple, and the frame rate is too low.

**Remedy (tracker):** Ultralytics bundles ByteTrack, which uses motion-aware matching and appearance cues, handling low frame rate much more gracefully. Drop the custom `BboxTracker` and enable ByteTrack within the Ultralytics pipeline (`model.track(...)`). No extra dependencies.

**Remedy (frame rate):** Largely resolved by replacing MiDaS with the depth sensor (removes ~100 ms of inference per frame). Further addressed in the performance section (§4.5).

---

#### F4: X-pose assumes the robot always moves in +X
The x-position sub-score is designed as a proxy for "is this person in the robot's path?" under the implicit assumption the robot only moves forward (+X image direction). The Ridgeback has mecanum wheels and moves omnidirectionally, so the assumption is wrong.

**Verdict: Partially accepted.** The specific formulation is wrong for omnidirectional robots. However, the underlying idea — weight risk by the likelihood that the robot and human trajectories will intersect — is sound.

**Remedy:** Subscribe to the robot's instantaneous velocity (`/cmd_vel` or `/odom`) and/or the Nav2 planned path. Project the human's bounding box position into the robot-relative frame, then compute how close the human is to the robot's predicted trajectory. This replaces x-image-position with a more semantically meaningful *path proximity* sub-score. When velocity data is unavailable, graceful degradation to the current x-pose score is acceptable.

---

#### F5: Gaze is a weak awareness proxy in industrial settings
SamXL notes that in a manufacturing environment, experienced workers may be fully aware of the robot without looking at it.

**Verdict: Partially accepted as a limitation, not a reason to remove gaze.** For the academic case (general human-robot interaction, domestic and logistics settings), gaze is a well-motivated and scientifically interesting awareness proxy. The module should document this limitation clearly. A configurable weight (`w_gaze = 0` effectively disables it for industrial users) is already supported.

**No structural change needed.** The fix to the gaze algorithm (F2) and a weight of zero for industrial deployment handle this adequately.

---

#### F6: No dynamic information (humans moving toward/away)
RiskAM evaluates each frame independently. A person walking away from the robot poses no collision risk, but scores the same as a stationary person at the same distance.

**Verdict: Accepted.** This is a real limitation for dynamic environments.

**Remedy:** Add a temporal velocity estimate per tracked person. Once ByteTrack gives stable IDs across frames, tracking centroid positions (in depth-calibrated coordinates) over a short window gives robot-relative human velocity. A person moving away at speed reduces proximity risk; a person moving toward the robot increases it. This adds a fourth sub-score or modulates the proximity sub-score.

---

### 2.2 Rejected suggestions

#### R1: Replace RiskAM with a VLM-based Vision Module
SamXL concluded that rather than fixing RiskAM's issues, it was more productive (given their time constraints) to move to a different module.

**Verdict: Rejected for this project.** SamXL's decision was driven by their project schedule, not by a technical assessment that RiskAM is unfixable. The improvements in §3 address every identified issue. Moreover, RiskAM's explicit, interpretable sub-score architecture is a scientific strength, not a weakness: interpretable scoring with well-defined semantics is more publishable and more auditable than a VLM black box, and directly aligns with EU human-robot safety research priorities.

---

## 3. Additional Improvements

The following improvements are not in the SamXL report but are identified through code review and consideration of publication and deliverable quality.

### 3.1 Head pose estimation replaces eye-symmetry heuristic (extends F2)
Full head pose (yaw, pitch, roll) from keypoints yields a more principled gaze score. Specifically:
- **Yaw** (left-right rotation): current approximate method is retained, made more robust
- **Pitch** (up-down rotation): derived from the vertical ratio of eye-to-chin vs eye-to-forehead distances (available from YOLO11-Pose keypoints)
- Score: Gaussian kernel centred on (yaw=0°, pitch=0°) with configurable standard deviations

This remains purely YOLO-keypoint based — no extra model — and is a meaningful algorithmic contribution for a paper.

### 3.2 Depth normalisation relative to scene and robot speed
With absolute depth in metres, the proximity score should be normalised by safety-relevant thresholds rather than raw percentile. Specifically:
- Define a safe distance `d_safe` (configurable, default 1.5 m)
- Proximity score = `max(0, 1 - depth / d_safe)` — linear falloff from 1 at distance 0 to 0 at `d_safe`, clipped to [0, 1]
- Optionally scale `d_safe` by robot speed: faster robot → larger safety bubble

This is physically interpretable, calibratable, and directly comparable across runs and scenes.

### 3.3 Per-person multi-frame risk with improved aggregation
Currently, the final score is an average over the last N frames. This should be improved:
- Per-tracked-person risk history (keyed by ByteTrack ID, not global average)
- Publish both the *scene risk* (max over all persons) and per-person risks
- Scene risk aggregation: `max(individual_risks) * (1 + alpha * log(1 + n_persons))` where `n_persons > 1` introduces a crowd penalty; alpha configurable (default 0.1)

This handles multi-person scenes more rigorously and is another publishable component.

### 4. Code Quality and Engineering Improvements

#### 4.1 Remove global mutable state from `score.py`
`prev_risks` is a module-level list mutated by `risk_awareness_score()`. This is not thread-safe, not testable, and breaks if the function is called from multiple contexts. Refactor into a stateful `RiskScorer` class instantiated in the ROS node.

#### 4.2 Lift the `numpy<2.0.0` constraint
This constraint exists because MiDaS's `timm`/hub-loaded model is incompatible with numpy 2.x. Removing MiDaS lifts this constraint, enabling full compatibility with current numpy and avoiding dependency conflicts with other CoreSense packages.

#### 4.3 Async frame processing with proper queue management
The ROS node currently processes frames synchronously in the image callback (queue size 2). At 3–7 Hz, this causes callback backup. Replace with:
- Callback writes incoming frame to a `threading.Event` + shared frame buffer (latest-only, not a queue)
- Separate processing thread consumes the latest frame
- Publishes results from processing thread
This is a standard ROS2 pattern for compute-heavy perception nodes.

#### 4.4 Structured logging and diagnostics
Add ROS2 `diagnostic_msgs` publication: model inference times, detection count per frame, per-frame FPS, and track ID count. This helps operators debug deployment issues without modifying code, and provides data for the paper's performance evaluation.

#### 4.5 Remove dead/unused code
`riskam/ml/gaze.py` (MediaPipe-based gaze, superseded), `riskam/ml/llava.py` (LLaVA integration, unused in deployment), and `riskam/ml/autoencoder.py` (apparently incomplete) should either be completed and integrated or removed. Keeping half-finished code in a deliverable codebase creates confusion.

If there is no plan to use LLaVA/autoencoder in the near future, remove them. They can always be recovered from git history.

#### 4.6 Input validation and error handling
- Validate that `w_proximity + w_gaze + w_position == 1.0` (or at least warn)
- Guard model inference with try/except and publish a clearly marked "error" score + log message on failure
- Handle the case where the depth topic is not available (fall back to MiDaS or fail clearly)

#### 4.7 Unit and integration tests
Add at minimum:
- Unit tests for `score.py` (deterministic formula, edge cases)
- Unit tests for the new head pose gaze calculation
- Unit tests for depth normalisation
- A smoke-test that runs the full pipeline on a single test image with mocked models

#### 4.8 Hardcoded topic names in `riskam_bagger.py`
All six topic strings are hardcoded. Move them to the YAML configuration and declare them as node parameters, consistent with `riskam_node.py`.

#### 4.9 Evaluation framework refactor (pre-T3.3)

The evaluation code (`riskam/experiments.py`, `scripts/run_experiments.py`, `scripts/run_experiments_cs_robocup_2023_all.sh`, `riskam/data/annotator.py`) was written against the pre-improvement pipeline and is not fit as a scientific benchmark for the new one. A code audit identified the following issues; they are prerequisite to T3.3/T3.4 as originally scoped, and the roadmap entries below are expanded accordingly.

**Correctness / pipeline parity with the live node:**
1. **Offline depth is not loaded.** `ml_datasets/cs_robocup_2023/raw_dataset/RB_XX/depth/` holds per-frame absolute-depth `.npy` files, but `run_experiment()` calls `extract_human_risk_awareness_features(..., depth_image_m=None)`. The proximity sub-score is therefore identically zero for the entire sweep — the single most important new sub-score is silently disabled.
2. **Approach sub-score is dead data.** `featextr` computes `features["approach"]` per T2.2, but `RiskScorer.score()` never multiplies it in; there is no `w_approach` weight. The improvement is unevaluable and has no runtime effect.
3. **Path-aware trajectory is ROS-only.** T2.1's path-proximity score lives in `riskam_node._path_proximity_score` and depends on `/cmd_vel`. Offline experiments fall back to the pre-improvement x-offset heuristic. T2.1 cannot be evaluated against the dataset in the current layout; either the projection must move into `featextr`/`RiskScorer`, or the offline loop must replay `/cmd_vel` from the source bag.

**Scientific reporting:**
4. **Metrics are coarse.** Only `correct` / `underestimate` / `overestimate` counts are produced. No per-class precision/recall/F1, no confusion matrix, no regression metric (MAE/RMSE against class centre). Config ranking is not meaningful.
5. **No cross-run aggregation.** Each `RB_XX` run writes its own JSONs in isolation. No summary across all runs, no ranked best-config table, no plots.
6. **No ablation harness and no baseline reconstruction.** Quantifying each individual improvement requires (a) first-class "disable sub-score X" beyond zeroing a weight and (b) a runnable pre-improvement baseline (MiDaS-style relative depth, eye-symmetry gaze, x-position trajectory) recovered from git history behind a feature flag.
7. **No train/val/test split on the annotations.** Parameter sweeps tune on the same set that reports the headline number. Overfitting concern.

**Engineering / workflow:**
8. **Hardcoded parameter grid.** `RISK_SCORE_WEIGHTS`, `GAZE_SIGMA_YAW_VALUES`, `GAZE_SIGMA_PITCH_VALUES` live in source; adding `w_approach`, `d_safe`, `crowd_alpha` requires code edits. Grid definitions belong in a config file.
9. **No feature caching.** YOLO detection/pose inference is deterministic per image yet is re-run for every parameter combination. The standard 36-config × 7-run sweep is ≈250× redundant; cached bboxes/keypoints/depth-at-bbox turn the sweep from O(hours) to O(minutes).
10. **No eval-only mode.** Raw predictions are saved per run but cannot be re-scored under a different metric or risk-breakpoint set without re-running inference.
11. **No reproducibility metadata.** Results are not stamped with git SHA, config hash, package versions, hostname, or seed.
12. **No CI regression gate.** Nothing fails a PR if a change silently regresses known accuracy on a curated subset.

**Annotator:**
13. The tool saves a single integer label per frame, overwrites existing annotations destructively, offers no resume/progress, displays no current-pipeline prediction to calibrate against, and cannot label multi-person scenes per-person. These gaps must be closed before the T3.4 re-annotation pass to keep the cost manageable.

#### 4.10 Sub-score input-availability contract

Rather than handling optional inputs ad hoc at every call site, RiskAM formalises a per-sub-score contract in `riskam/ml/subscores.py`:

- **Required inputs (hard floor):** RGB image + absolute depth in metres. `FrameInputs.validate()` enforces this at the pipeline boundary; missing either raises `ValueError`. A risk module that silently zeroes its dominant sub-score when depth drops out is a safety hazard, not a degradation mode.
- **Optional inputs:** `cmd_vel` (enables path-aware x_offset); track continuity (enables approach via ByteTrack IDs + velocity history).
- **Per-sub-score status:** each `compute_*` function returns a `SubScoreResult(values, status, reason)` where `status ∈ {ACTIVE, FALLBACK, UNAVAILABLE}`. ACTIVE = live inputs used. FALLBACK = a declared fallback produced a value (e.g. centre-offset when `cmd_vel` is missing). UNAVAILABLE = no fallback available; the value is neutral and carries no information.
- **Scoring semantics unchanged:** the scorer multiplies `values * weight` regardless of status. Weights are **not** silently redistributed — that would change scoring semantics without telling the caller. Status flows out via `FrameExtraction.subscore_status`.
- **Diagnostics:** the ROS node publishes one `subscore_{name}` key per sub-score on `/riskam/diagnostics`, plus a `subscore_{name}_reason` when degraded; any non-ACTIVE status escalates the diagnostic level from OK to WARN so operators see degraded modes on the bus.
- **Framework-neutral:** `RobotVelocity` is a small dataclass (`linear_x`, `linear_y`, `speed`) constructible from a ROS `Twist` via `RobotVelocity.from_twist(...)`, so the subscores module has no ROS import and is usable from offline experiments, tests, and notebooks.

Concretely, the live pipeline and the offline experiment now both go through `featextr.extract(FrameInputs(...)) → FrameExtraction`. The ROS node supplies `cmd_vel`; offline experiments pass `cmd_vel=None` (x_offset reports FALLBACK) until bag-backed velocity replay lands.

---

## 5. ROS and Package Stack

**Conservative stance:** RiskAM must work with the rest of the CoreSense stack. No ROS version changes are proposed.

The following stack-level changes are safe:
- **Remove MiDaS** (`timm`, `torch.hub` MiDaS) from `requirements.txt` and `setup.py` once depth sensor integration is complete. This is a significant simplification.
- **Lift `numpy<2.0.0`** pin once MiDaS is removed.
- **Add `nav_msgs`** to `package.xml` (for `/odom` subscription in path-aware x-pose).
- **Keep Ultralytics / YOLO11** — current version is appropriate. ByteTrack is already bundled.
- **Keep MediaPipe** only if the face mesh gaze module is kept for comparison experiments. Otherwise remove.
- **Keep ROS2 Rolling** — do not propose ROS2 version changes.
- **Docker:** The ARM64 Dockerfile for the Jetson Orin should be maintained in the repository. A standard AMD64 Dockerfile for development should be added.

---

## 6. Prioritised Roadmap

**Status legend.** The *Status* column tracks implementation progress so this document remains reusable as the plan evolves. Values: **Done** — item fully implemented and on `main`; **Partial** — item partially addressed (see note); **Pending** — not yet started. Last reviewed: 2026-04-22 against commit `4fc6711` ("Tier 1 (SamXL feedback) and Tier 2 (further quality improvement) updates from the improvement plan implemented").

### Tier 1 — Critical correctness and performance (do first)

| # | Item | Effort | Impact | Status | Notes |
|---|------|--------|--------|--------|-------|
| T1.1 | Replace MiDaS with RealSense depth topic | Medium | Very high | **Done** | Eliminates F1, speeds up pipeline, removes numpy pin. `riskam/ml/depth.py` now operates on absolute metres; MiDaS code and the `numpy<2.0.0` pin are gone. |
| T1.2 | Depth-calibrated proximity score (§3.2) | Low | High | **Done** | Physically interpretable, requires T1.1. `extract_bbox_proximities` uses `max(0, 1 − depth/d_safe)` with configurable `d_safe`. |
| T1.3 | Fix gaze with 2D head pose (yaw + pitch) | Medium | High | **Done** | Eliminates F2, publishable algorithm change. `humandet._headpose_gaze` combines yaw and pitch Gaussians from YOLO11-Pose keypoints. |
| T1.4 | Replace custom tracker with ByteTrack | Low | High | **Done** | Eliminates F3, one Ultralytics flag change. `humandet.detect_humans` calls `model.track(..., persist=True)` and returns track IDs. |
| T1.5 | Async frame processing | Medium | High | **Done** | Fixes F3 performance root cause. `riskam_node` stores the latest synced color+depth pair and processes it in a dedicated worker thread. |
| T1.6 | Remove global state from `score.py` | Low | Medium | **Done** | Engineering prerequisite for tests. Replaced module-level `prev_risks` with `RiskScorer` class instantiated by the ROS node. |

### Tier 2 — Significant functionality and quality

| # | Item | Effort | Impact | Status | Notes |
|---|------|--------|--------|--------|-------|
| T2.1 | Path-aware trajectory sub-score | Medium-High | High | **Done** | Eliminates F4; subscribe to `/cmd_vel`. `riskam_node._path_proximity_score` projects the bbox centre onto the robot's instantaneous motion direction, with graceful fallback to centre-offset when stationary. |
| T2.2 | Human velocity tracking (per ByteTrack ID) | Medium | High | **Done** | Addresses F6. `humandet.update_velocity` + `approach_scores` maintain a per-track depth history and emit an approach score. |
| T2.3 | Multi-person risk aggregation (§3.3) | Low-Medium | Medium | **Done** | Publishable component. `RiskScorer` applies `max(per_person) * (1 + α·log(1 + n_extra))` with configurable `crowd_alpha`. |
| T2.4 | Diagnostic topic publication | Low | Medium | **Done** | Useful for deployment and benchmarking. `/riskam/diagnostics` (`diagnostic_msgs/DiagnosticArray`) publishes frame time, person count, track count, and risk score. |
| T2.5 | Remove dead code (LLaVA, autoencoder, old gaze.py) | Low | Medium | **Done** | Code hygiene. `riskam/ml/gaze.py`, `llava.py`, `autoencoder.py` and `scripts/llava_sandbox.py` removed; recoverable from git history if ever needed. |
| T2.6 | Unit and integration test suite | Medium | Medium | **Done** | Required for deliverable quality. `tests/` contains `test_score.py`, `test_depth.py`, and `test_gaze.py` covering the new scorer, depth calibration, and head-pose gaze. A mocked-model full-pipeline smoke test remains a candidate future addition. |

### Tier 3 — Polish, documentation, scientific presentation

| # | Item | Effort | Impact | Status | Notes |
|---|------|--------|--------|--------|-------|
| T3.1 | ARM64 + AMD64 Dockerfiles | Low | High for deployment | **Pending** | SamXL already has ARM64 variant. |
| T3.2 | Config param validation and better error handling | Low | Medium | **Partial** | Weight-sum validation and broad frame-processing try/except landed with the Tier 1/2 pass; depth-topic-unavailable fallback and full input validation still to do. |
| T3.3 | Evaluation framework refactor | — | — | **Pending** | Expanded into T3.3.1–T3.3.12 below (see §4.9). |
| T3.3.1 | Load offline depth in experiments | Low | **Very high** | **Done** | Depth extraction re-enabled in `extract_cs_robocup.py` (raw uint16 mm `.npy`); `CSRobocup2023DepthIndex` does nearest-neighbour RGB↔depth lookup via binary search; `experiments.run_experiment` now passes `depth_image_m` to `featextr`. Covered by `tests/test_cs_robocup_depth_index.py`. |
| T3.3.2 | Wire `approach` sub-score into `RiskScorer` | Low | **High** | **Done** | `w_approach` added to `RiskScorer.score()` (default 0.0 to preserve live behaviour pending T3.3.11 calibration); declared as ROS param + validated in the weight sum + plumbed through `riskam_node` and `riskam_config.yml`; offline experiments enable ByteTrack (`track_bboxes=True`) and reset `humandet._velocity_history` per run via the new `reset_velocity_history()` helper; sweep extended to alternate `w_approach ∈ {0.0, 0.15}` so the ablation is visible. Covered by 4 new tests in `test_score.py::TestRiskScorerApproach`. |
| T3.3.3 | Sub-score input contract + offline evaluability of path-aware trajectory | Medium | High | **Partial** | Contract landed (see §4.10). Path-projection math lives in `riskam/ml/subscores.compute_x_offset`; live + offline paths share the same orchestrator (`featextr.extract(FrameInputs(...))`). Status (ACTIVE/FALLBACK/UNAVAILABLE) flows through to the diagnostics topic. Remaining: bag-backed `/cmd_vel` extraction + nearest-neighbour index so offline path-aware x_offset can be ACTIVE; blocked on working with the data. |
| T3.3.4 | Expanded classification + regression metrics | Low | High | **Done** | `riskam/eval_metrics.py`: per-class precision/recall/F1/support, 4×4 confusion matrix, macro/micro averages (micro ≡ accuracy), MAE/RMSE vs class-centre targets. Exposed via `classification_report(y_true, y_pred_continuous)` and merged into each experiment's `results.json` under `classification`. Per-run summary line printed at completion. 18 unit tests in `tests/test_eval_metrics.py`. |
| T3.3.5 | Cross-run aggregation + ranked reports | Medium | High | **Done** | `riskam/eval_summary.py`: walks `exp_results/<dataset>/<run>/<params_slug>/results.json`, pools confusion matrices cell-wise across runs per config, recomputes per-class / macro / micro metrics from the pooled CM (statistically correct), sample-weights MAE/RMSE across runs. Emits `summary.json` (machine-readable) + `summary.md` (ranked top-K table + per-run breakdown of the best config). CLI at `scripts/summarize_experiments.py` with `--rank-metric {macro_f1,accuracy,mae,rmse} --top-k N`. 14 unit tests in `tests/test_eval_summary.py`. Plots deferred to later (not a blocker for the paper numbers). |
| T3.3.6 | Config-driven parameter sweep | Low | Medium | **Done** | `riskam/sweep_config.py` + `configs/sweeps/default.yaml`: weight tuples (coupled, sum to 1 ± 0.01) plus independent `gaze_sigma_yaw` / `gaze_sigma_pitch` axes; `SweepConfig.iter_experiments()` yields the Cartesian product. Hardcoded grid removed from `experiments.py`. CLI gains `--sweep-config path/to/foo.yaml`; default path is `configs/sweeps/default.yaml`. Weight-sum validation, unknown-key detection, missing-field errors all raise with source path in the message. 15 unit tests in `tests/test_sweep_config.py`. Extension axes (`d_safe`, `crowd_alpha`) are trivial follow-ups once ablation dictates; deferred until needed. |
| T3.3.7 | Feature-extraction caching | Medium | High | Pending | Cache YOLO bboxes/keypoints + depth-at-bbox per (dataset, run, model SHA). Reduces the standard sweep from O(hours) to O(minutes). |
| T3.3.8 | Eval-only mode | Low | Medium | **Done** | `riskam/reeval.py`: given an experiment's cached `raw_predictions.json`, recomputes all metrics (correct/under/over counts, classification report, predictions.json) without re-running YOLO/featextr. Stamps a fresh provenance block with `reeval_of` pointing at the original inference-run git SHA so the audit chain is preserved. Walks `exp_results/<dataset>/<bucket>/` via `reevaluate_tree`. CLI at `scripts/reeval_experiments.py` with `--bucket` and `--output-suffix` (default overwrites; suffix writes alongside). 11 unit tests in `tests/test_reeval.py`. |
| T3.3.9 | Reproducibility metadata | Low | Medium | **Done** | `riskam/provenance.reproducibility_metadata()` stamps git SHA, git-dirty flag, hostname, ISO-8601 UTC timestamp, Python version, and tracked package versions (torch, ultralytics, numpy, opencv-python). Merged into each `results.json` under `provenance`; params dict is also mirrored under `params` so a result is self-describing. 4 smoke tests in `tests/test_provenance.py`. |
| T3.3.10 | Val/test split on annotations (research methodology) | Low | Medium | **Done** | Note: there is no train split — RiskAM has no trainable parameters, so the sweep does *hyperparameter selection* on val and the final report comes from a single pass on test. This is research methodology used by the RiskAM authors when evaluating, **not** a user workflow. `riskam/data/splits.py`: stratified within-run split (whole-run holdout rejected for cs_robocup_2023 — RB_07 empty, RB_05 single-class, etc.), seed=42, default test_fraction=0.3. CLI at `scripts/generate_split.py`; canonical split written to `ml_datasets/cs_robocup_2023/split.json` (22 055 val / 9 452 test). `experiments.run_experiment(split=...)` filters per bucket; output layout becomes `exp_results/<dataset>/{all,val,test}/<run>/<slug>/`. 10 unit tests in `tests/test_splits.py`. |
| T3.3.11 | Ablation harness + baseline reconstruction | Medium | **High for paper** | **Done** | Scope reduced after the RGB+depth-required decision: MiDaS baseline is out (regression on T1.1), so only the gaze algorithm needs a named alternative. `_eye_symmetry_gaze` (pre-T1.3 yaw-only baseline, F2 weakness retained by construction) added to `humandet`; `gaze_scores(..., algorithm=...)` dispatches; `compute_gaze` and `featextr.extract` thread the selector. `SweepConfig.gaze_algorithm` is a new Cartesian axis (defaults to `("head_pose",)` so old configs are unchanged); `configs/sweeps/ablation_gaze.yaml` crosses both algorithms for the paper comparison (72 cells). No side-by-side comparison tool needed: the algorithm is part of `params_slug`, so the existing summarizer already ranks both algorithms together. Other ablation dimensions exist as first-class config knobs already: `w_approach ∈ {0, 0.15}` (T3.3.2 sweep), `cmd_vel=None` fallback for x_offset (T3.3.3 contract). 15 new tests in `tests/test_gaze_ablation.py`. |
| T3.3.12 | CI regression gate | Medium | Medium | Pending | Small curated subset; fail PRs whose key metric regresses beyond a documented budget. |
| T3.4 | Annotation tool and dataset re-annotation | — | — | **Pending** | Expanded into T3.4.1–T3.4.5 below. |
| T3.4.1 | Show current-pipeline prediction while annotating | Low | Medium | Pending | Run the new pipeline once per frame; display its class alongside the image to calibrate annotator judgment. |
| T3.4.2 | Non-destructive save + resume/progress | Low | Medium | Pending | Backup prior annotations on save; resume from the last labelled frame. |
| T3.4.3 | Per-person (per-bbox) annotation | Medium | Medium | Pending | Scene-level label is a max; per-person labels enable per-sub-score evaluation and support multi-person scenes properly. |
| T3.4.4 | Optional per-sub-score annotations | Medium | **High for paper** | Pending | Separate ground truth for proximity / gaze / path / approach where feasible; enables fine-grained ablation and validates each sub-score in isolation. |
| T3.4.5 | Re-annotate cs_robocup_2023 against the new pipeline | Medium (human) | **High for paper** | Pending | Uses T3.4.1–T3.4.4. Blocks the ablation evaluation in T3.3.11. |
| T3.5 | `riskam_bagger.py` configurable topics | Low | Low | **Done** | Bagger topics are now declared as ROS parameters in `riskam_bagger.py` and mirrored in `riskam_config.yml`. Landed incidentally in the Tier 1/2 pass. |
| T3.6 | API documentation (docstrings, README) + user deployment guide | Low | Medium | **Pending** | EU deliverable presentation. Should explicitly distinguish *user-facing* configuration (physical params: `d_safe`, gaze sigmas defensible from camera/robot specs) from *research-methodology* tooling (val/test split, ablation harness, sweep). Roboticists deploying RiskAM must not be required to run a hyperparameter sweep; sensible calibrated defaults ship in `riskam_config.yml`. |
| T3.7 | Sphinx API docs | Low | Low | **Pending** | Optional, if deliverable template requires. |

---

## 7. Scientific Contribution Framing

For a reputable venue (e.g., ICRA, IROS, RA-L, RAS), the paper's contribution should be:

> *An interpretable, real-time multi-factor risk awareness module for human-robot interaction, featuring absolute-depth-calibrated proximity, 2D-head-pose-corrected gaze awareness, robot-trajectory-aware path risk, and temporally consistent tracking. Evaluated quantitatively against annotated real-world deployment data on an omnidirectional mobile platform.*

Key technical novelties that differentiate from prior work:
1. Fusing absolute depth from a RGB-D sensor with 3D head pose into a jointly calibrated, physically interpretable risk score
2. Robot-trajectory-aware collision risk weighting (path projection from velocity/nav)
3. Per-tracked-person temporal risk aggregation with multi-person crowd penalty
4. Quantitative ablation study comparing the original heuristic sub-scores with each proposed improvement

The explicit score architecture (vs black-box VLMs) is a selling point for explainability-focused venues and EU safety-focused robotics workshops.

---

## 8. Summary of Decisions

| SamXL suggestion | Decision | Rationale |
|------------------|----------|-----------|
| Use RealSense depth instead of MiDaS | **Accept** | Correct, high priority (T1.1) |
| 3D bounding box fusion | **Simplify** | Depth at bbox centroid sufficient; full 3D box overkill |
| Fix gaze with head tilt | **Accept, extend** | Full 2D head pose (T1.3) |
| Path-aware x-pose | **Accept** | Subscribe to velocity/nav (T2.1) |
| Abandon gaze for industrial settings | **Reject** | Configurable weight handles this; gaze is scientifically useful |
| Add human velocity tracking | **Accept** | Temporal tracking via ByteTrack IDs (T2.2) |
| Abandon RiskAM for VLM module | **Reject** | Issues are fixable; interpretability is a scientific asset |

---

*This plan should be reviewed and updated as implementation progresses. The Tier 1 items represent the minimum viable improvement set for the next deliverable milestone.*

---

## 9. Implementation Status Log

A running record of the plan's progress. Extend by appending new entries; do not rewrite history.

- **2026-04-22** — Verified commit `4fc6711` against the roadmap. All Tier 1 items (T1.1–T1.6) and all Tier 2 items (T2.1–T2.6) are implemented. Tier 3 landed incidentally: T3.5 (bagger configurable topics) is complete; T3.2 (config validation and error handling) is partial. T3.1, T3.3, T3.4, T3.6, T3.7 remain pending.
- **2026-04-22** — Pre-T3.3 audit of the evaluation framework. Three pipeline-parity bugs were found: offline experiments pass `depth_image_m=None` despite per-frame depth `.npy` files being available (T3.3.1); `RiskScorer.score()` never consumes `features["approach"]`, so T2.2 is dead data (T3.3.2); the path-aware trajectory score is computed only inside the ROS node, so T2.1 is not evaluable offline against cs_robocup_2023 (T3.3.3). The framework also lacks standard ML metrics, cross-run aggregation, a config-driven sweep, feature caching, eval-only re-scoring, reproducibility metadata, an ablation harness with a pre-improvement baseline, annotation train/val/test split, and a CI regression gate. The annotator tool is likewise too primitive for the T3.4 re-annotation pass (destructive save, single label per frame, no resume, no prediction-assisted view). Plan updated: new §4.9 narrative; T3.3 decomposed into T3.3.1–T3.3.12, T3.4 into T3.4.1–T3.4.5.
- **2026-04-22** — T3.3.1 landed. Depth saving re-enabled in `riskam/data/extract_cs_robocup.py` (raw uint16 mm `.npy` under `RB_XX/depth/`); `CSRobocup2023DepthIndex` added to `riskam/data/cs_robocup_2023.py` for nearest-neighbour RGB↔depth lookup with a 0.2 s tolerance; `riskam/experiments.py` builds the index per run and passes `depth_image_m` into `featextr`. Six-test unit suite added at `tests/test_cs_robocup_depth_index.py`; full test suite passes (40 tests). Follow-up: a re-extraction pass against the source bags is needed before the next experiment run — users who already have extracted RGB will need to re-run `scripts/extract_ros2_dataset.py` to populate the `depth/` directories.
- **2026-04-22** — T3.3.2 landed. `RiskScorer.score()` now consumes `features["approach"]` via a new `w_approach` weight (defaulting to 0.0 to keep live runtime behaviour identical until T3.3.11 produces a calibrated value). `w_approach` is declared as a ROS parameter in `riskam_node`, included in the weight-sum validation, and mirrored in `riskam_bringup/config/riskam_config.yml`. Enabling the sub-score required two sibling fixes in the offline path: ByteTrack is now enabled in `experiments.py` (`track_bboxes=True`), and `humandet.reset_velocity_history()` is called at the start of each run so depth observations from a prior run/config don't pollute the linear-fit window. The experiment sweep now alternates `w_approach ∈ {0.0, 0.15}` so the ablation between approach-off and approach-on is directly visible in the results. Four new tests in `tests/test_score.py::TestRiskScorerApproach` cover default, weighted contribution, proportionality, and backwards-compat for callers that omit the key; full suite passes (44 tests).
- **2026-04-22** — T3.3.3 (refactor half) landed. Sub-score input-availability contract introduced in `riskam/ml/subscores.py` (see §4.10): `FrameInputs` (required RGB + depth, optional `cmd_vel`), `RobotVelocity` (ROS-neutral), `SubScoreStatus` enum, `SubScoreResult`, `FrameExtraction`, and `compute_proximity/gaze/x_offset/approach` functions. `featextr` now orchestrates via the contract (`featextr.extract(FrameInputs) → FrameExtraction`); the legacy 4-tuple API is replaced. Supporting migrations: `humandet.detect_humans` no longer returns `offset_scores` (now owned by `compute_x_offset`); `_path_proximity_score` deleted from `riskam_node` (unified with the offline path); experiments.py skips frames with no matching depth (new `skipped_no_depth` metric); `scripts/test_run_with_video.py` and `scripts/featextr_sandbox.py` migrated to load depth via the index. The ROS node escalates the diagnostic level from OK to WARN whenever any sub-score is non-ACTIVE and publishes per-sub-score status + reason on `/riskam/diagnostics`. One latent bug discovered and fixed in the process: the velocity tracker was being fed proximity scores instead of depth metres (`humandet.update_velocity(track_ids, bbox_proximities)` in the old featextr), inverting the units of the linear-fit slope and producing wrong-signed approach scores; fixed by introducing `depth.extract_bbox_depths` and feeding raw metres to `update_velocity`. 16 new tests in `tests/test_subscores.py` cover validation, status transitions, and end-to-end contract enforcement; full suite passes (60 tests). Remaining for T3.3.3: bag-backed `/cmd_vel` extraction + index so the offline path can drive x_offset to ACTIVE — blocked on working with the data.
- **2026-04-22** — T3.3.4 + T3.3.9 landed. Two new modules: `riskam/eval_metrics.py` (per-class P/R/F1, confusion matrix, macro/micro averages, class-centre MAE/RMSE; exposed via `classification_report`) and `riskam/provenance.py` (git SHA + dirty flag, hostname, UTC timestamp, Python version, tracked-package versions). Both are plumbed into `experiments.run_experiment`: after the per-frame loop, `metrics["classification"]` gets the full report, `metrics["provenance"]` gets the environment stamp, and `metrics["params"]` mirrors the sweep params so each `results.json` is self-describing. End-of-run print adds a one-line summary (`Macro F1 | Accuracy | MAE | RMSE`). 22 new tests (`tests/test_eval_metrics.py` ×18, `tests/test_provenance.py` ×4); full suite passes (82 tests).
- **2026-04-22** — T3.3.5 landed. New module `riskam/eval_summary.py` walks the per-experiment `results.json` files under `exp_results/<dataset>/<run>/<params_slug>/` and pools metrics across runs per config: confusion matrices summed cell-wise with per-class/macro/micro recomputed from the pooled CM, MAE/RMSE sample-weighted, avg_time meaned. Writes `summary.json` and a ranked-top-K markdown table (`summary.md`) to the dataset root, plus a per-run breakdown of the best config. CLI at `scripts/summarize_experiments.py` with `--rank-metric {macro_f1,accuracy,mae,rmse}` and `--top-k N`. 14 new tests (`tests/test_eval_summary.py`) against synthetic fixtures; full suite passes (96 tests). Plots (confusion-matrix heatmaps, per-run accuracy charts) left as a later addition — not a blocker for the paper numbers.
- **2026-04-22** — T3.3.10 landed (renamed from "train/val/test" to "val/test"). Reframed as a research-methodology tool: RiskAM has no trainable parameters, so the sweep selects hyperparameters on val and the final paper number comes from a single pass on test. **This is for the RiskAM authors when publishing, not a roboticist workflow** — deployed RiskAM must work out of the box with calibrated defaults (T3.6 was extended to call this distinction out explicitly). Implementation: `riskam/data/splits.py` does stratified within-run by class (whole-run holdout rejected for cs_robocup_2023: RB_07 has zero annotations, RB_05 contains only class 3, classes 0/1 live almost entirely in RB_01–RB_03 — any whole-run partition leaves classes missing from one bucket or only one run in test). CLI `scripts/generate_split.py` writes the canonical split to `ml_datasets/cs_robocup_2023/split.json`; on real ground truth this gives a clean 70/30 per (run, class) — 22 055 val / 9 452 test frames across the six populated runs. `experiments.run_experiment(split={'val','test',None})` filters ground truth to the requested bucket; output layout adds a bucket level (`exp_results/<dataset>/{all,val,test}/<run>/<slug>/`); `summarize_experiments.py` takes `--bucket` to match. 10 new tests in `tests/test_splits.py`; full suite passes (106 tests).
- **2026-04-22** — T3.3.8 landed. New module `riskam/reeval.py`: `reevaluate_experiment(exp_dir, ground_truth)` reads a cached `raw_predictions.json`, pairs each frame with its ground-truth class, and recomputes correct/under/over counts + the full `classification_report` — no model inference needed. Params, split metadata, and `avg_time` are preserved from the original `results.json`; a fresh provenance stamp is written with `reeval_of` = the original inference-run git SHA so the audit chain survives. `reevaluate_tree(dataset_root, gt_by_run)` walks the entire `exp_results/<dataset>/<bucket>/` layout. CLI at `scripts/reeval_experiments.py` with `--bucket {all,val,test}` and `--output-suffix` (default overwrites `results.json` / `predictions.json`; a suffix writes alongside the originals). 11 new tests in `tests/test_reeval.py`; full suite passes (117 tests).
- **2026-04-22** — T3.3.6 landed. New module `riskam/sweep_config.py` + `configs/sweeps/default.yaml`: weight tuples are coupled (each must sum to 1.0 ± 0.01) while `gaze_sigma_yaw` / `gaze_sigma_pitch` are independent Cartesian-product axes. `SweepConfig.iter_experiments()` yields one params dict per cell. Hardcoded `RISK_SCORE_WEIGHTS` / `GAZE_SIGMA_*_VALUES` constants removed from `experiments.py`; `run_experiments()` now accepts an optional `SweepConfig` argument (loads default if omitted). `scripts/run_experiments.py` gains `--sweep-config path/to/foo.yaml`. Schema validation raises with the source path in the message on missing keys, unknown keys, empty lists, or non-normalised weight tuples. 15 new tests in `tests/test_sweep_config.py`; full suite passes (132 tests). Future extension axes (`d_safe`, `crowd_alpha`) are a trivial addition to the YAML schema once ablation design dictates them.
- **2026-04-22** — T3.3.11 landed with reduced scope. The original plan called for a "pre-improvement baseline" covering MiDaS relative depth + eye-symmetry gaze + x-position trajectory; after the earlier in-session decision to require RGB + absolute depth (MiDaS out), only the gaze algorithm needed a named alternative — centre-offset x_offset already exists as a FALLBACK inside the T3.3.3 contract and `w_approach = 0` is already an explicit sweep axis. New: `_eye_symmetry_gaze` in `humandet` (yaw-only Gaussian around the nose↔eye-midpoint axis, F2 weakness retained by construction and pinned by tests); `gaze_scores(..., algorithm=...)` dispatcher with `GAZE_ALGORITHMS = ("head_pose", "eye_symmetry")`. `compute_gaze` and `featextr.extract` thread the selector through. `SweepConfig.gaze_algorithm` is a new independent Cartesian axis (defaults to `("head_pose",)` so existing configs keep their 36-cell size and slug shape); `configs/sweeps/ablation_gaze.yaml` crosses both algorithms for the paper comparison (72 cells). The algorithm name is part of `params_slug`, so the existing T3.3.5 summarizer already ranks both algorithms side-by-side in the top-K table — no separate comparison tool required. 15 new tests in `tests/test_gaze_ablation.py` plus a one-line fix to an existing sweep-config test; full suite passes (147 tests).
