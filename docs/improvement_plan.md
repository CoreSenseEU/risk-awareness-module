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

### 3.4 Sensor compatibility (RealSense + Xtion)

The T1.1/T1.2 redesign was driven by SamXL's RealSense + Ridgeback deployment. RealSense D4xx delivers clean absolute depth from ~0.3 m onward — `extract_bbox_depths` interprets "no valid depth in bbox" as "person far away" (fall back to `d_safe`, proximity 0), which is roughly correct for that sensor.

The cs_robocup_2023 dataset that backs the offline evaluation framework was recorded on a different platform (TIAGo + PAL Xtion / PrimeSense Carmine). Soundness testing against that data exposed three coupled failure modes:

1. **Visualisation:** the depth overlay rendered uniformly black on every frame. Cause: the viz normalises by `d_safe = 1.5 m`, but the Xtion's working range starts at ~2 m (~85% of pixels are zero, the rest are 2–5 m) so almost everything cliffs to black.
2. **Score collapse near the camera:** a close-up face frame got a scene risk of 0.16 despite the person being directly in front of the lens. Cause: the Xtion has a ~0.5–0.8 m near-clip dead zone — a face standing inside it returns 0% valid depth pixels, the legacy fallback assigns `d_safe` (proximity 0), and the score is left to be carried by gaze + x_offset only (`0.25·gaze + 0.05·x_offset ≈ 0.16` for a centred, aware face).
3. **Wrong "closest bbox" indicator:** with all proximity scores tied at 0, the per-person max-risk index is decided by gaze + x_offset alone, which often disagrees with what a human eye would call "closest".

This is a **calibration & semantics issue, not a bug.** The pipeline does what its code says; the code was tuned for one sensor class. The structurally honest fix is to acknowledge that "depth = 0 in a person-shaped bbox" carries different information depending on the sensor: for RealSense it usually means "far / occluded" (legacy fallback is fine); for structured-light sensors with a hard near-clip it more often means "too close to measure" — and treating that as proximity 0 inverts the safety direction.

T2.7 introduces two physical knobs:

- `depth_near_clip_m` (default `0.0` → fallback off, RealSense behaviour preserved bit-identical): the closest distance the sensor can measure. Set to the sensor's spec-sheet value (e.g. ~0.6 m for Xtion).
- `near_clip_bbox_min_frac` (default `0.05`): minimum bbox area as a fraction of the full frame for the close-fallback to fire. Guards against treating distant noise-filled bboxes as "very close".

When both conditions hold (zero valid depth + bbox above threshold), the person is somewhere in the [0, near_clip] dead zone and we cannot measure where; `extract_bbox_depths` therefore assigns depth `0` (max proximity 1.0) rather than `d_safe` (proximity 0) — safety-conservative.

The depth-to-image visualisation in `depth_to_visualization` was also updated to expand its normalisation upper bound to `max(d_safe, p95 of valid pixels)` so the viz remains depth-varied on sensors whose typical scene range exceeds `d_safe`. When all valid pixels are within `d_safe` (the standard RealSense / SamXL case), the upper bound stays at `d_safe` and the viz is bit-identical to pre-T2.7.

Pre-T2.7 deployment behaviour (RealSense, SamXL, default config) is preserved in full. Tests pin the default RealSense fallback values so any future change that would alter SamXL behaviour fails fast.

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

Stack-level status at end of 2026-04-22 (✅ = landed, ◻ = pending):

- ✅ **MiDaS removed.** `timm` and `torch.hub` MiDaS are gone from `requirements.txt` and `setup.py` (T1.1).
- ✅ **`numpy<2.0.0` pin lifted.** Verified — `requirements.txt` declares `numpy` unpinned.
- ✅ **MediaPipe removed.** Old face-mesh gaze module deleted in T2.5; `requirements.txt` no longer declares it.
- ✅ **Ultralytics / YOLO11 + ByteTrack** kept. Single inference model for the full pipeline.
- ✅ **ROS 2 Rolling** kept. No distro change.
- ✅ **Depth sensors.** RealSense D4xx default-calibrated (live SamXL deployment target); structured-light sensors with a hard near-clip dead zone (PrimeSense Xtion / Carmine — what the cs_robocup_2023 dataset uses) supported via the T2.7 close-fallback (`depth_near_clip_m` parameter). New sensor classes are added by reading the spec-sheet near-clip and setting the parameter.
- ◻ **`nav_msgs` declared in `riskam_ros/package.xml`** but path-aware x_offset actually uses `geometry_msgs/Twist` from `/cmd_vel` (T2.1) — `nav_msgs` is a dead dep; drop when convenient.
- ◻ **AMD64 Dockerfile** (T3.1) pending; the ARM64 Dockerfile for the Jetson Orin should stay.

---

## 6. Prioritised Roadmap

**Status legend.** The *Status* column tracks implementation progress so this document remains reusable as the plan evolves. Values: **Done** — item fully implemented and on `main`; **Partial** — item partially addressed (see note); **Pending** — not yet started. Last reviewed: 2026-04-22 at end of the session adding T3.3.1–T3.3.11 + T3.6 partial on top of `4fc6711`; see the Implementation Status Log (§9) for per-item commits.

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
| T2.4 | Diagnostic topic publication | Low | Medium | **Done** | Useful for deployment and benchmarking. `/riskam/diagnostics` (`diagnostic_msgs/DiagnosticArray`) publishes frame time, person count, track count, and risk score. Expanded during T3.3.3 with per-sub-score status (`active` / `fallback` / `unavailable`) and reason fields, and the diagnostic level escalates OK → WARN whenever any sub-score is non-`active` so degraded modes are visible on the bus. |
| T2.5 | Remove dead code (LLaVA, autoencoder, old gaze.py) | Low | Medium | **Done** | Code hygiene. `riskam/ml/gaze.py`, `llava.py`, `autoencoder.py` and `scripts/llava_sandbox.py` removed; recoverable from git history if ever needed. |
| T2.6 | Unit and integration test suite | Medium | Medium | **Done** | Required for deliverable quality. `tests/` grew substantially during the Tier 3 eval-framework pass; at end of 2026-04-22 it contains 14 test files / 157 tests covering the scorer, depth module, gaze (including the head-pose / eye-symmetry ablation), sub-score input contract, val/test splits, sweep config, eval metrics + regression, provenance stamping, cross-run summary aggregation, eval-only re-scoring, feature cache, and the CS RoboCup 2023 depth index. A mocked-model full-pipeline smoke test remains a candidate future addition. |
| T2.7 | Sensor compatibility (RealSense + Xtion) | Low | High | **Done** | See §3.4. Adds `depth_near_clip_m` (default `0.0` → RealSense behaviour preserved bit-identical) and `near_clip_bbox_min_frac` (default `0.05`) knobs to `extract_bbox_depths`. When the sensor has a hard near-clip dead zone and a bbox occupies enough of the frame to plausibly be a close person, missing-depth resolves to `0 m` (max proximity) instead of `d_safe` (proximity 0), so RiskAM no longer inverts the safety direction on Xtion-class sensors. Depth viz upper bound also expands to `max(d_safe, p95 of valid pixels)` so the visualisation is informative on sensors whose working range exceeds `d_safe`. Tests pin the RealSense default. Threaded through the ROS node + `riskam_config.yml`. Surfaced from soundness-testing the offline pipeline against cs_robocup_2023 (Xtion). |

### Tier 3 — Polish, documentation, scientific presentation

| # | Item | Effort | Impact | Status | Notes |
|---|------|--------|--------|--------|-------|
| T3.1 | ARM64 + AMD64 Dockerfiles | Low | High for deployment | **Pending** | SamXL already has ARM64 variant. |
| T3.2 | Config param validation and better error handling | Low | Medium | **Done** | Weight-sum validation in `riskam_node` (all four weights), broad frame-processing try/except, and `FrameInputs.validate()` covering the full required-input set (RGB + depth with shape checks) all landed. The originally-proposed "depth-topic-unavailable fallback" was deliberately rejected during T3.3.3: silent zeroing of the dominant proximity sub-score is a safety hazard, not a degradation mode. RGB + depth are required inputs; missing either raises a clear `ValueError` at the pipeline boundary. See §4.10 for the full sub-score contract. |
| T3.3 | Evaluation framework refactor | — | — | **Partial** | Expanded into T3.3.1–T3.3.12 below (see §4.9). 9 of 12 sub-items done, 1 partial (T3.3.3 cmd_vel data half — blocked on working with the data), 2 pending (T3.3.3 data half + T3.3.12 CI regression gate — both need a baseline run to compare against). The framework is complete for data-independent work; the remaining items gate on data access. |
| T3.3.1 | Load offline depth in experiments | Low | **Very high** | **Done** | Depth extraction re-enabled in `extract_cs_robocup.py` (raw uint16 mm `.npy`); `CSRobocup2023DepthIndex` does nearest-neighbour RGB↔depth lookup via binary search; `experiments.run_experiment` now passes `depth_image_m` to `featextr`. Covered by `tests/test_cs_robocup_depth_index.py`. |
| T3.3.2 | Wire `approach` sub-score into `RiskScorer` | Low | **High** | **Done** | `w_approach` added to `RiskScorer.score()` (default 0.0 to preserve live behaviour pending T3.3.11 calibration); declared as ROS param + validated in the weight sum + plumbed through `riskam_node` and `riskam_config.yml`; offline experiments enable ByteTrack (`track_bboxes=True`) and reset `humandet._velocity_history` per run via the new `reset_velocity_history()` helper; sweep extended to alternate `w_approach ∈ {0.0, 0.15}` so the ablation is visible. Covered by 4 new tests in `test_score.py::TestRiskScorerApproach`. |
| T3.3.3 | Sub-score input contract + offline evaluability of path-aware trajectory | Medium | High | **Partial** | Contract landed (see §4.10). Path-projection math lives in `riskam/ml/subscores.compute_x_offset`; live + offline paths share the same orchestrator (`featextr.extract(FrameInputs(...))`). Status (ACTIVE/FALLBACK/UNAVAILABLE) flows through to the diagnostics topic. Remaining: bag-backed `/cmd_vel` extraction + nearest-neighbour index so offline path-aware x_offset can be ACTIVE; blocked on working with the data. |
| T3.3.4 | Expanded classification + regression metrics | Low | High | **Done** | `riskam/eval_metrics.py`: per-class precision/recall/F1/support, 4×4 confusion matrix, macro/micro averages (micro ≡ accuracy), MAE/RMSE vs class-centre targets. Exposed via `classification_report(y_true, y_pred_continuous)` and merged into each experiment's `results.json` under `classification`. Per-run summary line printed at completion. 18 unit tests in `tests/test_eval_metrics.py`. |
| T3.3.5 | Cross-run aggregation + ranked reports | Medium | High | **Done** | `riskam/eval_summary.py`: walks `exp_results/<dataset>/<run>/<params_slug>/results.json`, pools confusion matrices cell-wise across runs per config, recomputes per-class / macro / micro metrics from the pooled CM (statistically correct), sample-weights MAE/RMSE across runs. Emits `summary.json` (machine-readable) + `summary.md` (ranked top-K table + per-run breakdown of the best config). CLI at `scripts/summarize_experiments.py` with `--rank-metric {macro_f1,accuracy,mae,rmse} --top-k N`. 14 unit tests in `tests/test_eval_summary.py`. Plots deferred to later (not a blocker for the paper numbers). |
| T3.3.6 | Config-driven parameter sweep | Low | Medium | **Done** | `riskam/sweep_config.py` + `configs/sweeps/default.yaml`: weight tuples (coupled, sum to 1 ± 0.01) plus independent `gaze_sigma_yaw` / `gaze_sigma_pitch` axes; `SweepConfig.iter_experiments()` yields the Cartesian product. Hardcoded grid removed from `experiments.py`. CLI gains `--sweep-config path/to/foo.yaml`; default path is `configs/sweeps/default.yaml`. Weight-sum validation, unknown-key detection, missing-field errors all raise with source path in the message. 15 unit tests in `tests/test_sweep_config.py`. Extension axes (`d_safe`, `crowd_alpha`) are trivial follow-ups once ablation dictates; deferred until needed. |
| T3.3.7 | Feature-extraction caching | Medium | High | **Done** | `riskam/feature_cache.py` caches `CachedFrameFeatures` (bboxes, keypoints, track_ids, per-bbox depths, depth_viz) under `feature_cache/<dataset>/<run>/<model_sha>/<frame_stem>.npz`. `featextr` split into `extract_primitives` (parameter-independent, cacheable) and `extract_features` (parameter-dependent, cheap), with `extract_with_cache` as the cache-aware orchestrator. `run_experiment(use_cache=True)` by default; CLI `--no-cache` disables. Hit/miss stats recorded in `results.json` under `feature_cache`. Assumption documented: ByteTrack's temporal coupling means the cache is most useful after a full uncached pass populates it. 10 unit tests in `tests/test_feature_cache.py`. |
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
| T3.6 | API documentation (docstrings, README) + user deployment guide | Low | Medium | **Partial** | README fully rewritten against `riskam_config.yml` as the source of truth (removed stale `depth_gamma` / `gaze_bounds`; added `w_approach`, `d_safe`, `crowd_alpha`, `n_frames_aggregate`, all gaze sigmas, `cmd_vel_topic`, `sync_slop`, `topic_approach`, and the `/riskam/diagnostics` subscore-status contract). Python example corrected to the current `featextr.extract(FrameInputs(...))` API. Dedicated "For deployers" section (RGB+depth required, which params to tune) and "For researchers" section (full eval-framework toolchain + pointer to §4.9 / §4.10). Pending: full API docstring pass (module-level + public-function docstrings beyond the current state) and a standalone `docs/user_deployment_guide.md` if the EU deliverable template requires it. |
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
- **2026-04-22** — T3.3.7 landed. New module `riskam/feature_cache.py`: `CachedFrameFeatures` (bboxes, keypoints, track_ids, per-bbox depths, depth_viz) round-trips via `np.savez_compressed`; `FeatureCache` keyed on `(root, model_sha, dataset)` with layout `<root>/<dataset>/<run>/<model_sha>/<frame_stem>.npz`. `featextr` split into `extract_primitives` (parameter-independent, cacheable) and `extract_features` (parameter-dependent, cheap); `extract()` now composes both, `extract_with_cache(...)` wraps them with a disk cache. `experiments.run_experiment(use_cache=True)` by default; `scripts/run_experiments.py` gains `--no-cache`. Hit/miss stats recorded in `results.json["feature_cache"]` and printed in the per-experiment summary. Subscore change: `compute_proximity` now takes per-bbox depths (previously took the full depth image + bboxes), so it works the same whether depths were computed fresh or loaded from the cache — one existing test updated to the new signature. `feature_cache/` and `exp_results/` added to `.gitignore`. Assumption documented in the module docstring: ByteTrack's temporal coupling means the cache is most useful after a full uncached pass populates it; mixed hit/miss within a single run can desynchronise track IDs. 10 new tests in `tests/test_feature_cache.py`; full suite passes (157 tests).
- **2026-04-22** — T3.6 partial: README fully rewritten against `riskam_config.yml` as the source of truth. Previous README listed `depth_gamma` (removed in T1.1 with MiDaS) and `gaze_bounds` (removed in T1.3); copy-paste deployment would have produced a silent no-op. New README: summary architecture table, a "Hard requirements" section (RGB + depth required, not optional; cmd_vel + tracking are optional enhancers), complete parameters reference grouped by concern (topics, weights, proximity, gaze, temporal aggregation, bagger), published-topics table including `/riskam/diagnostics` and the per-subscore status fields, a "For deployers" section scoped to physical params only (no sweep required for deployment), and a "For researchers" section mapping each eval-framework CLI (`extract_ros2_dataset.py`, `generate_split.py`, `run_experiments.py --sweep-config --split --no-cache`, `summarize_experiments.py --bucket`, `reeval_experiments.py`) to its purpose. The in-README Python example now uses the current `featextr.extract(FrameInputs(...))` contract and was smoke-imported to confirm the example compiles. Remaining for T3.6: API-level docstring pass across public modules, and a standalone `docs/user_deployment_guide.md` if the EU deliverable template requires it; full suite passes unchanged (157 tests — docs-only change).
- **2026-05-06** — T2.7 landed. Soundness-testing the offline pipeline against cs_robocup_2023 (Xtion sensor) revealed three coupled symptoms (uniform-grey depth viz, score collapse on close-up faces, and wrong "closest bbox" markers) that traced to a single root cause: the legacy missing-depth fallback assigns `d_safe` (proximity 0) and the depth viz normaliser cliffs at `d_safe`, both of which were calibrated for RealSense and inverted the safety direction on structured-light sensors with a hard near-clip dead zone. Fix introduces two physical knobs in `riskam/ml/depth.py`: `depth_near_clip_m` (default `0.0` keeps the legacy fallback intact for RealSense / SamXL) and `near_clip_bbox_min_frac` (default `0.05`); when both fire, `extract_bbox_depths` returns `0 m` (max proximity) for the safety-conservative "person in the dead zone" case. `depth_to_visualization` now uses `max(d_safe, p95 of valid pixels)` as the upper bound so the viz stays depth-varied on Xtion-class sensors while remaining bit-identical when all valid pixels are within `d_safe`. Threaded through `featextr.{extract_primitives,extract,extract_with_cache,extract_human_risk_awareness_features}`, the ROS node, and `riskam_config.yml`. 9 new tests in `tests/test_depth.py::TestNearClipFallback` plus 2 new viz tests pin both the RealSense default (regression guard for SamXL) and the Xtion-like depth-varied behaviour; full suite passes (166 tests). Documented under §3.4 above.
- **2026-05-06** — T2.7 visualisation follow-ups. Two further symptoms surfaced after the algorithmic fix landed: (i) close-fallback bboxes still rendered as black blobs in the depth-overlaid view because `depth_to_visualization` paints zero-pixels black, contradicting the "max proximity" decision the close-fallback just made; (ii) every bbox came out yellow because the gaze-driven colour logic in `riskam/visualization.py` was a strict-equality check against `{0, 1}` — a leftover from the pre-T1.3 thresholded gaze score that's now a continuous Gaussian and never lands on either value. Fix (i): new helper `riskam.ml.depth.apply_close_fallback_to_viz` paints any bbox whose assigned depth is exactly `0.0` (the unique close-fallback sentinel — real measurements are filtered by `DEPTH_MIN_M=0.1`) to 255 in `depth_viz`; called from `featextr.extract_primitives` after the bbox-depth pass. Fix (ii): bbox colour switched to a continuous red↔white gradient by gaze score (`gaze=0` → BGR `(0,0,255)`, `gaze=1` → BGR `(255,255,255)`, intermediate → linearly interpolated pinks), with a thicker black outline drawn underneath so the box stays visible on red walls / shirts and on overexposed white backgrounds. README usage section updated to describe both. 5 new tests for the viz painter (`tests/test_depth.py::TestApplyCloseFallbackToViz`) plus a new `tests/test_visualization.py` (5 tests) pin the gradient endpoints, mid-point monotonicity, out-of-range clamping, and outline presence; full suite passes (176 tests).
- **2026-05-06** — T2.7 visualisation follow-up #3: bbox-only depth blend. Soundness testing surfaced that the whole-frame depth-fog blend (image × (1 − soft_weight) + grey-overlay × soft_weight) was too brutal on Xtion data — with 50–85% of pixels carrying no valid depth, almost every pixel pulled close to the dark-grey overlay and the scene context washed out into uniform grey. Since the only thing this module acts on is detected humans, the depth gradient outside bboxes is information-poor: scoping the blend to bbox interiors only keeps the "close = vivid, far = grey" gradient where it matters and leaves the rest of the scene untouched. Implementation: build a bbox mask in `riskam/visualization.visualize_risk` and multiply `soft_weight` by it before blending; `lap>=0.5.12` also added to `pyproject.toml` deps because Ultralytics ByteTrack soft-imports it on first `model.track(...)` call and the auto-pip-install fails inside a uv-managed venv (no `pip` shipped). 3 new tests in `tests/test_visualization.py::TestBboxOnlyDepthBlend` pin the new contract (no fog without bboxes, no fog outside bboxes, blend still applied inside); full suite passes (179 tests).
- **2026-05-06** — T2.7 architecture cleanup: introduced ``riskam/platforms.py`` with two-level ``DepthSensor`` → ``RobotPlatform`` dataclass abstractions, and migrated the per-dataset sensor / safety knobs into named platform presets. The previous flat ``DATASETS["..."]["depth_near_clip_m"]`` / ``["near_clip_valid_frac_max"]`` keys were a category error: those are sensor properties, and ``d_safe`` (just-introduced calibration knob) is a robot property — neither is genuinely a *dataset* property, even though datasets are recorded by hardware. Two presets land: ``RIDGEBACK_D435`` (SamXL — `d_safe=1.5`, ``REALSENSE_D4XX`` sensor with close-fallback off, matching the ROS YAML defaults bit-identically) and ``TIAGO_XTION`` (cs_robocup_2023 — `d_safe=2.5`, ``PRIMESENSE_XTION`` sensor with close-fallback enabled). ``DATASETS["cs_robocup_2023"]["platform"] = TIAGO_XTION`` replaces the two flat keys; offline callers (sandbox, video viz, experiments.py) read ``platform.d_safe_m``, ``platform.sensor.near_clip_m``, ``platform.sensor.valid_frac_max``. Live ROS path is untouched — the node still consumes individual scalars from ``riskam_config.yml`` so deployers can tune any field without inventing a new platform. README "For deployers" section gained a per-platform recommended-values table + custom-platform derivation guide. The cs_robocup_2023 ``d_safe`` bump from 1.5 → 2.5 m (TIAGo's actual stopping distance + Xtion's deeper usable range) directly addresses the "score feels VERY low" feedback on mid-range frames: 1537360295.585 (person at 1.55 m, gaze 0.83) lifts from 0.093 to 0.359; close-fallback cases unchanged. 9 new tests in ``tests/test_platforms.py`` pin the preset values + the SamXL ROS-YAML alignment regression guard. Full suite passes (190 tests).
- **2026-05-06** — T2.7 robustness extension: close-fallback condition generalised from "exactly zero valid pixels" to "valid-pixel fraction at or below ``near_clip_valid_frac_max``". Surfaced from soundness testing on cs_robocup_2023 RB_08 frames 14 s apart: at 1537360299.612 a person walking close to the robot occupied 56% of the frame but only 1.3% of bbox pixels carried valid depth — those few stray pixels were background bleed-through (whole-frame p50 = 1.85 m), not the person, but the percentile path trusted them and reported p10 = 1.72 m → proximity 0. 14 s later at 1537360313.658 the same person had walked into the Xtion's near-clip dead zone, the bbox went to 0% valid, and the close-fallback fired correctly → proximity 1. The discontinuity was an algorithmic robustness gap, not a calibration question: when valid pixels are sparse on a structured-light sensor, they're almost certainly noise, not signal. New parameter ``near_clip_valid_frac_max`` (default `0.0`, preserving RealSense pre-T2.7 behaviour bit-identically) lets deployers extend the dead-zone fallback to "mostly empty" bboxes; the cs_robocup_2023 dataset entry sets it to `0.05`. Threaded through `extract_bbox_depths` / `extract_bbox_proximities`, all four `featextr` entry points, the ROS node + `riskam_config.yml`, the offline scripts (sandbox, video viz, experiments). 4 new tests in `tests/test_depth.py::TestNearClipFallback` pin (a) default = strict-zero condition (RealSense regression guard), (b) sparse + threshold = fallback fires, (c) dense valid pixels = percentile path even with the threshold set, (d) ``near_clip_bbox_min_frac`` size guard still applies. Frame 299's big bbox now scores proximity 1.0 (was 0.0); sandbox risk jumps 0.195 → 0.569; full suite passes (181 tests).
- **2026-05-06** — T2.7 visualisation follow-up #4: bbox interior switched from grey-fog-by-depth to **red-by-proximity**. The grey-fog gradient was technically scoped to bboxes (#3) but still semantically off: it communicated "darkness = far", inverted from the natural "red = close = danger" semantic that matches what the score actually means. Bbox interior is now tinted red with `α = proximity · 0.7`, uniform per bbox, driven by the per-bbox proximity sub-score (not raw per-pixel `depth_viz`). Two consequences: (i) the visualisation tracks the score that actually drives risk, not the noisy underlying depth field — so Xtion sensors' heavy zero-fill no longer splotches the overlay; (ii) the legacy `apply_close_fallback_to_viz` helper (added earlier this session) became dead code — `visualize_risk` no longer reads `depth_viz` at all — and was removed along with its 5 tests. The `depth_viz` parameter was dropped from `visualize_risk`'s signature and the four callers (`riskam_node`, `experiments`, `featextr_sandbox`, `test_run_with_video`) updated. `depth_viz` is still computed and cached in `CachedFrameFeatures` for now (cleanup deferred — would need a cache schema bump). 6 new tests in `tests/test_visualization.py::TestProximityRedInterior` pin the contract (proximity 0 → translucent, 1 → strongly red, monotonic, scoped to bbox, close-fallback renders strongly red end-to-end); full suite passes (177 tests).
- **2026-04-22** — End-of-session plan cleanup against reality. Reviewed §5, §6 legend, T2.4, T2.6, T3.2, and T3.3 parent-row notes against code; updated to match. **§5** stack-level changes now mark MiDaS removal, the `numpy<2.0.0` unpin, and MediaPipe removal as landed, and flag `nav_msgs` in `riskam_ros/package.xml` as a dead dep (path-aware x_offset uses `geometry_msgs/Twist`, not `nav_msgs/Odometry`). **T2.4** note expanded to record the per-subscore status / WARN-level escalation added in T3.3.3. **T2.6** note refreshed: 14 test files / 157 tests rather than the original three. **T3.2** moved Partial → Done — the originally-proposed depth-unavailable fallback was deliberately rejected as incompatible with the required-input contract from T3.3.3, and `FrameInputs.validate()` provides the full validation. **T3.3** parent row moved Pending → Partial (9/12 sub-items done). Commit `08a80f7` (approach sub-score publisher + bagger topic; landed between T3.3.2 and T3.3.3 as a follow-up to T3.3.2) is logged here explicitly. **Session totals.** Today added 12 commits on top of `4fc6711`: T3.3.1 + audit, T3.3.2, the approach publisher follow-up, T3.3.3 (refactor half), T3.3.4 + T3.3.9, T3.3.5, T3.3.10, T3.3.8, T3.3.6, T3.3.11, T3.3.7, and T3.6 partial. **Open items blocked on data:** T3.3.3 cmd_vel half (verify `/cmd_vel` is in the cs_robocup_2023 bags first), T3.3.12 CI regression gate (needs a baseline run to compare against), T3.4.5 re-annotation (pure human work). **Code-only remaining (next data-free session):** T3.1 (AMD64 Dockerfile), T3.4.1–T3.4.4 (annotator improvements), T3.6 docstring pass + standalone deployment guide, T3.7 Sphinx (optional).
