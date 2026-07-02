# RiskAM — Implement the paper-plan risk metrics and evaluate on cs_robocup_2023

## Context

`docs/private/paper-plan.md` diagnoses the deployed risk score (`riskam/score.py`: weighted sum of proximity/gaze/x_offset/approach, breakpoints [0, 0.3, 0.6, 1]) as ad hoc — no units, additive fusion of incommensurable terms, hazard conflated with awareness — and proposes two composable replacements:

- **Direction A**: physically-grounded kinematic hazard (distance, closing speed, bearing → time-to-closest-approach `t_cpa`, predicted miss distance `d_min`) multiplicatively modulated by awareness: `risk = hazard · (1 + β(1−awareness))`. Awareness scales hazard, never creates it.
- **Direction B (B1)**: proportional-odds ordinal regression fit on the annotations — weights become coefficients with SEs/p-values, and probabilistic output unlocks proper scoring rules (Brier, log-loss, reliability, RPS) on the existing val/test split.

This work **extends** the repo — the deployed ROS pipeline, `RiskScorer`, experiments harness, and sweep configs stay untouched — then runs both metrics offline on the 31,507 annotated frames and produces a comparison against the current metric (M0).

Key facts from exploration:

- Feature cache (`riskam/feature_cache.py`) stores per-frame primitives (bboxes, keypoints, ByteTrack IDs, per-bbox metric depths) — everything needed to derive kinematics offline. Currently **empty**; one YOLO11n-pose populate pass over labelled frames required.
- Untracked `scripts/metric_probe.py` is a prior prototype (check/populate/analyze, frame-stem timestamps, OrderedModel fits M1/M2). It gets **absorbed into the new runner and deleted**.
- **New data finding**: bags contain dense `/mobile_base_controller/odom` in every run (2.5k–16k msgs) and `/xtion/rgb/camera_info` (exact intrinsics); `/cmd_vel` is sparse/absent (0 msgs in RB_01/06/07). Full Direction A with real robot velocity is feasible; T3.3.3's "blocked on data" resolves to **odom**, not cmd_vel.
- Canonical stratified within-run val/test split: 22,055 val / 9,452 test, seed 42 (`ml_datasets/cs_robocup_2023/split.json`). Severe class imbalance (class 3 = 19,116).
- `statsmodels 0.14.6` already in the `experiments` dep group (the uncommitted pyproject change — keep it). No pandas needed (stdlib csv); scipy transitive.
- `humandet` velocity history uses `time.monotonic()` + module globals — wrong offline; new tracker takes explicit frame timestamps, instantiated per run.
- Depth sentinels from `riskam/ml/depth.py`: `0.0` = dead-zone close-fallback, exactly `d_safe` = no-valid-pixels far-fallback. Both are censored values, not measurements — must not pollute velocity fits.

## Design

Five new `riskam/` modules + one runner script + a small bag-extraction extension. Only `riskam/platforms.py` (additive, defaulted fields) and `riskam/data/` (new loaders) among existing production files are touched.

### New files

**`riskam/kinematics.py`** — Direction A core (pure numpy, no ML deps):
- `CameraModel(fx_px, cx_px)` with `from_intrinsics(fx, cx)` and pinhole fallback `from_hfov(hfov_deg, image_width)`; `bearing_rad(u) = atan2(u − cx, fx)`.
- `KinematicParams(d_safe_m, tau_s=2.0, beta=1.0, footprint_radius_m, window=8, max_gap_s=1.0, min_span_s=0.05, v_eps_ms=0.05)` — defaults imported from new `platforms.py` referent constants.
- `PlanarTwist(t_s, vx_ms, vy_ms, wz_rads)` — odom sample type.
- `KinematicTracker(params, camera)` — per-track deque of `(t, x, y)` planar positions in robot frame (`x = d`, `y = −d·(u−cx)/fx`), **explicit timestamps**. `update(t_s, bboxes, bbox_depths_m, track_ids, ego=None) -> list[PersonKinematics]`; `reset()`. One instance per run.
- **Key physics point (corrects paper-plan §2A)**: the camera measures *relative* position directly, so the per-track planar derivative (`vx, vy = polyfit` over the window) already **is** `v_rel` with ego-motion folded in. Odom is NOT subtracted in the full path; it serves (a) the single-observation fallback (static-human assumption: `v_rel = −(v_ego + ω×p)`), (b) a `v_robot_speed` severity feature for B1, (c) a consistency check.
- CPA math: `t_cpa = −(p·v)/|v|²`, `t_cpa* = max(0, t_cpa)`, `d_min = |p + v·t_cpa*|`,
  `hazard = clip(1 − max(0, d_min − r_footprint)/d_safe, 0, 1) · exp(−t_cpa*/τ)`,
  `fuse_risk(hazard, awareness, beta) = clip(hazard·(1 + β(1−awareness)), 0, 1)`. Awareness = existing gaze sub-score.
- Status ladder per person: `FULL` (≥2 obs) → `STATIC_ODOM` (1 obs + ego) → `STATIC` (1 obs, no ego: `d_min = r`, `t_cpa = 0` — degrades to exactly today's proximity, per paper-plan §3) → `DEAD_ZONE` (depth 0.0 sentinel: hazard = 1, observation excluded from history). `d == d_safe` far-fallback also excluded from history (censored). Track gap > `max_gap_s` resets that track's history; `|v_rel| < v_eps` → static handling; `t_cpa < 0` (receding) → `d_min = r` by the formula itself.

**`riskam/ordinal.py`** — Direction B1:
- `fit_ordinal(X, y, feature_names) -> OrdinalFit` using `statsmodels OrderedModel(distr="logit")`, BFGS (lbfgs retry), val-set standardization stored in the artifact.
- `OrdinalFit` dataclass: feature names, μ/σ, coefficients {coef, se, z, p}, thresholds α₁..₃, llf/ll_null/AIC/n/converged; `predict_proba` reimplemented from stored params (numpy only — evaluate step doesn't import statsmodels); `to_json`/`from_json`.
- `lr_test(full, reduced)` — likelihood-ratio tests for the ablation story.

**`riskam/proba_metrics.py`** — calibration metrics, mirrors `eval_metrics.py` style (separate module so the baseline stays untouched): `brier_score`, `log_loss_score`, `ranked_probability_score` (ordinal-proper), `cumulative_auc` (AUC of P(y≥k), k=1..3), `reliability_table` (bins + ECE), `spearman_rho`, `probabilistic_report` bundle.

**`riskam/scene_table.py`** — builds the persisted per-frame scene-level feature table from the cache: walks labelled frames per run in stem-timestamp order, fresh `RiskScorer` + `KinematicTracker` per run, emits one row per frame with columns: run, frame, t_s, bucket, gt, n_humans; M0 risk/class + its max-risk person's four sub-scores; kinematics (d, bearing, closing, tan_speed, t_cpa, d_min, inv_ttc); Direction-A channels (hazard, awareness, risk_a, kin_status); v_robot_speed, ego_available. CSV via stdlib (NaN-safe). **Scene aggregation = max-fused-risk person** (labels are scene-level worst-case; same "worst actor" convention as M0; fixes the probe's closest-person mis-ranking of a fast closer vs a static bystander). No-human frames get the min-hazard sentinel row.

**`scripts/metric_lab.py`** — runner absorbing `metric_probe.py` (which is deleted). Subcommands:
```
check | populate [--runs ... --limit-per-run N --device cpu|mps] |
build-table [--force --no-odom] | fit [--force] | evaluate [--figures] | all
```
- `populate`: always all labelled frames per run in stem order (no split filter — split-subset population would desync ByteTrack IDs vs the cache, per the `feature_cache.py` header warning); `reset_velocity_history()` per run; `--device mps` moves the YOLO model post-import.
- Models fit/evaluated: `m0` (deployed scorer, deterministic), `m0c` (ordinal head on m0 score — makes the baseline comparable on proper scoring rules), `a_raw` (risk_a, fitting-free physics evidence: Spearman + hazard plots), `a_cal` (ordinal on risk_a), `b1_sub` (ordinal on the four sub-scores), `b1_kin` (ordinal on [d, closing, tan_speed, d_min, inv_ttc, awareness, n_humans] — this IS the A+B layered variant).
- Protocol: fit on val (unweighted MLE — reweighting would deliberately miscalibrate; imbalance handled on the reporting side with macro-F1/per-class recall + proper scoring rules), apply frozen fits to test **once**. LR tests: b1_sub vs drop-approach; b1_kin vs drop-awareness, drop-n_humans; each vs intercept-only.
- Outputs under `exp_results/cs_robocup_2023/metric_lab/` (sibling of the all/val/test bucket dirs; contains no `results.json` so the existing summarizer never sees it): `scene_table.csv`, `table_meta.json` (model SHA, params, camera source, ego availability, provenance), `fits/*.json`, `report_val.json`, `report_test.json`, `lr_tests.json`, `comparison.md` (headline table per model × split), `figures/` (reliability diagram, per-feature Spearman, hazard-by-class violin, kin_status histogram).

### Modified files

**`riskam/platforms.py`** (additive, defaulted): `DepthSensor.rgb_hfov_deg: float | None = None` (Xtion 58.0, RealSense D435 69.0); `RobotPlatform.footprint_radius_m: float = 0.0` (TIAGo 0.27, Ridgeback 0.48); module constants `TAU_REACTION_S_DEFAULT = 2.0` (perception-reaction window), `BETA_UNAWARE_DEFAULT = 1.0` (oblivious human doubles hazard ≈ SSM worst-case posture). Every knob has a physical referent.

**`riskam/data/extract_cs_robocup.py` + `scripts/extract_ros2_dataset.py`** — new fast aux pass `extract_cs_robocup2023_aux()` (no image decoding): per run writes `raw_dataset/RB_0X/odom.csv` (t_s, linear_x, linear_y, angular_z from `/mobile_base_controller/odom` twist) and `camera_info.json` (width, height, fx, cx from `/xtion/rgb/camera_info` first msg). Run on macOS via the existing Docker wrapper `scripts/extract_ros2_dataset_macos.sh`.

**`riskam/data/cs_robocup_2023.py`** — `CSRobocup2023OdomIndex(run, tolerance_s=0.25)` (nearest-neighbour twist lookup, mirrors the depth index) and `load_camera_model(run, image_width, sensor)` (camera_info.json → exact intrinsics; else FOV fallback). Missing files degrade gracefully — nothing hard-depends on aux extraction. Documented approximation: odom twist is base-frame, camera assumed aligned with base forward (TIAGo head pan ignored; /joint_states correction noted as future refinement).

**Docs**: new `docs/experimental-metrics.md` (Direction A math + degradation ladder, B1 protocol, runner usage, artifact layout), linked from `docs/README.md` + pointer in `docs/scoring.md`; status updates in `docs/private/paper-plan.md` §5 (both "cheap validation" items done; odom/camera_info finding recorded); `docs/changelog.md` entry; `docs/improvement-plan.md` T3.3.3 note (velocity source = odom, unblocked).

**Tests** (synthetic, no GPU/dataset, existing style): `tests/test_kinematics.py` (head-on closer t_cpa≈d/v & d_min≈0; lateral passer d_min≈offset — the F4-emergence test; receding; STATIC vs STATIC_ODOM; ω-only ego; dead-zone sentinel keeps history clean; gap reset; `fuse_risk(0, ·) == 0` pathology-death property; intrinsics-vs-FOV bearing agreement at centre), `tests/test_ordinal.py` (recover known coefficients from simulated proportional-odds data; noise feature p≈1; JSON roundtrip predict_proba matches statsmodels to 1e-8), `tests/test_proba_metrics.py` (perfect forecast → 0; uniform closed-form; RPS ordinal sensitivity; bins partition n), `tests/test_scene_table.py` (tmp-dir FeatureCache with synthetic frames — pattern from `tests/test_feature_cache.py`; per-run isolation; sentinel row; CSV roundtrip; bucket assignment), extend `tests/test_platforms.py`.

## Implementation steps (each with verification)

1. `platforms.py` fields/constants → `uv run pytest` (all green proves additive change broke nothing).
2. `riskam/kinematics.py` + tests → `uv run pytest tests/test_kinematics.py`.
3. `riskam/proba_metrics.py` + tests.
4. `riskam/ordinal.py` + tests (`uv run --group experiments pytest tests/test_ordinal.py`).
5. `riskam/scene_table.py` + tests.
6. `scripts/metric_lab.py`; delete `scripts/metric_probe.py` → smoke: `check`, then `populate --runs RB_01 --limit-per-run 50` + `build-table` → sane 50-row CSV.
7. Aux extraction (`extract_cs_robocup2023_aux`, odom/camera loaders) → run Docker wrapper → verify odom.csv row counts ≈ bag metadata message counts; camera_info fx vs 58°-pinhole cross-check (Xtion VGA fx ≈ 570 px expected). `check` reports ego availability per run.
8. Full populate: `uv run scripts/metric_lab.py populate --device mps` (AC power; `PYTORCH_ENABLE_MPS_FALLBACK=1`; est. ~20–40 min MPS, ~1–1.5 h CPU fallback; ~2–6 GB in gitignored `feature_cache/`; per-run retry granularity via `--runs`).
9. `build-table` → `fit` → `evaluate --figures` → eyeball `comparison.md`: M0 row should reproduce known baseline behaviour (cross-check vs one `run_experiments.py` cell — doubles as the no-regression proof); coefficient signs sane (proximity/closing +, gaze −).
10. Docs + changelog; final full `uv run pytest`.

## Verification (end-to-end)

- Unit: `uv run --group experiments pytest` — all new + existing tests green.
- No-regression: existing harness untouched by construction; step 9's M0 cross-check against a fresh `run_experiments.py` cell confirms identical baseline numbers.
- Deliverable check: `exp_results/cs_robocup_2023/metric_lab/comparison.md` contains the M0 / m0c / a_raw / a_cal / b1_sub / b1_kin × {val, test} table with accuracy, macro-F1, MAE, Brier, log-loss, RPS, ECE, Spearman; `lr_tests.json` answers "does approach/awareness matter" with p-values; reliability figure renders.

## Risks

- MPS op gaps in ultralytics pose/ByteTrack → env fallback flag; CPU is the guaranteed path.
- Sparse labelled-frame timing (depth-missing gaps) may push many rows to STATIC status — made visible via the kin_status histogram in reports, not silent.
- Class-3 dominance can flatten argmax metrics across models — by design the probabilistic metrics + LR tests carry the story.
- `d == d_safe` far-fallback detected by float equality — heuristic; the clean fix (validity flag in the cache format) would invalidate the cache and touch the deployed path, explicitly deferred.
- Mid-run populate resume desyncs ByteTrack IDs across the resume boundary — retries at per-run granularity.
