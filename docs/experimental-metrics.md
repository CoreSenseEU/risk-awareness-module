# Experimental metrics — kinematic hazard and ordinal calibration

Implementation of the two metric-redesign directions explored in
[`private/paper-plan.md`](private/paper-plan.md). Everything here is
**offline research tooling**: the deployed pipeline (`riskam/score.py`,
the ROS node, the sweep harness) is untouched, and nothing below runs on
the robot. The entry point is `scripts/metric_lab.py`.

## Direction A — kinematic hazard × awareness (`riskam/kinematics.py`)

The deployed score adds incommensurable [0, 1] sub-scores; the kinematic
metric instead derives one physical hazard and lets awareness *scale* it:

```
p  = (x, y)                     planar position, robot frame (x fwd, y left)
v  = d p / d t                  relative velocity, fit over a per-track window
t_cpa  = max(0, −(p·v)/|v|²)    time to closest approach            [s]
d_min  = |p + v·t_cpa|          predicted miss distance             [m]

h(t)   = clip(1 − max(0, |p + v·t| − r_footprint)/d_safe, 0, 1) · exp(−t/τ)
hazard = max over t ∈ [0, t_cpa] of h(t)      # trajectory hazard
risk_A = clip(hazard · (1 + β·(1 − awareness)), 0, 1)
```

The hazard scores the **whole predicted pass**, not just the
closest-approach moment: scoring only `t = t_cpa` is discontinuous at
`v → 0` (a person standing at 2 m scores their proximity shortfall while
the same person drifting in at 6 cm/s has a huge `t_cpa` and scores ≈ 0),
which made the metric flicker with velocity noise. The trajectory max is
continuous — with no velocity the max sits at `t = 0` and reduces exactly
to the static proximity shortfall.

Two smoothing stages keep the output steady without delaying alarms
(the kinematic counterpart of the deployed scorer's 5-frame mean):

- **per-track channel EMA** (`ema_tau_s`, default 0.5 s, real-timestamp
  based) on hazard and awareness, fused after smoothing;
- **scene-level release** (`release_tau_s`, default 0.5 s): the scene
  risk is `max(instant, supported · exp(−dt/τ_release))` — rise is always
  instant, decay bridges single-frame detector dropouts (the dominant
  flicker source: a person YOLO misses for one 33 ms frame is still
  there). The release is **evidence-bounded** (`hold_max_s`, default
  1.0 s — the same horizon as the tracker's `max_gap_s`): it extrapolates
  at most that far past the last detection-backed value, so a person who
  genuinely leaves does not leave behind a multi-second ghost risk in an
  empty frame. The video overlay tags release-carried values with
  `hold`, so memory is never disguised as measurement.

`awareness` is the existing gaze sub-score. `hazard = 0 ⇒ risk = 0`: an
unaware person with no collision geometry accrues nothing (the additive
pathology dies). "In the path" and "approaching" are emergent in `d_min`
and the sign of `p·v` — they are no longer separate weighted sub-scores.

Positions come from cached primitives only: bbox centre-x → bearing via
`CameraModel` (exact per-run `camera_info.json` intrinsics when extracted,
datasheet-HFOV pinhole fallback otherwise), bbox depth → range.

**Ego-motion.** Because the camera measures *relative* position, the
per-track derivative already **is** relative velocity — ego-motion folded
in, no odometry subtraction (this corrects the paper-plan §2A sketch).
Odometry (`odom.csv`, see below) plays three secondary roles: the
single-observation fallback (static-human assumption,
`v_rel = −(v_ego + ω×p)`), the `v_robot_speed` severity feature, and a
consistency check.

**Degradation ladder** (per person, per frame; visible as `kin_status`):

| status | when | behaviour |
|---|---|---|
| `full` | ≥ 2 tracked observations | full CPA extrapolation |
| `static_odom` | 1 observation + odometry | static-human assumption |
| `static` | 1 observation, no ego info | `d_min = r`, `t_cpa = 0` → hazard reduces to exactly the deployed proximity sub-score |
| `dead_zone` | depth sentinel `0.0` (near-clip fallback) | hazard = 1, observation censored from velocity history |

The far-fallback sentinel (depth exactly `d_safe`) is likewise censored
from velocity fits. Track gaps > 1 s reset that track's history.
Timestamps are explicit (frame stems offline) — never wall-clock.

Every knob has a physical referent (`riskam/platforms.py`): `d_safe` =
stopping distance, `τ` = reaction window (`TAU_REACTION_S_DEFAULT` 2.0 s),
`β` = oblivious-human penalty (`BETA_UNAWARE_DEFAULT` 1.0 ≈ the ISO/TS
15066 SSM worst-case posture, applied only when awareness is absent),
`footprint_radius_m`, `rgb_hfov_deg`.

## Direction B1 — ordinal calibration (`riskam/ordinal.py`)

A proportional-odds logit, `logit P(class ≥ k) = x·β − α_k`, fit by MLE on
the **val** bucket of the canonical split (the split module's stated
"fit on val, report once on test" purpose). The β's replace hand-swept
weights and arrive with standard errors and p-values; likelihood-ratio
tests answer "does this feature matter?" directly. Fits are persisted as
self-contained JSON (`OrdinalFit`) whose `predict_proba` needs only numpy.

Class imbalance (61 % class 3) is deliberately **not** reweighted at fit
time — reweighting would miscalibrate the probabilities, and calibration
is the point. Imbalance is handled on the reporting side (macro-F1,
per-class recall, proper scoring rules).

## Evaluation upgrade (`riskam/proba_metrics.py`)

Probabilistic outputs unlock proper scoring rules: Brier, log-loss,
ranked probability score (the ordinal-proper rule), cumulative AUC
(P(y ≥ k) per threshold), reliability tables + ECE, Spearman ρ.
`riskam/eval_metrics.py` (the deployed `results.json` schema) is untouched.

## The lab (`scripts/metric_lab.py`)

Models compared (fit on val where applicable, frozen single report on test):

| id | definition |
|---|---|
| `m0` | deployed scorer + breakpoints (deterministic baseline) |
| `m0c` | ordinal head on the `m0` scalar — baseline made comparable on scoring rules |
| `a_raw` | Direction A fused scalar, no fitting (Spearman/physics evidence) |
| `a_cal` | ordinal head on the Direction-A scalar |
| `b1_sub` | ordinal on the four deployed sub-scores — "are the swept weights right?" |
| `b1_kin` | ordinal on the kinematic features — the A + B layered variant |

```
uv run            python scripts/metric_lab.py check
uv run            python scripts/metric_lab.py populate [--device mps]   # the one GPU step
uv run --group experiments python scripts/metric_lab.py build-table
uv run --group experiments python scripts/metric_lab.py fit
uv run --group experiments python scripts/metric_lab.py evaluate --figures
uv run            python scripts/metric_lab.py video [--runs RB_02] [--fps 10]
```

`video` renders per-run MP4s annotated with the Direction-A channels only —
per person the measured geometry (`d`, predicted miss distance, time to
closest approach), the hazard/awareness/fused-risk values, a box coloured
green → red by fused risk, and an asterisk on the scene's max-risk person.
Ground truth and the deployed metric are deliberately absent, so the videos
support unbiased eyeballing of the new metric. Output:
`exp_results/cs_robocup_2023/metric_lab/videos/RB_0X.mp4`.

`populate` always covers each run's full labelled frame sequence
(ByteTrack IDs are sequence-coupled; retry failed runs whole via
`--runs`). Artifacts land in `exp_results/cs_robocup_2023/metric_lab/`
(invisible to the baseline summarizer — no `results.json` inside):
`scene_table.csv` + `table_meta.json`, `fits/*.json`, `lr_tests.json`,
`report_val.json`, `report_test.json`, `comparison.md`, `figures/`.

Scene aggregation: labels are scene-level worst-case, so Direction A
reports its **max-fused-risk person**; `m0` reports its own max-risk
person, as deployed.

## Findings and known behaviours (2026-07-02 evaluation)

Headline results on the test split (9,451 frames; full tables in
`exp_results/cs_robocup_2023/metric_lab/comparison.md`):

| finding | evidence |
|---|---|
| The **unfitted** physics metric matches — and slightly beats — the annotation-swept baseline on rank agreement | `a_raw` Spearman ρ **+0.758** vs `m0` +0.750; hazard medians by class form a clean staircase (0 → 0.01 → 0.32 → 0.79, see `figures/hazard_by_class_violin.png`) |
| The physics scalar calibrates best | `a_cal` ECE 0.035 vs `m0c` 0.059 |
| The hand-swept weights were roughly right | `b1_sub` coefficients rank proximity (+2.60) ≫ x_offset (+0.53) > gaze (−0.45), all p ≪ 0.001 — same ordering as the deployed 0.7/0.05/0.25 |
| The deployed `approach` sub-score is **degenerate offline** | its `time.monotonic()` history never accumulates in fast replay → constant 0.5, zero variance on all 31,505 frames; `closing_ms` (frame-timestamp based) is the principled replacement |
| **Awareness carries the *opposite* sign in the data** | in `b1_kin`, awareness has a *positive* coefficient (+0.12, p ≈ 9e-13): given the geometry, frames where the person looks at the robot were rated *riskier*. Most plausibly the annotators rated geometry only (close people tend to face the robot). Consequence: **these labels cannot validate the awareness-modulated-SSM hook**; that claim waits on the T3.4.4/T3.4.5 per-channel re-annotation |

### Smoothness (video iteration)

Raw per-frame output was 1.4–2× jumpier than the deployed metric; three
causes, each measured before fixing (mean per-frame |Δrisk| on RB_01:
0.065 → 0.022 vs m0's 0.046; jumps > 0.3: 91 → ~15):

1. **CPA-moment discontinuity** (83–95 % of big jumps): scoring only the
   closest-approach moment collapses to ≈0 for slow approachers while a
   static person scores the full proximity shortfall → replaced by the
   trajectory hazard (continuous at v → 0).
2. **No temporal smoothing** → per-track channel EMA (`ema_tau_s`).
3. **Detector flicker** (46 of RB_01's 60 residual jumps coincided with a
   person appearing/disappearing for single 33 ms frames) → the
   evidence-bounded scene release (`release_tau_s`/`hold_max_s`). The
   bound matters: an unbounded release left ghost risk gliding through
   multi-second *empty* scenes (RB_01 t+16.5 s: 4.3 s of nobody with
   risk decaying 0.61 → 0.008); bounding it to the track-gap horizon
   also *raised* a_raw's ρ (+0.747 → +0.758) because annotators label
   empty frames class 0.

### Known-open: the CPA display numbers are ill-conditioned (not a bug)

The overlay's raw "d / miss / in" line stays deliberately unfiltered and
*feels* jumpy even though the risk is smooth. Measured on consecutive
single-person full-kinematics frames (~40 ms steps): |Δt_cpa| median
0.03 s but **p90 5.2 s**; |Δd_min| median 0.02 m, p90 0.42 m — a heavy
tail, while the underlying measurements are stable (|Δdepth| median
4 mm; per-frame closing-speed wobble ≤ 0.15 m/s p90). The tail is the
nonlinear map, not the sensors:

- **t_cpa = r / closing speed** is a ratio with a small noisy
  denominator: 91 % of big t_cpa jumps occur at closing < 0.3 m/s, and
  jump sizes correlate 0.70 (log–log) with the analytic sensitivity
  r·Δv/v².
- **d_min ≈ r·sin∠(p, −v)** is the angle of a nearly radial vector for
  head-on approach — exactly when miss distance matters most: p90
  |Δd_min| is 1.41 m for near-radial movers vs 0.20 m for tangential.
- Contributing wobble is partly *real*: gait oscillation (the
  10th-percentile bbox depth rides swinging limbs) sampled by a fit
  window shorter than half a stride.

The fused risk is insensitive to this (trajectory max + clip + EMA sit
downstream), matching how TCAS/maritime systems treat CPA estimates —
filter and quantise before display. Options if it ever matters, cheapest
first: quantised display bins ("miss < 0.5 m, ~2 s"), reuse the channel
EMA on the displayed numbers, gait-length velocity window (~1.2 s),
per-track constant-velocity Kalman filter, torso-anchored depth.

## Odometry / intrinsics extraction

`scripts/extract_ros2_dataset.py cs_robocup_2023_aux` (macOS: via
`scripts/extract_ros2_dataset_macos.sh`) writes per run `odom.csv`
(`/mobile_base_controller/odom` twists; `/cmd_vel` is empty in half the
runs) and `camera_info.json` (exact fx/cx). Both files are optional —
their absence degrades to the HFOV fallback and the no-ego ladder rung.
Known approximation: odometry twist is base-frame and the camera is
assumed aligned with base forward (TIAGo head pan ignored;
`/joint_states` correction is a possible refinement).
