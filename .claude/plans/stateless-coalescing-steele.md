# Layer 2 eval revamp (hindsight oracle) + 2024 risk video

## Context

Per `docs/private/paper-plan.md` §Layer 2 and its sequencing, agreement with manual labels stops being the primary evidence. The new backbone is a per-frame **event table** (score channels + hindsight outcomes) per dataset; L2 scores every metric as an **early-warning signal** against a hindsight oracle: "a person actually came within r m within the next T s" (r ∈ {0.5, 1.0, 1.5} m, T ∈ {1, 2, 3} s), computed non-causally from the full recording. Zero manual labels. Headline: ISO/TS 15066 worst-case SSM vs awareness-modulated SSM at matched recall → "X% fewer alarms at equal safety".

Scope decision (user AFK, plan-sequencing default): **L2 fully now on cs_robocup_2023 + cs_robocup_2024, plus the 2024 risk video; L3 (VLM pairwise protocol) is the next iteration** once L2 numbers exist — matches the paper plan's "publishable core exists before any VLM work starts".

**Circularity firewall** (paper-plan hard constraint): the oracle module reads only cache + geometry + future; score channels read only the past. They meet only in the CSV. Enforced by module imports + a firewall test.

## Facts that shape the design (validated on disk)

- Feature cache holds per frame: bboxes, ByteTrack `track_ids` (sequence-coupled — whole-run passes only), `bbox_depths_m` (0.0 = dead-zone sentinel, ==d_safe(2.5) = far sentinel), keypoints. Planar position reconstruction is 2 lines from `kinematics.py:352-357` (needs per-run `camera_info.json` — present).
- Cache coverage: 2023 dense except **RB_01 partial (1300/5579 → whole-run --refresh repopulate)**, **RB_07 empty (~3.4k)**; **RB_04 has no frames on disk → excluded**. 2024: empty, 96k frames → one overnight YOLO populate.
- Non-causal machinery to reuse: `riskam/mocap_gt.py` (`smooth_and_differentiate` savgol, `cpa_vec`). Causal walk template: `riskam/scene_table.py` (frozen legacy — mirror, don't refactor). Populate pattern: `scripts/metric_lab.py` (legacy — mirror, don't extend).
- `experiments.py` transitively loads YOLO at import → dataset wiring must move to a data module, scripts never import `experiments`.

## New/modified files

1. **New `riskam/data/run_datasets.py`** — `RunDataset` registry (raw_dir, runs, platform, depth/odom index classes, camera loader, gt/split paths) for both datasets; `RUN_BASED_DEPTH_INDEXES` moves here (`experiments.py` re-imports it — 2-line change). 2023 runs = glob minus empty (RB_04 auto-excluded), RB_07 included (oracle needs no labels).
2. **New `riskam/hindsight.py`** — the oracle. `decode_frame` (cache → per-person planar d; far sentinel → NaN censored; dead zone → d=0 + flag), `smoothed_scene_min_d` (per-track-segment savgol via mocap_gt helpers, **scene-level min** — robust to ByteTrack ID switches), `hindsight_columns` (windowed future-min via monotonic deque: `future_min_d_T*`, 9 `event_r*_T*` cells, `in_event_r*`, `t_to_onset_r*`, `oracle_cov_T*` for truncation/frame-drop masking), `onset_times` (crossing with 0.2 m hysteresis — single source of truth for "event"). Imports only numpy/cache/CameraModel/mocap_gt helpers — firewall.
3. **New `riskam/ssm.py`** — ISO/TS 15066-style protective distance `S_p = v_h(t_r+t_s) + v_r·t_r + C`; `SSMParams` (v_h=1.6 worst case, v_h_aware=0.5, t_r=0.3, t_s=0.6, C=0.3 — every constant cited); awareness-modulated v_h interpolation; per-frame **margin channels** (threshold-sweepable, not a fixed binary).
4. **New `riskam/event_table.py`** — sibling of scene_table (scene_table untouched): causal pass (fresh KinematicTracker + RiskScorer per run, ALL cached frames, not GT-gated) → score channels + `nearest_d_m` + SSM margins; oracle pass (hindsight) → oracle block; join by frame walk. `gt`/`bucket` columns filled for 2023, empty for 2024. CSV + meta JSON (provenance, model SHA, params).
5. **New `riskam/early_warning.py`** — channels: m0, proximity-only, TTC-only, hazard, risk_a, ssm_worst, ssm_aware (sign-aware). Frame-level ROC/PR AUC per (r,T) — all-frames AND pre-event-only variants; negatives with `oracle_cov<0.5` masked. Episode logic (sustained ≥0.2 s alarms, 0.5 s merge): event recall, lead time (onset − alarm-episode start), FA/min. `threshold_for_recall` sweep; `matched_recall_headline(ref=ssm_worst @ margin≤0, cand=ssm_aware)` → FA/alarm-time reduction %. `render_report_md`.
6. **New `scripts/layer2.py`** — metric_lab-style subcommands: `check / populate / build / evaluate / report / all`, `--dataset`, `--device mps`, `--runs`, `--refresh`. Populate is whole-run resumable, sequential from frame 0 with unconditional cache.put (ByteTrack invariant), per-run timing logs. Outputs → `exp_results/<dataset>/layer2/{event_table.csv, event_table_meta.json, early_warning.json, report.md}`.
7. **Modified `scripts/test_run_with_video.py`** — replace the 2023-only depth-index branch (lines 49–57) with the registry map → works for cs_robocup_2024; risk video fps=10 + raw fps=30 into `test_results/`.
8. **New tests** (~35–45, synthetic, no GPU): `test_hindsight.py` (approach triggers cells at the right time; recede → 0; sentinel censoring; dead zone; ID-switch mid-approach still events; coverage on gaps/run-end; import firewall), `test_ssm.py` (closed-form, monotonicity, aware ≥ worst margin), `test_event_table.py` (synthetic cache fixture, unlabelled rows, alignment, CSV round-trip), `test_early_warning.py` (episode/lead/FA logic on hand-built series, matched-recall on a constructed table).

## Execution order

1. `run_datasets.py` + video-script generalization → **render receptionist_1 risk video** (live inference, no populate needed) — the user-requested deliverable lands first.
2. `hindsight.py` + tests (the risky part), then `ssm.py`, `event_table.py`, `early_warning.py` + tests.
3. `layer2.py`; then **2023 end-to-end first**: check → populate RB_01 (--refresh) + RB_07 (~9k frames, 5–10 min MPS) → build → evaluate → read report (validates pipeline on dense data).
4. **2024 populate overnight** (96k frames, ~45 min–2 h on MPS, whole-run resumable) → build → evaluate → report.
5. Cross-checks in report: per-cell onset counts (flag cells <10 onsets; headline cell = largest adequately-populated, pre-registered rule), `d_oracle_now_m` vs causal `nearest_d_m` correlation ≈ 1 (decode-bug canary), proximity saturates on in-event frames but drops on pre-event mask (mask sanity).

## Verification

- Full pytest suite green (275 existing + new; legacy untouched except 2 import lines).
- `layer2.py check` before/after populate shows expected coverage; build/evaluate runtimes minutes.
- Report.md contains: AUC tables per (r,T)×channel (pooled + per-run), lead-time distributions, FA-vs-recall curves, the SSM matched-recall headline, per-cell event counts.
- Videos: `test_results/videos/cs_robocup_2024_receptionist_1_{risk,raw}.avi` play and show sane overlays.

## Risks (mitigated in design)

Event scarcity at r=0.5 (dead zone 0.6 m) → per-cell counts + pre-registered cell selection. Shared-perception circularity residue → oracle_cov masking + L1 independence argument, stated in report. RB_01 cache overwrite affects only future metric_lab rebuilds (frozen artifacts untouched). SSM constants contestable → all in SSMParams with citations + matched-recall framing. MPS overnight → whole-run resume + caffeinate note.

## Out of scope (next iteration)

Layer 3 VLM pairwise protocol (calibration on 2023 labels, clip-pair sampling from the event table, Bradley–Terry, 2-VLM ensemble) — the event table built here is its sampling substrate. L4b behavioural yield analysis — also a query over this table.
