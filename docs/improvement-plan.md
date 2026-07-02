# RiskAM Improvement Plan

*Author: Jan Zahálka | Last reviewed: 2026-06-29*

---

## Scope

This document tracks **outstanding** work only. Functionality already
implemented is documented in the topic docs and is no longer repeated here:

- [`architecture.md`](architecture.md) — pipeline + sub-score input contract
- [`scoring.md`](scoring.md) — sub-score math + aggregation
- [`sensors-and-platforms.md`](sensors-and-platforms.md) — depth, near-clip, platforms
- [`ros-deployment.md`](ros-deployment.md) — node, params, topics, bagger
- [`evaluation-framework.md`](evaluation-framework.md) — offline eval toolchain
- [`visualization.md`](visualization.md) — overlay semantics
- [`design-rationale.md`](design-rationale.md) — SamXL findings + decisions (the "why")
- [`changelog.md`](changelog.md) — what landed, when

All Tier 1 (T1.1–T1.6) and Tier 2 (T2.1–T2.7) items are **done**, as are most of
Tier 3. What remains is below.

---

## Remaining roadmap

| # | Item | Effort | Impact | Status | Notes |
|---|------|--------|--------|--------|-------|
| T3.1 | AMD64 Dockerfile | Low | High for deployment | **Pending** | The ARM64 Dockerfile for the Jetson Orin exists and should stay; add an AMD64 variant. |
| T3.3.3 | Offline `/cmd_vel` replay for path-aware `x_offset` | Medium | High | **Partial — data unblocked** | The sub-score contract and path-projection math are done (see [`architecture.md`](architecture.md) / [`scoring.md`](scoring.md)). Bag verification done: `/cmd_vel` is empty in RB_01/06/07, but `/mobile_base_controller/odom` is dense in every run and is now extracted (`odom.csv` + `CSRobocup2023OdomIndex`, see [`experimental-metrics.md`](experimental-metrics.md)). Remaining: feed the odom twist into `run_experiment` so offline `x_offset` can be `ACTIVE` instead of `FALLBACK`. |
| T3.3.12 | CI regression gate | Medium | Medium | **Pending** | Small curated subset; fail PRs whose key metric regresses beyond a documented budget. Needs a baseline run to compare against. |
| T3.4.1 | Show current-pipeline prediction while annotating | Low | Medium | **Pending** | Run the pipeline once per frame; display its class alongside the image to calibrate annotator judgment. |
| T3.4.2 | Non-destructive save + resume/progress | Low | Medium | **Pending** | Back up prior annotations on save; resume from the last labelled frame. |
| T3.4.3 | Per-person (per-bbox) annotation | Medium | Medium | **Pending** | Scene-level label is a max; per-person labels enable per-sub-score evaluation and multi-person scenes. |
| T3.4.4 | Optional per-sub-score annotations | Medium | High for paper | **Pending** | Separate ground truth per sub-score where feasible; enables fine-grained ablation. |
| T3.4.5 | Re-annotate cs_robocup_2023 against the new pipeline | Medium (human) | High for paper | **Pending** | Uses T3.4.1–T3.4.4. Blocks the ablation evaluation. Pure human work. |
| T3.6 | API docstring pass + standalone deployment guide | Low | Medium | **Partial** | README rewritten and topic docs landed. Remaining: a full module-level + public-function docstring pass, and a standalone deployment guide if the EU deliverable template requires one beyond [`ros-deployment.md`](ros-deployment.md). |
| T3.7 | Sphinx API docs | Low | Low | **Pending** | Optional, only if the deliverable template requires it. |

---

## Notes on blocked items

Two items gate on data access rather than engineering:

- **T3.3.12 (CI gate)** — needs a trusted baseline metric run to regress against.
- **T3.4.5 (re-annotation)** — pure human annotation work, unblocked once
  T3.4.1–T3.4.4 land.

The framework is complete for all data-independent work.

---

*The annotator-tool items (T3.4.1–T3.4.4) and T3.1 are the next data-free
session's work. Append landed items to [`changelog.md`](changelog.md) and remove
them from this table as they complete.*
