# RiskAM documentation

Documentation for the CoreSense Risk Awareness Module. The root
[`../README.md`](../README.md) is the quickstart; the docs below are the
in-depth, categorised reference.

## Implemented functionality

| Doc | Covers |
|-----|--------|
| [architecture.md](architecture.md) | Frame-to-score pipeline, `featextr` split, and the sub-score input-availability contract |
| [scoring.md](scoring.md) | The four sub-scores (proximity, gaze, x_offset, approach) and scene aggregation — the algorithmic core |
| [sensors-and-platforms.md](sensors-and-platforms.md) | Absolute-depth handling, the near-clip dead-zone fallback, and `riskam/platforms.py` presets |
| [ros-deployment.md](ros-deployment.md) | Building/running the node; exhaustive parameter, topic, and bagger reference; deployment checklist |
| [evaluation-framework.md](evaluation-framework.md) | Offline benchmarking toolchain (sweep, metrics, summary, reeval, cache, splits, provenance) |
| [visualization.md](visualization.md) | How to read the annotated overlay |

## Background and planning

| Doc | Covers |
|-----|--------|
| [design-rationale.md](design-rationale.md) | SamXL deployment findings (F1–F6), decisions taken and rejected, scientific contribution framing |
| [changelog.md](changelog.md) | Distilled implementation history |
| [improvement-plan.md](improvement-plan.md) | **Outstanding work only** — the remaining roadmap |
| [private/paper-plan.md](private/paper-plan.md) | Exploratory metric-redesign notes (not a committed plan) |

## Audience shortcuts

- **Deploying RiskAM on a robot** → [ros-deployment.md](ros-deployment.md) +
  [sensors-and-platforms.md](sensors-and-platforms.md).
- **Understanding/extending the score** → [architecture.md](architecture.md) +
  [scoring.md](scoring.md).
- **Benchmarking / the paper** → [evaluation-framework.md](evaluation-framework.md) +
  [design-rationale.md](design-rationale.md).
