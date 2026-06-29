# Architecture

How a frame becomes a risk score, and the input-availability contract that
governs which sub-scores are allowed to contribute.

For the per-sub-score math see [`scoring.md`](scoring.md); for the depth/sensor
specifics see [`sensors-and-platforms.md`](sensors-and-platforms.md); for the
ROS wiring see [`ros-deployment.md`](ros-deployment.md).

---

## Pipeline

A single frame flows through three stages:

```
FrameInputs ──► featextr.extract ──► FrameExtraction ──► RiskScorer.score ──► scene risk
   (rgb,           (YOLO11-Pose +        (features,          (weights,          (float in
    depth_m,        ByteTrack +           subscore_status,    per-track          [0, 1],
    cmd_vel?)       sub-scores)           track_ids, …)       history)           max idx,
                                                                                 per-person)
```

The live ROS node and the offline experiment harness both go through this exact
path. The node supplies `cmd_vel`; offline experiments pass `cmd_vel=None` until
bag-backed velocity replay lands (see the improvement plan, T3.3.3).

### `featextr` two-way split

`riskam/ml/featextr.py` separates parameter-independent inference from
parameter-dependent scoring so the expensive half can be cached:

| Function | Cost | Depends on sweep params? | Cacheable |
|----------|------|--------------------------|-----------|
| `extract_primitives` | YOLO11-Pose + ByteTrack + per-bbox depth | no | yes (`riskam.feature_cache`) |
| `extract_features` | sub-score arithmetic | yes (`d_safe`, gaze sigmas, …) | no (cheap) |
| `extract` | composes both, no cache | — | — |
| `extract_with_cache` | composes both, disk cache for primitives | — | — |

`extract_primitives` returns `CachedFrameFeatures` (bboxes, keypoints,
track_ids, per-bbox depths, depth_viz). Because the result is deterministic per
image, a parameter sweep re-runs only `extract_features`. See
[`evaluation-framework.md`](evaluation-framework.md) for the cache caveat under
ByteTrack's temporal coupling.

---

## Sub-score input-availability contract

Implemented in `riskam/ml/subscores.py`. The contract formalises which inputs
each sub-score needs and what happens when an *optional* input is missing,
rather than handling that ad hoc at every call site.

### Required vs optional inputs

- **Required (hard floor):** RGB image + absolute depth in metres.
  `FrameInputs.validate()` enforces this at the pipeline boundary; missing
  either — or a wrong-rank array — raises `ValueError`. A risk module that
  silently zeroes its dominant proximity sub-score when depth drops out is a
  safety hazard, not a degradation mode. This is why there is **no**
  depth-unavailable fallback (a deliberate rejection, see
  [`design-rationale.md`](design-rationale.md)).
- **Optional:**
  - `cmd_vel` (`RobotVelocity`) → enables path-aware `x_offset`; absent →
    centre-offset fallback.
  - track continuity (ByteTrack IDs + depth history) → enables `approach`;
    absent → neutral, no contribution.

### Per-sub-score status

Each `compute_*` function returns a `SubScoreResult(values, status, reason)`
with `status ∈ SubScoreStatus`:

| Status | Meaning |
|--------|---------|
| `ACTIVE` | computed from live inputs |
| `FALLBACK` | a declared fallback produced the value (e.g. centre-offset when `cmd_vel` is missing) |
| `UNAVAILABLE` | no fallback; value is neutral and carries no information |

Status flows out of the pipeline via `FrameExtraction.subscore_status` (and
`subscore_reasons`). The ROS node publishes one `subscore_{name}` key per
sub-score on `/riskam/diagnostics`, escalating the diagnostic level OK → WARN
whenever any sub-score is non-`ACTIVE`, so degraded modes are visible on the bus.

### Scoring semantics

The scorer multiplies `values × weight` **regardless of status**. Weights are
**not** silently redistributed when a sub-score degrades — that would change the
scoring semantics without telling the caller. A `FALLBACK` or `UNAVAILABLE`
sub-score still contributes its (fallback / neutral) value at its configured
weight; the *status*, not the *weight*, is how degradation is communicated.

### Framework neutrality

`RobotVelocity` is a small dataclass (`linear_x`, `linear_y`, derived `speed`)
constructible from a ROS `geometry_msgs/Twist` via `RobotVelocity.from_twist`.
The subscores module therefore has no ROS import and is usable from offline
experiments, tests, and notebooks.

---

## Data types

Defined in `riskam/ml/subscores.py`:

- **`FrameInputs`** — `rgb` (BGR `(H,W,3)` uint8), `depth_m` (`(H,W)` float32
  metres), optional `cmd_vel`. `validate()` is the required-input gate.
- **`SubScoreResult`** — `values` (`(n_persons,)` float in [0,1]), `status`,
  `reason`.
- **`FrameExtraction`** — `human_bboxes`, `depth_viz`, `features`
  (`{name: array}` or `None` when no humans), `subscore_status`,
  `subscore_reasons`, `track_ids`.

---

## Statefulness and threading

- **`RiskScorer` (`riskam/score.py`)** is a stateful class holding per-track
  risk history (keyed by ByteTrack ID, with a `None` bucket for untracked
  detections). It replaced the prototype's module-level `prev_risks` global,
  making the scorer thread-safe to instantiate, testable, and safe in
  multi-context pipelines. `score()` returns `(scene_risk, max_risk_idx,
  per_person)`; `reset()` clears history on node restart.
- **The ROS node** processes frames on a dedicated worker thread, not in the
  image callback. The callback stores only the latest synced color+depth pair
  (latest-only, not a growing queue); the worker consumes it and publishes
  results. This is the standard ROS 2 pattern for compute-heavy perception nodes
  and prevents callback backup at the 3–7 Hz deployment frame rate.

---

## Minimal programmatic use

```python
import cv2, numpy as np
from riskam.ml import featextr
from riskam.ml.subscores import FrameInputs, RobotVelocity
from riskam.score import RiskScorer

scorer = RiskScorer()
result = featextr.extract(
    FrameInputs(
        rgb=cv2.imread("frame.png"),          # BGR
        depth_m=np.load("frame_depth_m.npy"), # float32 metres
        cmd_vel=RobotVelocity(0.3, 0.0),      # or None
    ),
)
risk, max_idx, per_person = scorer.score(result.features, track_ids=result.track_ids)
```

`result.subscore_status` / `result.subscore_reasons` carry the same per-sub-score
status the ROS node publishes on `/riskam/diagnostics`.
