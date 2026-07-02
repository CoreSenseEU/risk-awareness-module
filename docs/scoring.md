# Risk scoring

The algorithmic core of RiskAM: how each sub-score is computed and how they
aggregate into a single scene risk. For how a frame reaches these functions see
[`architecture.md`](architecture.md); for the depth handling that feeds
proximity see [`sensors-and-platforms.md`](sensors-and-platforms.md).

The exploratory metric-redesign notes (units, referent, generalisation across
platforms) live in [`private/paper-plan.md`](private/paper-plan.md); their
offline implementation (kinematic hazard × awareness, ordinal calibration) is
described in [`experimental-metrics.md`](experimental-metrics.md). This
document describes the **currently deployed** score.

---

## Per-person risk

For each detected person, the instantaneous risk is a weighted sum of four
sub-scores (`riskam/score.py`):

```
risk_person = w_proximity · proximity
            + w_gaze      · (1 − gaze)      # high gaze = aware = lower risk
            + w_position  · x_offset
            + w_approach  · approach
```

Defaults: `w_proximity = 0.7`, `w_gaze = 0.25`, `w_position = 0.05`,
`w_approach = 0.0`. Weights should sum to 1.0 (validated by the ROS node). Each
sub-score is in [0, 1] with "1 = more dangerous", except `gaze` which is
inverted in the formula because a high gaze score means the person *is* aware.

`w_approach` defaults to `0.0` so wiring the approach sub-score into the formula
does not shift live behaviour; the evaluation sweep exercises non-zero values
and a calibrated default will come from the ablation study.

---

## Sub-scores

### proximity — `riskam/ml/depth.py`, `subscores.compute_proximity`

Absolute-depth-calibrated, replacing the prototype's MiDaS relative depth (which
was not a stable proximity proxy — see [`design-rationale.md`](design-rationale.md), F1).

```
proximity = clip(1 − depth / d_safe, 0, 1)
```

Linear falloff: `1` at the camera, `0` at or beyond the safety distance
`d_safe`. Physically interpretable and comparable across runs and scenes.
`d_safe` is a platform property (stopping distance + margin), default `1.5 m` —
see [`sensors-and-platforms.md`](sensors-and-platforms.md) for per-platform
values and for how missing depth in a bbox is resolved (near-clip fallback).

`compute_proximity` takes **per-bbox depths**, not the full depth image, so it
behaves identically whether depths were computed fresh or loaded from the
feature cache.

### gaze — `riskam/ml/humandet.py`, `subscores.compute_gaze`

Awareness proxy from YOLO11-Pose keypoints; **no extra model**. Two algorithms
are selectable via `gaze_scores(..., algorithm=...)`:

- **`head_pose`** (default, post-T1.3) — full 2-D head pose. A person is "aware"
  only when **both** yaw and pitch are near the camera-facing direction:

  ```
  gaze = yaw_Gaussian(σ_yaw) · pitch_Gaussian(σ_pitch)
  ```

  Yaw is the left-right rotation (offset of the nose from the eye midpoint,
  normalised by inter-eye distance). Pitch comes from the vertical
  `(nose.y − eye_mid.y) / inter_eye_dist` ratio compared against the expected
  frontal ratio. Defaults: `σ_yaw = 0.3`, `σ_pitch = 0.5`,
  `frontal_pitch_ratio = 0.7`.

- **`eye_symmetry`** (pre-T1.3 baseline) — yaw-only Gaussian around the
  nose↔eye-midpoint axis. Retained **for the ablation study**; its known
  weakness (false "aware" readings for people looking up/down/behind — F2) is
  preserved by construction and pinned by tests. Not for deployment.

The status is `ACTIVE` for either algorithm — algorithm choice is orthogonal to
input availability. The algorithm name is part of each experiment's `params` so
the summariser ranks both side by side.

> **Limitation.** In industrial settings, experienced workers may be aware of a
> robot without looking at it, so gaze is a weak awareness proxy there. This is a
> documented limitation, not a reason to remove gaze: set `w_gaze = 0` to disable
> it for industrial deployment. For general HRI (domestic, logistics) gaze is a
> well-motivated awareness signal. See [`design-rationale.md`](design-rationale.md), F5.

### x_offset — `subscores.compute_x_offset`

"Is this person in the robot's path?" — corrected for omnidirectional robots
(the prototype wrongly assumed forward-only motion, F4).

- **`ACTIVE`** (path-aware): when `cmd_vel` is supplied and the robot is moving
  (speed ≥ 0.05 m/s), the bbox centre is projected onto the robot's instantaneous
  motion direction; score is a Gaussian around the lane the robot is heading
  into. Uses ROS REP-103 conventions (`vx>0` forward → danger at image centre;
  `vy>0` left → robot's left appears on image left).
- **`FALLBACK`** (centre-offset): when `cmd_vel` is `None` or the robot is
  effectively stationary, score is the legacy centre-offset heuristic
  (`1 − offset²`: 1 at image centre, 0 at the edges).

### approach — `riskam/ml/humandet.py`, `subscores.compute_approach`

Temporal velocity per tracked person (addresses F6 — a person walking away poses
less collision risk than a stationary one at the same distance). Once ByteTrack
gives stable IDs, per-track depth observations over a short window are
linear-fit; the depth slope maps to an approach score in [0, 1]:

- `0.5` = stationary / unknown (neutral baseline)
- `> 0.5` = approaching (raises risk)
- `< 0.5` = moving away (lowers risk)

`update_velocity` maintains the per-track depth history (fed **raw metres**, not
proximity scores); `reset_velocity_history` clears it between offline runs.
Status is `UNAVAILABLE` when no track has enough history to be informative.

---

## Scene aggregation

`RiskScorer.score()` combines per-person risks into a single scene risk
(`riskam/score.py`):

1. **Temporal smoothing.** Each person's instantaneous risk is averaged over the
   last `n_frames_aggregate` frames (default `5`), keyed by ByteTrack ID — so the
   history follows the person, not a global average.
2. **Crowd penalty.** The scene risk is the max over persons, scaled by a crowd
   factor:

   ```
   scene_risk = max(per_person) · (1 + α · log(1 + n_extra))
   ```

   where `n_extra = n_persons − 1` and `α = crowd_alpha` (default `0.1`). Clipped
   to [`VERY_SMALL_RISK_VALUE`, 1].

`score()` returns `(scene_risk, max_risk_idx, per_person)`. `max_risk_idx` is the
index of the highest-risk person (marked in the visualisation); `per_person`
maps each track ID to its smoothed risk.

`RISK_SCORE_BREAKPOINTS = [0, 0.3, 0.6, 1]` defines the low/medium/high bands
used by downstream consumers and the visualisation.
