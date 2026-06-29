# Design rationale

The "why" behind RiskAM's current architecture: the SamXL deployment findings
that drove the redesign, the decisions taken (including what was deliberately
*not* done), and the scientific contribution framing. This is preserved history
— for what is still outstanding see [`improvement-plan.md`](improvement-plan.md).

The original motivation, description, and prototype documentation are in
[CoreSense deliverable D3.5](http://zahalka.net/wp-content/uploads/2025/04/CoreSense___CS_067_D3_5__RiskAM_deliverable.pdf).

---

## SamXL findings

SamXL tested RiskAM live on the SAMXL Ridgeback (RealSense D435i + NVIDIA Jetson
Orin AGX, ARM64). They identified six problems and decided to move to a different
module. Each finding and how it was resolved:

### F1 — Relative depth is not a stable proximity proxy → **accepted**
MiDaS relative depth changes for a stationary human when other objects enter/leave
the frame — inconsistent across scenes and runs, a correctness issue for a
safety-critical module. **Resolved:** replaced MiDaS with the RealSense absolute
depth topic (metres). Bonus: removed an inference model (perf win) and lifted the
`numpy<2.0.0` pin. See [`scoring.md`](scoring.md) (proximity) and
[`sensors-and-platforms.md`](sensors-and-platforms.md).

### F2 — Gaze calculation has a logical error → **accepted**
Eye symmetry around the nose-to-neck midpoint is maximised facing the camera, but
*equally* maximised looking straight up/down/behind — false "aware" readings.
**Resolved:** replaced the 1-D horizontal-offset heuristic with 2-D head pose
(yaw + pitch) from YOLO11-Pose keypoints; a person is "aware" only when both are
near camera-facing. See [`scoring.md`](scoring.md) (gaze).

### F3 — Tracker fails at low frame rates (3–7 Hz) → **accepted**
IoU-based tracking assumes small inter-frame motion; at 3–7 Hz walking humans
move far enough to break continuity. **Resolved (tracker):** dropped the custom
`BboxTracker` for Ultralytics ByteTrack (`model.track(...)`), motion-aware, no
extra dependency. **Resolved (frame rate):** removing MiDaS frees ~100 ms/frame;
async worker-thread processing (see [`architecture.md`](architecture.md)) handles
the rest.

### F4 — X-pose assumes the robot always moves in +X → **partially accepted**
The x-position sub-score assumed forward-only motion; the Ridgeback has mecanum
wheels and moves omnidirectionally. The *idea* (weight risk by trajectory
intersection) is sound; the formulation was wrong. **Resolved:** subscribe to
`/cmd_vel` and project the bbox onto the robot's motion direction (path-aware
`x_offset`), with graceful degradation to centre-offset when stationary. See
[`scoring.md`](scoring.md) (x_offset).

### F5 — Gaze is a weak awareness proxy in industrial settings → **partially accepted**
Experienced workers may be aware without looking. **Accepted as a limitation, not
a reason to remove gaze:** for general HRI (domestic, logistics) gaze is
well-motivated and scientifically interesting. Industrial users set `w_gaze = 0`.
No structural change beyond the F2 fix.

### F6 — No dynamic information (humans moving toward/away) → **accepted**
Each frame was evaluated independently; a person walking away scored the same as
a stationary one at the same distance. **Resolved:** the per-ByteTrack-ID
approach sub-score tracks depth over a short window and modulates risk by whether
the person approaches or recedes. See [`scoring.md`](scoring.md) (approach).

### R1 — Replace RiskAM with a VLM-based vision module → **rejected**
SamXL's decision was driven by their project schedule, not a technical assessment
that RiskAM is unfixable. The improvements above address every identified issue.
RiskAM's explicit, interpretable sub-score architecture is a scientific
*strength*: interpretable scoring with well-defined semantics is more publishable
and auditable than a VLM black box, and aligns with EU human-robot safety
priorities.

---

## Deliberate non-decisions

### No depth-unavailable fallback (T3.2)
An early plan item proposed falling back to MiDaS (or some default) when the
depth topic drops out. **Rejected:** silently zeroing the dominant proximity
sub-score is a safety hazard, not a degradation mode. RGB + depth are *required*
inputs; missing either raises a clear `ValueError` at the pipeline boundary
(`FrameInputs.validate()`). See [`architecture.md`](architecture.md) (input contract).

### MiDaS ruled out as an ablation baseline (T3.3.11)
The ablation study's "pre-improvement baseline" originally included MiDaS
relative depth. Once RGB + absolute depth became a hard requirement, a MiDaS
baseline would be a regression on F1/T1.1, so it was dropped. Only the gaze
algorithm needs a named alternative — `eye_symmetry` is retained for the
ablation. Centre-offset `x_offset` and `w_approach = 0` already exist as explicit
sweep axes. See [`evaluation-framework.md`](evaluation-framework.md).

---

## Summary of decisions

| SamXL suggestion | Decision | Rationale |
|------------------|----------|-----------|
| Use RealSense depth instead of MiDaS | **Accept** | Correct, high priority (F1) |
| 3-D bounding-box fusion | **Simplify** | Depth at bbox centroid sufficient; full 3-D box overkill |
| Fix gaze with head tilt | **Accept, extend** | Full 2-D head pose (F2) |
| Path-aware x-pose | **Accept** | Subscribe to velocity (F4) |
| Abandon gaze for industrial settings | **Reject** | Configurable weight handles it; gaze is scientifically useful (F5) |
| Add human velocity tracking | **Accept** | Temporal tracking via ByteTrack IDs (F6) |
| Abandon RiskAM for a VLM module | **Reject** | Issues are fixable; interpretability is a scientific asset (R1) |

---

## Scientific contribution framing

Targeting a reputable venue (ICRA, IROS, RA-L, RAS), the intended contribution:

> *An interpretable, real-time multi-factor risk awareness module for
> human-robot interaction, featuring absolute-depth-calibrated proximity,
> 2-D-head-pose-corrected gaze awareness, robot-trajectory-aware path risk, and
> temporally consistent tracking. Evaluated quantitatively against annotated
> real-world deployment data on an omnidirectional mobile platform.*

Key novelties vs prior work:

1. Fusing absolute depth from an RGB-D sensor with head pose into a jointly
   calibrated, physically interpretable risk score.
2. Robot-trajectory-aware collision risk weighting (path projection from velocity).
3. Per-tracked-person temporal risk aggregation with a multi-person crowd penalty.
4. Quantitative ablation comparing the original heuristic sub-scores with each
   proposed improvement.

The explicit score architecture (vs black-box VLMs) is a selling point for
explainability-focused venues and EU safety-focused robotics workshops. Forward-
looking metric-redesign thinking is parked in
[`private/paper-plan.md`](private/paper-plan.md).
