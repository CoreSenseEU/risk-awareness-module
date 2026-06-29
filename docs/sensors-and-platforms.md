# Sensors and platforms

How RiskAM handles absolute depth across sensor classes, and the platform-preset
abstraction that bundles sensor + safety parameters. This underpins the
proximity sub-score ([`scoring.md`](scoring.md)).

---

## Why "depth = 0 in a bbox" is sensor-dependent

RiskAM requires absolute depth in metres (see
[`architecture.md`](architecture.md) — RGB + depth is a hard floor). But a
bbox with **no valid depth pixels** carries different information depending on
the sensor, and getting this wrong inverts the safety direction:

- **Active-stereo IR sensors (RealSense D4xx):** clean depth from ~0.3 m onward;
  zero-pixels are rare and usually mean "far / occluded". The legacy
  interpretation — fall back to `d_safe` (proximity 0) — is correct here.
- **Structured-light sensors with a hard near-clip (PrimeSense Xtion /
  Carmine):** a ~0.5–0.8 m near-clip dead zone means a person standing *too
  close* returns 0% valid depth. Treating that as proximity 0 is exactly
  backwards — the person is at maximum risk, not minimum.

This is a **calibration & semantics issue, not a bug**: the pipeline did what
its code said; the code was tuned for one sensor class. The structurally honest
fix makes the interpretation a configurable, physical property of the sensor.

### Symptoms this fixed (cs_robocup_2023 / Xtion soundness testing)

1. **Uniform-black depth viz** — normaliser cliffed at `d_safe = 1.5 m`, but the
   Xtion's usable range starts ~2 m, so almost everything went black.
2. **Score collapse on close-up faces** — a face inside the near-clip dead zone
   returned 0% valid depth → legacy fallback `d_safe` → proximity 0 → scene risk
   carried only by gaze + x_offset (≈ 0.16 for a centred, aware face directly in
   front of the lens).
3. **Wrong "closest bbox" marker** — with all proximities tied at 0, the
   max-risk index was decided by gaze + x_offset alone.

---

## The three depth knobs

`riskam/ml/depth.py` (`extract_bbox_depths` / `extract_bbox_proximities`):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `depth_near_clip_m` | `0.0` | Sensor near-clip dead zone in metres. `0` → close-fallback **off** (RealSense behaviour preserved bit-identically). Set to the spec-sheet value (e.g. `0.6` for Xtion) to enable the dead-zone interpretation. |
| `near_clip_bbox_min_frac` | `0.05` | Minimum bbox area (fraction of full frame) for the close-fallback to fire. Guards against tiny noise-filled bboxes triggering max risk. Ignored when `depth_near_clip_m == 0`. |
| `near_clip_valid_frac_max` | `0.0` | Maximum valid-pixel fraction within a bbox for the close-fallback to fire. `0.0` → strict zero (RealSense pre-T2.7 contract). A small positive value (e.g. `0.05` for Xtion) also fires on "mostly empty" bboxes whose few stray valid pixels are likely background bleed-through, not the person. |

When the close-fallback conditions hold (valid fraction ≤ threshold **and** bbox
large enough), the person is somewhere in the `[0, near_clip]` dead zone and we
cannot measure where; `extract_bbox_depths` assigns depth `0 m` (proximity 1.0)
rather than `d_safe` (proximity 0) — the safety-conservative choice.

`0 m` is a unique sentinel: real measurements are filtered by `DEPTH_MIN_M`, so
the visualisation can detect close-fallback bboxes unambiguously.

### Why `near_clip_valid_frac_max` exists (the sparse-pixel case)

On cs_robocup_2023 RB_08, a person walking close occupied 56% of the frame but
only 1.3% of bbox pixels carried valid depth — and those few pixels were
background bleed-through (whole-frame p50 = 1.85 m), not the person. The strict
"exactly zero" condition trusted them and reported proximity 0; 14 s later the
same person reached the dead zone, the bbox went to 0% valid, and the fallback
fired correctly. The discontinuity was a robustness gap: when valid pixels are
sparse on a structured-light sensor, they are almost certainly noise.
`near_clip_valid_frac_max` extends the fallback to "mostly empty" bboxes.

### Depth visualisation upper bound

`depth_to_visualization` normalises against `max(d_safe, p95 of valid pixels)`
so the viz stays depth-varied on sensors whose working range exceeds `d_safe`.
When all valid pixels are within `d_safe` (the RealSense / SamXL case), the upper
bound stays at `d_safe` and the viz is bit-identical to pre-T2.7.

> **Regression guard.** Tests pin the RealSense defaults so any future change
> that would alter SamXL behaviour fails fast (`tests/test_depth.py`).

---

## Platform presets — `riskam/platforms.py`

A two-level dataclass split reflects the physics: a near-clip dead zone is a
*sensor* property; `d_safe` is a *robot* property (speed + stopping distance).
Neither is genuinely a *dataset* property even though datasets are recorded by
hardware.

```
DepthSensor(name, near_clip_m, valid_frac_max)
RobotPlatform(name, d_safe_m, sensor)
```

These presets configure the **offline research toolchain** for the hardware that
recorded each dataset. The **live ROS node reads scalars from
`riskam_config.yml` directly** and does not consume these constants, so a
deployment can override any field without inventing a new platform name.

### Shipped presets

| Sensor preset | `near_clip_m` | `valid_frac_max` | Hardware |
|---------------|---------------|------------------|----------|
| `REALSENSE_D4XX` | `0.0` | `0.0` | Intel RealSense D435/D435i/D455 — active stereo, clean from ~0.3 m |
| `PRIMESENSE_XTION` | `0.6` | `0.05` | PrimeSense Carmine / PAL Xtion — structured light, ~0.6 m dead zone, 50–85% zero-fill |

| Robot preset | `d_safe_m` | Sensor | Notes |
|--------------|------------|--------|-------|
| `RIDGEBACK_D435` | `1.5` | `REALSENSE_D4XX` | SamXL deployment; matches `riskam_config.yml` defaults bit-identically |
| `TIAGO_XTION` | `2.5` | `PRIMESENSE_XTION` | cs_robocup_2023; `d_safe` bumped 1.5→2.5 m for the Xtion's deeper usable range so mid-range pedestrians aren't penalised |

### Deriving a custom platform

- **`d_safe`** ≈ stopping distance at max speed + safety margin. ~1.5 m for slow
  indoor mobile (~1 m/s); larger for faster or heavier robots.
- **`depth_near_clip_m`** from the sensor spec sheet. `0` for active-stereo
  (RealSense, Azure Kinect, ZED). Published near-clip for structured-light:
  Xtion/Carmine ≈ 0.6 m, Astra ≈ 0.6 m.
- **`near_clip_valid_frac_max`** — `0` if the sensor delivers clean data;
  ~`0.05` for structured-light that produces noisy zero-fill + background
  bleed-through. If unsure leave at `0` and bump only if you see "person clearly
  close, but proximity stays zero" on `/riskam/diagnostics`.

Contributions of new presets to `riskam/platforms.py` are welcome — keep the
dataclass, document the values, add a sanity test in `tests/test_platforms.py`.
