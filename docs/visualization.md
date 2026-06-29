# Visualisation

How to read the annotated overlay published on `/riskam/annotated_image` (live)
or written to `test_results/videos/*_risk.avi` (offline). Implemented in
`riskam/visualization.py` (`visualize_risk`).

---

## Overlay elements

1. **Scene risk readout** — colour-coded aggregated scene risk in the top-right
   corner (bands from `RISK_SCORE_BREAKPOINTS = [0, 0.3, 0.6, 1]`).
2. **Bounding boxes** for all detected humans.
3. **Gaze colour** — each bbox is coloured along a continuous **red↔white
   gradient** by its gaze sub-score:
   - **Red** (gaze ≈ 0): pose undetectable, or person likely *not* aware of the
     robot — BGR `(0, 0, 255)`.
   - **White** (gaze ≈ 1): person clearly aware (looking at the camera) — BGR
     `(255, 255, 255)`.
   - **Pinks** in between for ambiguous gazes (linear interpolation).
   - A thicker **black outline** is drawn underneath each box so it stays visible
     on red walls / shirts and on white / overexposed backgrounds.
4. **Max-risk marker** — the bbox driving the scene-risk maximum is marked with
   an asterisk.
5. **Proximity tint** — each bbox interior is tinted **red with opacity
   proportional to its proximity sub-score** (`α = proximity · 0.7`): red = close
   = danger, transparent = far = safe. The tint is per-bbox uniform and driven by
   the proximity *sub-score*, not raw per-pixel depth, so structured-light sensor
   noise (Xtion) does not splotch the overlay. T2.7 close-fallback bboxes
   ("person too close to measure") render as full red panels. Outside bboxes the
   original scene is untouched.

---

## Design notes

The current overlay is the result of several soundness-testing iterations
against the cs_robocup_2023 (Xtion) data:

- **Bbox-only blend.** The original whole-frame depth-fog blend was too brutal on
  Xtion data — with 50–85% of pixels carrying no valid depth, almost every pixel
  pulled toward the grey overlay and scene context washed out. Since the module
  only acts on detected humans, the blend is now scoped to bbox interiors.
- **Red-by-proximity, not grey-by-depth.** The interior tint was switched from a
  grey-fog-by-depth gradient to red-by-proximity. The grey gradient communicated
  "darkness = far", inverted from the natural "red = close = danger" semantic
  that matches what the score actually means; and tying the tint to the proximity
  sub-score (rather than the noisy per-pixel depth field) keeps the overlay clean
  on heavy zero-fill sensors.
- **Gaze gradient, not thresholded colours.** The earlier strict-equality colour
  check against `{0, 1}` was a leftover from the pre-T1.3 thresholded gaze score;
  the post-T1.3 gaze is a continuous Gaussian that never lands exactly on either
  value, hence the continuous gradient.

Contracts (gradient endpoints, mid-point monotonicity, out-of-range clamping,
outline presence, bbox-scoped blend) are pinned by `tests/test_visualization.py`.
