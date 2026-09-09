"""
ml.facegate

Awareness measurability gate: is a person's gaze reading a *measurement*?

Pose models must emit face keypoints for every detected person. When the face
region carries no usable evidence — privacy-defaced datasets (Gaussian-blur
ellipses), heavy occlusion, or simply too few pixels — the keypoints degrade
to a learned frontal-face template drawn onto the blob. The downstream gaze
score then reads "looking at the robot" for people who may be staring at a
phone. That error is doubly harmful: it grants awareness credit (smaller
protective separation, lower fused risk) exactly where nothing was measured.

The gate declares a gaze reading a measurement only when both hold:

1. **Resolution floor** — inter-eye distance ≥ ``GATE_MIN_INTER_EYE_PX``.
   Yaw is quantised at ~1 px / inter-eye, so below ~7 px the yaw noise
   exceeds the scoring Gaussian's σ and no head pose is resolvable,
   defaced or not.
2. **Face-texture floor** — Laplacian variance of the eye-region patch ≥
   ``GATE_MIN_FACE_TEXTURE``. Real eye regions carry strong local structure;
   blur ellipses are smooth apart from sensor noise. The default threshold
   is calibrated so that ~90% of an unblurred reference population
   (cs_robocup_2023, inter-eye ≥ 7 px) passes, while ~4 in 5 hand-labelled
   defacing blobs fail. Like ``d_safe`` or the depth near-clip, it is a
   documented per-camera constant, not a swept hyperparameter.

Ungated consequences flow through :func:`riskam.ml.subscores.compute_gaze`:
an unmeasurable face contributes gaze 0 (treated as unaware — the
conservative direction for the weighted sum, the kinematic fusion, and the
SSM margins alike) and the sub-score status degrades to FALLBACK so the
share of unmeasured faces is visible on diagnostics.

Known residuals (documented, accepted): (1) keypoints mislocalised onto
textured clothing or backgrounds can pass the texture floor — their gaze
geometry is arbitrary rather than frontal-biased, so they rarely grant
awareness credit; per-keypoint model confidence would catch them and is the
natural extension. (2) **Printed faces** (posters, advertisements) are real,
sharp, frontal faces and pass both floors — they can even pass person
detection upstream (observed on crowdbot_v2: a shop-window poster detected
as two persons, granting up to ~0.9 m of separation credit). Liveness
discrimination is out of scope for this gate; awareness-modulated behaviour
in environments with printed faces should be validated per deployment.

The texture *measurement* is parameter-independent (image + keypoints only)
and is computed in :func:`riskam.ml.featextr.extract_primitives`, cached per
frame. The gate *decision* is two cheap comparisons at scoring time.
"""

from __future__ import annotations

import cv2
import numpy as np

# Resolution floor: below this inter-eye distance the yaw quantisation noise
# (~1 px / inter-eye ≈ 0.14 at 7 px) approaches sigma_yaw (0.3) and no head
# pose is resolvable regardless of image content.
GATE_MIN_INTER_EYE_PX_DEFAULT = 7.0

# Face-texture floor (Laplacian variance of the eye patch). Calibrated
# 2026-09-02: unblurred reference (cs_robocup_2023, inter-eye ≥ 7 px)
# p10 ≈ 75; hand-labelled crowdbot_v2 defacing blobs median ≈ 16, ~79%
# below this threshold. Per-camera constant in spirit; raise for very
# sharp/bright cameras, lower for dim ones.
GATE_MIN_FACE_TEXTURE_DEFAULT = 75.0

# Eye-patch geometry relative to inter-eye distance: wide enough to span
# both eyes plus the nose bridge, tight enough to stay on the face.
_PATCH_HALF_W_IE = 1.0
_PATCH_HALF_H_IE = 0.6
_MIN_PATCH_PX = 25  # below this many pixels the statistic is meaningless

# COCO keypoint indices (see humandet._headpose_gaze).
_NOSE, _LEFT_EYE, _RIGHT_EYE = 0, 1, 2


def _eye_geometry(kpts: np.ndarray) -> tuple[float, float, float] | None:
    """Return (eye_mid_x, eye_mid_y, inter_eye_px) or None when the face
    triple is absent (all-zero keypoints — the pose model's "undetected")."""
    nose, le, re = kpts[_NOSE], kpts[_LEFT_EYE], kpts[_RIGHT_EYE]
    if np.all(nose == 0) or np.all(le == 0) or np.all(re == 0):
        return None
    mid = (le + re) / 2.0
    return float(mid[0]), float(mid[1]), float(np.linalg.norm(re - le))


def measure_face_texture(
    rgb: np.ndarray,
    keypoints_np: np.ndarray | None,
) -> np.ndarray:
    """Per-person eye-region texture (Laplacian variance), NaN when unmeasured.

    Parameter-independent: depends only on the image and the keypoints, so the
    result is cached with the other primitives. NaN means "no face triple or
    patch too small" — the gate treats NaN as unmeasurable.
    """
    if keypoints_np is None or len(keypoints_np) == 0:
        return np.zeros((0,), dtype=np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    out = np.full(len(keypoints_np), np.nan, dtype=np.float32)
    for i in range(len(keypoints_np)):
        geom = _eye_geometry(keypoints_np[i])
        if geom is None:
            continue
        cx, cy, ie = geom
        half_w = max(4.0, _PATCH_HALF_W_IE * ie)
        half_h = max(3.0, _PATCH_HALF_H_IE * ie)
        x1, x2 = int(max(0, cx - half_w)), int(min(w, cx + half_w))
        y1, y2 = int(max(0, cy - half_h)), int(min(h, cy + half_h))
        patch = gray[y1:y2, x1:x2]
        if patch.size < _MIN_PATCH_PX:
            continue
        out[i] = cv2.Laplacian(patch.astype(np.float32), cv2.CV_32F).var()
    return out


def gaze_measurable(
    keypoints_np: np.ndarray | None,
    face_texture: np.ndarray | None,
    min_inter_eye_px: float = GATE_MIN_INTER_EYE_PX_DEFAULT,
    min_face_texture: float = GATE_MIN_FACE_TEXTURE_DEFAULT,
) -> np.ndarray:
    """Per-person boolean mask: True = the gaze reading is a measurement.

    ``face_texture is None`` (primitives cached before the texture field
    existed) degrades to the geometry-only gate — resolution floor applied,
    texture floor skipped. Per-person NaN texture (no face triple / patch
    too small) is unmeasurable.
    """
    if keypoints_np is None or len(keypoints_np) == 0:
        return np.zeros((0,), dtype=bool)
    n = len(keypoints_np)
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        geom = _eye_geometry(keypoints_np[i])
        if geom is None or geom[2] < min_inter_eye_px:
            continue
        if face_texture is None:
            mask[i] = True  # stale cache: geometry-only gating
            continue
        tex = float(face_texture[i]) if i < len(face_texture) else float("nan")
        mask[i] = not np.isnan(tex) and tex >= min_face_texture
    return mask
