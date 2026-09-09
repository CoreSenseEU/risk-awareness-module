"""Awareness measurability gate (riskam.ml.facegate) + its integration into
compute_gaze / extract_features / the feature cache."""

import numpy as np
import pytest

from riskam.feature_cache import CachedFrameFeatures
from riskam.ml import facegate
from riskam.ml.featextr import extract_features
from riskam.ml.subscores import SubScoreStatus, compute_gaze


def _person_kpts(nose, left_eye, right_eye):
    """(17, 2) keypoints with only the face triple set."""
    k = np.zeros((17, 2), dtype=np.float32)
    k[0], k[1], k[2] = nose, left_eye, right_eye
    return k


def _frontal_kpts(cx=50.0, cy=50.0, ie=12.0):
    """A frontal face triple centred at (cx, cy) with inter-eye ``ie``."""
    return _person_kpts(
        nose=(cx, cy + 0.7 * ie),
        left_eye=(cx - ie / 2, cy),
        right_eye=(cx + ie / 2, cy),
    )


def _textured_image(h=100, w=100, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def _flat_image(h=100, w=100, value=128):
    return np.full((h, w, 3), value, dtype=np.uint8)


# ── measure_face_texture ─────────────────────────────────────────────────────


def test_texture_high_on_structured_eye_region():
    kpts = np.stack([_frontal_kpts()])
    tex = facegate.measure_face_texture(_textured_image(), kpts)
    assert tex.shape == (1,)
    assert tex[0] > facegate.GATE_MIN_FACE_TEXTURE_DEFAULT


def test_texture_near_zero_on_flat_region():
    kpts = np.stack([_frontal_kpts()])
    tex = facegate.measure_face_texture(_flat_image(), kpts)
    assert tex[0] < 1.0


def test_texture_nan_without_face_triple():
    kpts = np.stack([np.zeros((17, 2), dtype=np.float32)])
    tex = facegate.measure_face_texture(_textured_image(), kpts)
    assert np.isnan(tex[0])


def test_texture_empty_for_no_persons():
    assert facegate.measure_face_texture(_textured_image(), None).shape == (0,)
    empty = np.zeros((0, 17, 2), dtype=np.float32)
    assert facegate.measure_face_texture(_textured_image(), empty).shape == (0,)


# ── gaze_measurable ──────────────────────────────────────────────────────────


def test_gate_resolution_floor():
    small = np.stack([_frontal_kpts(ie=4.0)])  # below the 7 px floor
    tex = np.array([1e4], dtype=np.float32)  # texture alone would pass
    assert not facegate.gaze_measurable(small, tex)[0]


def test_gate_texture_floor():
    kpts = np.stack([_frontal_kpts(ie=12.0)])
    assert facegate.gaze_measurable(kpts, np.array([200.0]))[0]
    assert not facegate.gaze_measurable(kpts, np.array([10.0]))[0]
    assert not facegate.gaze_measurable(kpts, np.array([np.nan]))[0]


def test_gate_geometry_only_on_stale_cache():
    """face_texture=None (pre-gate cache) → resolution floor only."""
    kpts = np.stack([_frontal_kpts(ie=12.0), _frontal_kpts(ie=4.0)])
    mask = facegate.gaze_measurable(kpts, None)
    assert mask.tolist() == [True, False]


# ── compute_gaze integration ─────────────────────────────────────────────────


def test_compute_gaze_all_measurable_is_active():
    kpts = np.stack([_frontal_kpts()])
    res = compute_gaze(kpts, face_texture=np.array([500.0]))
    assert res.status is SubScoreStatus.ACTIVE
    assert res.measurable.tolist() == [True]
    assert res.values[0] > 0.0


def test_compute_gaze_unmeasurable_scores_zero_and_falls_back():
    kpts = np.stack([_frontal_kpts(cx=30), _frontal_kpts(cx=70)])
    tex = np.array([500.0, 10.0])  # second face: defaced/textureless
    res = compute_gaze(kpts, face_texture=tex)
    assert res.status is SubScoreStatus.FALLBACK
    assert "1/2 faces unmeasurable" in res.reason
    assert res.measurable.tolist() == [True, False]
    assert res.values[0] > 0.0
    assert res.values[1] == 0.0


def test_compute_gaze_gate_thresholds_are_parameters():
    kpts = np.stack([_frontal_kpts(ie=12.0)])
    res = compute_gaze(
        kpts, face_texture=np.array([10.0]), gate_min_face_texture=5.0
    )
    assert res.status is SubScoreStatus.ACTIVE


# ── extract_features + cache round-trip ──────────────────────────────────────


def _primitives(kpts, face_texture):
    n = len(kpts)
    return CachedFrameFeatures(
        human_bboxes=[[10.0 * i, 10.0, 10.0 * i + 8.0, 60.0] for i in range(n)],
        keypoints_np=np.stack(kpts) if n else None,
        track_ids=[None] * n,
        bbox_depths_m=[2.0] * n,
        depth_viz=np.zeros((100, 100), dtype=np.uint8),
        face_texture_np=face_texture,
    )


def test_extract_features_carries_measurability_mask():
    prim = _primitives(
        [_frontal_kpts(cx=30), _frontal_kpts(cx=70)],
        np.array([500.0, 10.0], dtype=np.float32),
    )
    ext = extract_features(prim, image_shape=(100, 100))
    assert ext.gaze_measurable.tolist() == [True, False]
    assert ext.subscore_status["gaze"] is SubScoreStatus.FALLBACK
    assert ext.features["gaze"][1] == 0.0
    assert ext.bbox_depths_m == [2.0, 2.0]  # raw metres pass through


def test_cache_roundtrip_preserves_face_texture(tmp_path):
    prim = _primitives(
        [_frontal_kpts()], np.array([123.0], dtype=np.float32)
    )
    path = tmp_path / "frame.npz"
    prim.to_npz(path)
    loaded = CachedFrameFeatures.from_npz(path)
    assert loaded.face_texture_np is not None
    assert loaded.face_texture_np[0] == pytest.approx(123.0)


def test_cache_backward_compatible_without_texture(tmp_path):
    """An npz written by the pre-gate code (no texture keys) loads as None."""
    path = tmp_path / "old.npz"
    np.savez_compressed(
        path,
        bboxes=np.zeros((1, 4), dtype=np.float32),
        keypoints=np.stack([_frontal_kpts()]),
        keypoints_present=np.array([True]),
        track_ids=np.array([-1], dtype=np.int64),
        bbox_depths_m=np.array([2.0], dtype=np.float32),
        depth_viz=np.zeros((10, 10), dtype=np.uint8),
    )
    loaded = CachedFrameFeatures.from_npz(path)
    assert loaded.face_texture_np is None
    ext = extract_features(loaded, image_shape=(100, 100))
    # Geometry-only gating: the 12 px frontal face passes.
    assert ext.gaze_measurable.tolist() == [True]
