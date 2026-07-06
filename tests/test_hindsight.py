"""Tests for riskam.hindsight — synthetic trajectories, no GPU."""

import numpy as np
import pytest

from riskam.feature_cache import CachedFrameFeatures
from riskam.hindsight import (
    OracleParams,
    decode_frame,
    hindsight_columns,
    onset_times,
    smoothed_scene_min_d,
)
from riskam.kinematics import CameraModel

IMAGE_W, IMAGE_H = 640, 480
CAMERA = CameraModel.from_hfov(58.0, IMAGE_W)
D_SAFE = 2.5
FPS = 10.0


def person_frame(depth_m: float, track_id, u_px: float = 320.0) -> CachedFrameFeatures:
    return CachedFrameFeatures(
        human_bboxes=[[u_px - 20, 100, u_px + 20, 300]],
        keypoints_np=None,
        track_ids=[track_id],
        bbox_depths_m=[depth_m],
        depth_viz=np.zeros((IMAGE_H, IMAGE_W), dtype=np.uint8),
    )


def empty_frame() -> CachedFrameFeatures:
    return CachedFrameFeatures(
        human_bboxes=[], keypoints_np=None, track_ids=[],
        bbox_depths_m=[], depth_viz=np.zeros((IMAGE_H, IMAGE_W), dtype=np.uint8),
    )


def approach_frames(d0: float, v: float, n: int, track_id=1, fps: float = FPS):
    """A person on the optical axis approaching at v m/s from d0."""
    frames = []
    for i in range(n):
        t = i / fps
        d = d0 - v * t
        frames.append(
            decode_frame(person_frame(max(d, 0.05), track_id), CAMERA, D_SAFE, t_s=t)
        )
    return frames


def scene_series(frames):
    t = np.array([f.t_s for f in frames])
    return t, smoothed_scene_min_d(frames)


# ── decode_frame ─────────────────────────────────────────────────────────────


def test_decode_on_axis_position():
    obs = decode_frame(person_frame(2.0, 1), CAMERA, D_SAFE, t_s=1.5)
    assert obs.t_s == 1.5
    assert obs.x_m[0] == pytest.approx(2.0)
    assert obs.y_m[0] == pytest.approx(0.0, abs=1e-9)
    assert not obs.deadzone[0]


def test_decode_far_sentinel_censored():
    obs = decode_frame(person_frame(D_SAFE, 1), CAMERA, D_SAFE)
    assert np.isnan(obs.x_m[0])


def test_decode_deadzone_is_zero_distance():
    obs = decode_frame(person_frame(0.0, 1), CAMERA, D_SAFE)
    assert obs.deadzone[0]
    assert obs.d_planar_m[0] == 0.0


# ── oracle event cells ───────────────────────────────────────────────────────


def test_approach_triggers_event_at_right_time():
    # 1 m/s from 4.0 m: crosses 1.0 m at t = 3.0 s.
    frames = approach_frames(4.0, 1.0, 45)
    t, min_d = scene_series(frames)
    cols = hindsight_columns(t, min_d)
    ev = cols["event_r10_T2s"]
    # Positive from ~1.0 s (2 s lookahead reaches the 3.0 s crossing).
    first_pos = t[np.flatnonzero(ev)[0]]
    assert first_pos == pytest.approx(1.0, abs=2.5 / FPS)
    # Well before that, negative.
    assert ev[0] == 0
    # t_to_onset decreases linearly toward the crossing.
    tto = cols["t_to_onset_r10"]
    assert tto[0] == pytest.approx(3.0, abs=0.3)
    assert tto[10] == pytest.approx(tto[0] - 1.0, abs=0.15)


def test_receding_person_no_events():
    frames = []
    for i in range(40):
        t = i / FPS
        frames.append(decode_frame(person_frame(2.0 + 0.5 * t, 1), CAMERA, D_SAFE, t_s=t))
    t, min_d = scene_series(frames)
    cols = hindsight_columns(t, min_d)
    for r in ("r05", "r10", "r15"):
        for T in ("T1s", "T2s", "T3s"):
            assert cols[f"event_{r}_{T}"].sum() == 0


def test_far_sentinel_creates_no_event():
    frames = []
    for i in range(40):
        # Person "jumps" to the far sentinel — censored, not a 2.5 m reading.
        frames.append(decode_frame(person_frame(D_SAFE, 1), CAMERA, D_SAFE, t_s=i / FPS))
    t, min_d = scene_series(frames)
    assert np.all(np.isinf(min_d))
    cols = hindsight_columns(t, min_d)
    assert cols["event_r15_T3s"].sum() == 0


def test_deadzone_triggers_all_radii():
    frames = [decode_frame(empty_frame(), CAMERA, D_SAFE, t_s=i / FPS) for i in range(20)]
    frames.append(decode_frame(person_frame(0.0, 1), CAMERA, D_SAFE, t_s=20 / FPS))
    t, min_d = scene_series(frames)
    cols = hindsight_columns(t, min_d)
    for r in ("r05", "r10", "r15"):
        assert cols[f"event_{r}_T3s"][0] == 1  # 2.0 s ahead is within T=3
        assert cols[f"in_event_{r}"][-1] == 1


def test_id_switch_mid_approach_still_scene_event():
    # Track 1 approaches to 2.0 m then the ID switches to 7.
    frames = approach_frames(4.0, 1.0, 20, track_id=1)
    for i in range(20, 45):
        t = i / FPS
        d = max(4.0 - t, 0.05)
        frames.append(decode_frame(person_frame(d, 7), CAMERA, D_SAFE, t_s=t))
    t, min_d = scene_series(frames)
    cols = hindsight_columns(t, min_d)
    assert cols["event_r10_T2s"].sum() > 0


def test_untracked_detection_contributes():
    frames = [decode_frame(person_frame(0.8, None), CAMERA, D_SAFE, t_s=0.0)]
    _, min_d = scene_series(frames)
    assert min_d[0] == pytest.approx(0.8, abs=1e-6)


# ── coverage and truncation ──────────────────────────────────────────────────


def test_end_of_run_coverage_shrinks():
    frames = approach_frames(10.0, 0.0, 30)
    t, min_d = scene_series(frames)
    cols = hindsight_columns(t, min_d)
    cov = cols["oracle_cov_T2s"]
    assert cov[0] == pytest.approx(1.0, abs=0.1)
    assert cov[-1] == 0.0  # no frames after the last one
    assert cov[-10] < 0.6  # deep inside the truncated zone


def test_frame_drop_gap_lowers_coverage():
    frames = approach_frames(10.0, 0.0, 20)
    # 2-second hole, then more frames.
    for i in range(20):
        t = 4.0 + i / FPS
        frames.append(decode_frame(person_frame(10.0, 1), CAMERA, D_SAFE, t_s=t))
    t, min_d = scene_series(frames)
    cols = hindsight_columns(t, min_d)
    cov = cols["oracle_cov_T2s"]
    # Frames just before the hole see few lookahead frames.
    i_pre_gap = 18
    assert cov[i_pre_gap] < 0.5


# ── onsets and hysteresis ────────────────────────────────────────────────────


def test_onset_hysteresis_debounces():
    t = np.arange(10) * 0.1
    # Chatter around r=1.0 without ever releasing (never exceeds 1.2).
    d = np.array([2.0, 0.9, 1.05, 0.95, 1.1, 0.9, 1.15, 0.95, 2.0, 2.0])
    onsets = onset_times(t, d, r=1.0, hysteresis_m=0.2)
    assert len(onsets) == 1
    # With zero hysteresis the chatter creates multiple onsets.
    assert len(onset_times(t, d, r=1.0, hysteresis_m=0.0)) > 1


def test_run_starting_inside_event_has_onset_at_t0():
    t = np.arange(5) * 0.1
    d = np.array([0.5, 0.5, 2.0, 2.0, 2.0])
    onsets = onset_times(t, d, r=1.0, hysteresis_m=0.2)
    assert onsets[0] == t[0]


# ── the circularity firewall ─────────────────────────────────────────────────


def test_import_firewall():
    """hindsight must not touch the causal scoring stack."""
    import riskam.hindsight as h

    src = open(h.__file__, encoding="utf-8").read()
    for forbidden in ("KinematicTracker", "RiskScorer", "riskam.score",
                      "riskam.ssm", "riskam.ml"):
        assert forbidden not in src.replace(
            "``riskam.kinematics.KinematicTracker``, ``riskam.score``, "
            "``riskam.ml`` or\n``riskam.ssm``", ""
        ), forbidden
