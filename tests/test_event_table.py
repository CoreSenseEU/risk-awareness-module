"""Tests for riskam.event_table on a synthetic feature cache (no GPU)."""

from pathlib import Path

import numpy as np
import pytest

from riskam.event_table import (
    build_event_table,
    event_columns,
    read_event_table_csv,
    write_event_table_csv,
)
from riskam.feature_cache import CachedFrameFeatures, FeatureCache
from riskam.kinematics import CameraModel, KinematicParams

IMAGE_W, IMAGE_H = 640, 480
CAMERA = CameraModel.from_hfov(58.0, IMAGE_W)
PARAMS = KinematicParams(d_safe_m=2.5)


def frontal_keypoints(n: int) -> np.ndarray:
    """COCO keypoints for n persons facing the camera (gaze ≈ 1)."""
    kpts = np.zeros((n, 17, 2), dtype=np.float32)
    for i in range(n):
        kpts[i, 0] = (100, 107)  # nose at eye-mid x, frontal pitch ratio
        kpts[i, 1] = (105, 100)  # left eye
        kpts[i, 2] = (95, 100)   # right eye
    return kpts


def person_frame(depth_m: float, track_id, u_px: float = 320.0) -> CachedFrameFeatures:
    return CachedFrameFeatures(
        human_bboxes=[[u_px - 20, 100, u_px + 20, 300]],
        keypoints_np=frontal_keypoints(1),
        track_ids=[track_id],
        bbox_depths_m=[depth_m],
        depth_viz=np.zeros((IMAGE_H, IMAGE_W), dtype=np.uint8),
    )


def empty_frame() -> CachedFrameFeatures:
    return CachedFrameFeatures(
        human_bboxes=[], keypoints_np=None, track_ids=[],
        bbox_depths_m=[], depth_viz=np.zeros((IMAGE_H, IMAGE_W), dtype=np.uint8),
    )


@pytest.fixture
def populated(tmp_path):
    """Run A: person approaching 3.0 → 2.6 m, then an empty frame.

    Only the first frame is labelled — the event table must emit rows for
    all frames regardless.
    """
    cache = FeatureCache(tmp_path, "testsha", "toy")
    stems = ["100.0", "100.2", "100.4"]
    for stem, d in zip(stems, [3.0, 2.8, 2.6]):
        cache.put("A", stem, person_frame(d, track_id=1))
    cache.put("A", "100.6", empty_frame())

    gt_per_run = {"A": {"100.0.png": 2}}
    frames_per_run = {
        "A": [Path(f"/fake/A/rgb/{s}.png") for s in [*stems, "100.6"]]
    }
    return cache, frames_per_run, gt_per_run


def build(cache, frames, gt=None):
    return build_event_table(
        cache=cache, frames_per_run=frames, gt_per_run=gt,
        kin_params=PARAMS, camera_by_run={"A": CAMERA},
    )


def test_all_frames_get_rows_even_unlabelled(populated):
    cache, frames, gt = populated
    rows = build(cache, frames, gt)
    assert len(rows) == 4
    assert [r["is_labelled"] for r in rows] == [1, 0, 0, 0]
    assert rows[0]["gt"] == 2
    assert rows[1]["gt"] == -1


def test_columns_complete(populated):
    rows = build(*populated[:2])
    cols = set(event_columns())
    for row in rows:
        assert set(row) == cols


def test_oracle_and_causal_share_frames(populated):
    rows = build(*populated[:2])
    # d_oracle_now should track the causal nearest distance closely
    # (same measurements, smoothing direction differs).
    for row in rows[:3]:
        assert row["d_oracle_now_m"] == pytest.approx(row["nearest_d_m"], abs=0.15)
    # Empty frame: both sides say "nothing there".
    assert np.isinf(rows[3]["d_oracle_now_m"])
    assert np.isinf(rows[3]["nearest_d_m"])


def test_ssm_margins_present_and_ordered(populated):
    rows = build(*populated[:2])
    for row in rows[:3]:
        assert row["ssm_margin_aware"] >= row["ssm_margin_worst"]
    assert np.isinf(rows[3]["ssm_margin_worst"])


def test_no_events_far_person(populated):
    rows = build(*populated[:2])
    # Person never below 2.6 m: no oracle events at any (r, T).
    for row in rows:
        assert row["event_r15_T3s"] == 0


def test_close_approach_creates_event(tmp_path):
    cache = FeatureCache(tmp_path, "testsha", "toy")
    stems, frames = [], []
    for i in range(30):
        t = 100.0 + i * 0.1
        d = max(3.0 - 0.2 * i, 0.2)  # reaches 1.0 m at i=10, 0.2 m floor
        stem = f"{t:.1f}"
        cache.put("A", stem, person_frame(d, track_id=1))
        frames.append(Path(f"/fake/A/rgb/{stem}.png"))
    rows = build_event_table(
        cache=cache, frames_per_run={"A": frames},
        kin_params=PARAMS, camera_by_run={"A": CAMERA},
    )
    events = [r["event_r10_T2s"] for r in rows]
    assert sum(events) > 0
    # The first frame already sees the crossing at t+1.0 s within T=2.
    assert events[0] == 1
    in_ev = [r["in_event_r10"] for r in rows]
    assert in_ev[0] == 0 and sum(in_ev) > 0


def test_csv_round_trip(populated, tmp_path):
    rows = build(*populated[:2])
    path = tmp_path / "event_table.csv"
    write_event_table_csv(rows, path)
    back = read_event_table_csv(path)
    assert len(back) == len(rows)
    assert back[0]["run"] == "A"
    assert back[0]["t_s"] == pytest.approx(rows[0]["t_s"])
    assert back[0]["event_r10_T2s"] == pytest.approx(float(rows[0]["event_r10_T2s"]))
    assert back[1]["gt"] == -1
    # inf survives the round trip
    assert np.isinf(back[3]["nearest_d_m"])
