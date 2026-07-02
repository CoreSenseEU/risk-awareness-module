"""Tests for riskam.scene_table on a synthetic feature cache (no GPU)."""

from pathlib import Path

import numpy as np
import pytest

from riskam.data.splits import Split
from riskam.feature_cache import CachedFrameFeatures, FeatureCache
from riskam.kinematics import CameraModel, KinematicParams, PlanarTwist
from riskam.scene_table import (
    SCENE_COLUMNS,
    build_scene_table,
    read_table_csv,
    write_table_csv,
)

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


class StubOdom:
    def __init__(self, speed: float):
        self._speed = speed

    def __bool__(self) -> bool:
        return True

    def twist_for(self, t_s: float) -> PlanarTwist:
        return PlanarTwist(t_s=t_s, vx_ms=self._speed, vy_ms=0.0)


@pytest.fixture
def populated(tmp_path):
    """Two runs: A = approaching person + an empty frame; B = untracked person."""
    cache = FeatureCache(tmp_path, "testsha", "toy")
    frames_a = {"100.0": 3.0, "100.2": 2.8, "100.4": 2.6}
    for stem, d in frames_a.items():
        cache.put("A", stem, person_frame(d, track_id=1))
    cache.put("A", "100.6", empty_frame())
    cache.put("B", "200.0", person_frame(2.0, track_id=None))

    gt_per_run = {
        "A": {f"{s}.png": 2 for s in [*frames_a, "100.6"]},
        "B": {"200.0.png": 1},
    }
    frames_per_run = {
        run: [Path(f"/fake/{run}/rgb/{name}") for name in sorted(gt)]
        for run, gt in gt_per_run.items()
    }
    return cache, gt_per_run, frames_per_run


def build(cache, gt, frames, split=None, odom=None):
    return build_scene_table(
        cache=cache, gt_per_run=gt, frames_per_run=frames, split_obj=split,
        kin_params=PARAMS,
        camera_by_run={"A": CAMERA, "B": CAMERA},
        odom_by_run=odom,
    )


class TestBuild:
    def test_one_row_per_cached_frame(self, populated):
        rows = build(*populated)
        assert len(rows) == 5
        assert [set(r) == set(SCENE_COLUMNS) for r in rows]

    def test_approaching_person_gets_full_kinematics(self, populated):
        rows = build(*populated)
        third = rows[2]  # A/100.4: two prior observations of track 1
        assert third["kin_status"] == "full"
        assert third["closing_ms"] == pytest.approx(1.0, rel=1e-5)
        assert third["hazard"] > rows[0]["hazard"]  # first frame was static
        assert third["n_humans"] == 1
        # Frontal face → high awareness → fused risk ≈ hazard.
        assert third["awareness"] > 0.9
        assert third["risk_a"] == pytest.approx(
            third["hazard"] * (1 + PARAMS.beta * (1 - third["awareness"])),
            rel=1e-9,
        )

    def test_no_human_sentinel_row(self, populated):
        rows = build(*populated)
        sentinel = rows[3]  # A/100.6, right after a risky frame
        assert sentinel["n_humans"] == 0
        assert sentinel["kin_status"] == "no_human"
        assert sentinel["hazard"] == 0.0
        assert sentinel["awareness"] == 1.0
        assert sentinel["d_m"] == 2 * PARAMS.d_safe_m
        # Scene release bridges the (possible) detector dropout: risk_a
        # decays from the previous frame instead of snapping to zero.
        import math
        expected = rows[2]["risk_a"] * math.exp(-0.2 / PARAMS.release_tau_s)
        assert sentinel["risk_a"] == pytest.approx(expected, rel=1e-9)

    def test_runs_are_isolated(self, populated):
        rows = build(*populated)
        b_row = rows[4]  # B/200.0: fresh tracker, single untracked obs
        assert b_row["run"] == "B"
        assert b_row["kin_status"] == "static"
        # Static degradation: hazard = 1 − r/d_safe at r = 2 m.
        assert b_row["hazard"] == pytest.approx(1 - 2.0 / 2.5, rel=1e-6)

    def test_bucket_column_follows_split(self, populated):
        cache, gt, frames = populated
        split = Split(
            test_fraction=0.3, seed=1,
            assignments={
                "A": {"100.0.png": "val", "100.2.png": "test",
                      "100.4.png": "val", "100.6.png": "val"},
                "B": {"200.0.png": "test"},
            },
        )
        rows = build(cache, gt, frames, split=split)
        assert [r["bucket"] for r in rows] == ["val", "test", "val", "val", "test"]

    def test_no_split_gives_empty_bucket(self, populated):
        rows = build(*populated)
        assert {r["bucket"] for r in rows} == {""}

    def test_odom_populates_ego_columns(self, populated):
        cache, gt, frames = populated
        rows = build(cache, gt, frames, odom={"A": StubOdom(0.5), "B": None})
        a_rows = [r for r in rows if r["run"] == "A"]
        b_rows = [r for r in rows if r["run"] == "B"]
        assert all(r["ego_available"] == 1 for r in a_rows)
        assert all(r["v_robot_speed_ms"] == pytest.approx(0.5) for r in a_rows)
        assert all(r["ego_available"] == 0 for r in b_rows)
        assert all(r["v_robot_speed_ms"] is None for r in b_rows)
        # Single-obs frame with ego → static-human odometry fallback.
        assert a_rows[0]["kin_status"] == "static_odom"

    def test_skips_unpopulated_frames(self, populated):
        cache, gt, frames = populated
        gt["A"]["999.0.png"] = 3  # labelled but never cached
        frames["A"].append(Path("/fake/A/rgb/999.0.png"))
        rows = build(cache, gt, frames)
        assert len(rows) == 5


class TestCsvRoundTrip:
    def test_roundtrip_preserves_values_and_none(self, populated, tmp_path):
        rows = build(*populated)
        path = tmp_path / "scene_table.csv"
        write_table_csv(rows, path)
        loaded = read_table_csv(path)
        assert len(loaded) == len(rows)
        for orig, back in zip(rows, loaded):
            assert back["run"] == orig["run"]
            assert back["gt"] == orig["gt"]
            assert back["kin_status"] == orig["kin_status"]
            assert back["v_robot_speed_ms"] is None  # no odom in this build
            assert back["hazard"] == pytest.approx(orig["hazard"], rel=1e-9)
            assert back["m0_risk"] == pytest.approx(orig["m0_risk"], rel=1e-9)
