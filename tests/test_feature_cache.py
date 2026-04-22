"""Unit tests for riskam.feature_cache and the cache-aware featextr paths."""

from pathlib import Path

import numpy as np
import pytest

from riskam.feature_cache import CachedFrameFeatures, FeatureCache, model_sha


# ── CachedFrameFeatures round-trip ───────────────────────────────────────────


class TestCachedFrameFeaturesRoundTrip:
    def _one_person(self) -> CachedFrameFeatures:
        return CachedFrameFeatures(
            human_bboxes=[[10, 20, 100, 200]],
            keypoints_np=np.arange(17 * 2, dtype=np.float32).reshape(1, 17, 2),
            track_ids=[7],
            bbox_depths_m=[1.2],
            depth_viz=(np.random.rand(32, 48) * 255).astype(np.uint8),
        )

    def _no_persons(self) -> CachedFrameFeatures:
        return CachedFrameFeatures(
            human_bboxes=[],
            keypoints_np=None,
            track_ids=[],
            bbox_depths_m=[],
            depth_viz=np.zeros((10, 10), dtype=np.uint8),
        )

    def test_round_trip_one_person(self, tmp_path):
        orig = self._one_person()
        p = tmp_path / "a.npz"
        orig.to_npz(p)
        loaded = CachedFrameFeatures.from_npz(p)
        # Float32 round-trip is lossy at the 1e-7 level; use approx matching.
        assert np.allclose(loaded.human_bboxes, orig.human_bboxes, atol=1e-5)
        assert np.allclose(loaded.keypoints_np, orig.keypoints_np)
        assert loaded.track_ids == orig.track_ids
        assert np.allclose(loaded.bbox_depths_m, orig.bbox_depths_m, atol=1e-5)
        assert np.array_equal(loaded.depth_viz, orig.depth_viz)

    def test_round_trip_no_persons(self, tmp_path):
        orig = self._no_persons()
        p = tmp_path / "b.npz"
        orig.to_npz(p)
        loaded = CachedFrameFeatures.from_npz(p)
        assert loaded.human_bboxes == []
        assert loaded.keypoints_np is None
        assert loaded.track_ids == []
        assert loaded.bbox_depths_m == []

    def test_round_trip_preserves_none_track_ids(self, tmp_path):
        orig = CachedFrameFeatures(
            human_bboxes=[[0, 0, 10, 10], [20, 20, 30, 30]],
            keypoints_np=np.zeros((2, 17, 2), dtype=np.float32),
            track_ids=[3, None],
            bbox_depths_m=[1.0, 1.5],
            depth_viz=np.zeros((4, 4), dtype=np.uint8),
        )
        p = tmp_path / "c.npz"
        orig.to_npz(p)
        loaded = CachedFrameFeatures.from_npz(p)
        assert loaded.track_ids == [3, None]


# ── FeatureCache ─────────────────────────────────────────────────────────────


class TestFeatureCache:
    def _sample(self) -> CachedFrameFeatures:
        return CachedFrameFeatures(
            human_bboxes=[[0, 0, 10, 10]],
            keypoints_np=np.zeros((1, 17, 2), dtype=np.float32),
            track_ids=[1],
            bbox_depths_m=[0.5],
            depth_viz=np.zeros((5, 5), dtype=np.uint8),
        )

    def test_miss_then_hit(self, tmp_path):
        cache = FeatureCache(tmp_path, "sha", "cs_robocup_2023")
        assert cache.get("RB_01", "frame_1") is None
        assert cache.misses == 1
        cache.put("RB_01", "frame_1", self._sample())
        got = cache.get("RB_01", "frame_1")
        assert got is not None
        assert cache.hits == 1
        assert cache.hit_rate() == pytest.approx(0.5)

    def test_different_model_sha_isolates_caches(self, tmp_path):
        c1 = FeatureCache(tmp_path, "sha_a", "ds")
        c2 = FeatureCache(tmp_path, "sha_b", "ds")
        c1.put("RB_01", "frame_1", self._sample())
        assert c2.get("RB_01", "frame_1") is None

    def test_different_dataset_isolates_caches(self, tmp_path):
        c1 = FeatureCache(tmp_path, "sha", "ds_a")
        c2 = FeatureCache(tmp_path, "sha", "ds_b")
        c1.put("RB_01", "frame_1", self._sample())
        assert c2.get("RB_01", "frame_1") is None


# ── model_sha ────────────────────────────────────────────────────────────────


class TestModelSha:
    def test_deterministic_and_short(self, tmp_path):
        p = tmp_path / "model.pt"
        p.write_bytes(b"abc" * 1024)
        sha_a = model_sha(p)
        sha_b = model_sha(p)
        assert sha_a == sha_b
        assert len(sha_a) == 16

    def test_different_content_different_sha(self, tmp_path):
        a = tmp_path / "a.pt"
        b = tmp_path / "b.pt"
        a.write_bytes(b"content_one")
        b.write_bytes(b"content_two")
        assert model_sha(a) != model_sha(b)


# ── cache-aware featextr flow ────────────────────────────────────────────────


class TestExtractWithCache:
    def _patch_detection(self, monkeypatch, primitives_counter: dict):
        """Replace humandet.detect_humans with a counting stub."""
        import numpy as np

        from riskam.ml import featextr as featextr_mod

        def fake_detect(image, track_bboxes):
            primitives_counter["calls"] = primitives_counter.get("calls", 0) + 1
            return (
                [[10, 20, 100, 200]],
                np.zeros((1, 17, 2), dtype=np.float32),
                [1],
            )

        monkeypatch.setattr(featextr_mod.humandet, "detect_humans", fake_detect)

    def test_cache_miss_populates_and_hit_skips_detection(
        self, tmp_path, monkeypatch
    ):
        from riskam.ml import featextr as featextr_mod
        from riskam.ml.subscores import FrameInputs

        counter: dict = {}
        self._patch_detection(monkeypatch, counter)

        cache = FeatureCache(tmp_path, "sha", "ds_test")
        rgb = np.zeros((80, 120, 3), dtype=np.uint8)
        depth = np.full((80, 120), 0.5, dtype=np.float32)
        inputs = FrameInputs(rgb=rgb, depth_m=depth, cmd_vel=None)

        r1 = featextr_mod.extract_with_cache(inputs, cache, "RB_01", "f1")
        r2 = featextr_mod.extract_with_cache(inputs, cache, "RB_01", "f1")

        assert counter["calls"] == 1  # detection ran once; second call was a hit
        assert cache.hits == 1 and cache.misses == 1
        # Both results should have the same bboxes.
        assert r1.human_bboxes == r2.human_bboxes

    def test_extract_primitives_and_features_match_extract(
        self, tmp_path, monkeypatch
    ):
        """Regression check: the split pipeline should produce the same
        features dict as the one-shot ``extract`` for the same inputs."""
        from riskam.ml import featextr as featextr_mod
        from riskam.ml.subscores import FrameInputs

        counter: dict = {}
        self._patch_detection(monkeypatch, counter)

        rgb = np.zeros((80, 120, 3), dtype=np.uint8)
        depth = np.full((80, 120), 0.5, dtype=np.float32)
        inputs = FrameInputs(rgb=rgb, depth_m=depth, cmd_vel=None)

        oneshot = featextr_mod.extract(inputs)
        primitives = featextr_mod.extract_primitives(inputs)
        split = featextr_mod.extract_features(
            primitives, image_shape=inputs.rgb.shape[:2], cmd_vel=None
        )

        assert oneshot.human_bboxes == split.human_bboxes
        # The features dict should have the same keys and values (approach may
        # differ because update_velocity has persistent state — call reset).
        for k in ("proximity", "gaze", "x_offset"):
            assert np.allclose(oneshot.features[k], split.features[k])
