"""Unit tests for riskam.ml.depth."""

import numpy as np
import pytest

from riskam.ml.depth import (
    D_SAFE_DEFAULT,
    DEPTH_MIN_M,
    NEAR_CLIP_BBOX_MIN_FRAC_DEFAULT,
    NEAR_CLIP_M_DEFAULT,
    depth_mm_to_m,
    depth_to_visualization,
    extract_bbox_depths,
    extract_bbox_proximities,
)


class TestDepthMmToM:
    def test_basic_conversion(self):
        img = np.array([[1000, 2000], [500, 0]], dtype=np.uint16)
        out = depth_mm_to_m(img)
        assert out.dtype == np.float32
        assert out[0, 0] == pytest.approx(1.0)
        assert out[0, 1] == pytest.approx(2.0)
        assert out[1, 0] == pytest.approx(0.5)

    def test_zero_preserved(self):
        img = np.array([[0, 1000]], dtype=np.uint16)
        out = depth_mm_to_m(img)
        assert out[0, 0] == 0.0

    def test_output_dtype(self):
        img = np.ones((4, 4), dtype=np.uint16) * 1500
        assert depth_mm_to_m(img).dtype == np.float32


class TestExtractBboxProximities:
    def _depth_image(self, h=100, w=100, fill_m=0.5):
        return np.full((h, w), fill_m, dtype=np.float32)

    def test_very_close_person(self):
        img = self._depth_image(fill_m=0.1)
        scores = extract_bbox_proximities(img, [[10, 10, 50, 50]], d_safe=1.5)
        assert len(scores) == 1
        # 0.1 m → score ≈ 1 - 0.1/1.5 ≈ 0.933
        assert scores[0] == pytest.approx(1.0 - 0.1 / 1.5, abs=0.01)

    def test_person_at_safe_distance(self):
        img = self._depth_image(fill_m=D_SAFE_DEFAULT)
        scores = extract_bbox_proximities(img, [[0, 0, 30, 30]])
        assert scores[0] == pytest.approx(0.0, abs=0.01)

    def test_person_beyond_safe_distance(self):
        img = self._depth_image(fill_m=D_SAFE_DEFAULT + 1.0)
        scores = extract_bbox_proximities(img, [[0, 0, 30, 30]])
        assert scores[0] == 0.0

    def test_no_valid_depth_defaults_to_zero_score(self):
        img = np.zeros((50, 50), dtype=np.float32)  # all missing
        scores = extract_bbox_proximities(img, [[5, 5, 45, 45]])
        assert scores[0] == 0.0

    def test_multiple_bboxes(self):
        img = np.zeros((100, 200), dtype=np.float32)
        img[:, :100] = 0.5   # left half: 0.5 m
        img[:, 100:] = 2.0   # right half: beyond d_safe=1.5
        bboxes = [[0, 0, 80, 80], [110, 0, 190, 80]]
        scores = extract_bbox_proximities(img, bboxes, d_safe=1.5)
        assert scores[0] > 0.5
        assert scores[1] == 0.0

    def test_empty_bboxes(self):
        img = self._depth_image()
        assert extract_bbox_proximities(img, []) == []


class TestNearClipFallback:
    """T2.7: dead-zone fallback for sensors with a hard near-clip (e.g. Xtion).

    Defaults preserve RealSense behaviour exactly:
    ``depth_near_clip_m = 0`` disables the fallback, so a bbox with no valid
    depth reverts to ``d_safe`` (proximity 0) just like before T2.7.
    """

    def _zero_depth(self, h=100, w=100):
        return np.zeros((h, w), dtype=np.float32)

    def test_default_preserves_realsense_fallback(self):
        # No valid depth + defaults → legacy "far" fallback (d_safe).
        img = self._zero_depth()
        depths = extract_bbox_depths(img, [[5, 5, 95, 95]])
        assert depths[0] == D_SAFE_DEFAULT
        assert NEAR_CLIP_M_DEFAULT == 0.0  # contract: 0 = disabled

    def test_close_fallback_fires_for_large_bbox(self):
        # Xtion scenario: zero depth + bbox covers >5% of frame + near-clip > 0
        # → max-proximity fallback. depth=0 ⇒ proximity = 1 - 0/d_safe = 1.
        img = self._zero_depth()
        depths = extract_bbox_depths(
            img,
            [[5, 5, 95, 95]],  # ~81% of the frame, well above the 5% threshold
            depth_near_clip_m=0.6,
        )
        assert depths[0] == 0.0

    def test_close_fallback_does_not_fire_for_small_bbox(self):
        # Small bbox (well below the area threshold) → far fallback even with
        # near-clip > 0. Guards against tiny noise bboxes triggering max risk.
        img = self._zero_depth()
        depths = extract_bbox_depths(
            img,
            [[0, 0, 10, 10]],  # 1% of the frame
            depth_near_clip_m=0.6,
            near_clip_bbox_min_frac=NEAR_CLIP_BBOX_MIN_FRAC_DEFAULT,
        )
        assert depths[0] == D_SAFE_DEFAULT

    def test_close_fallback_threshold_boundary(self):
        # Bbox area exactly equal to near_clip_bbox_min_frac → triggers.
        img = self._zero_depth(h=100, w=100)
        # 32x32 = 1024 px = 10.24% of 10000
        depths = extract_bbox_depths(
            img,
            [[0, 0, 32, 32]],
            depth_near_clip_m=0.6,
            near_clip_bbox_min_frac=0.10,
        )
        assert depths[0] == 0.0

    def test_close_fallback_does_not_override_valid_depth(self):
        # When the bbox has any valid depth, the percentile path runs as
        # before — the close-fallback is no-op. Pinning this so a future
        # change can't accidentally widen the fallback to "preferred path".
        img = np.full((100, 100), 0.5, dtype=np.float32)
        depths = extract_bbox_depths(
            img,
            [[5, 5, 95, 95]],
            depth_near_clip_m=0.6,
        )
        assert depths[0] == pytest.approx(0.5, abs=0.01)

    def test_close_fallback_threads_through_extract_bbox_proximities(self):
        # Xtion deployment: depth_near_clip_m=0.6, d_safe=1.5 → max proximity 1.
        img = np.zeros((100, 100), dtype=np.float32)
        scores = extract_bbox_proximities(
            img,
            [[5, 5, 95, 95]],
            d_safe=1.5,
            depth_near_clip_m=0.6,
        )
        assert scores[0] == pytest.approx(1.0)

    def test_realsense_default_proximities_unchanged(self):
        # Pin the RealSense pre-T2.7 contract: the all-zero bbox returns
        # proximity 0. Any change to defaults that breaks this would silently
        # change SamXL deployment behaviour.
        img = np.zeros((100, 100), dtype=np.float32)
        scores = extract_bbox_proximities(img, [[5, 5, 95, 95]], d_safe=1.5)
        assert scores[0] == 0.0

    def test_sparse_valid_pixels_default_uses_percentile(self):
        # Default ``near_clip_valid_frac_max = 0.0`` keeps the strict-zero
        # condition (RealSense pre-T2.7 contract). A bbox with even a single
        # valid pixel must still go through the percentile path.
        img = np.zeros((100, 100), dtype=np.float32)
        # ~1% of bbox pixels at 2 m, the rest zero/missing.
        img[20, 20:60] = 2.0
        depths = extract_bbox_depths(
            img,
            [[5, 5, 95, 95]],
            depth_near_clip_m=0.6,  # close-fallback enabled but condition unmet
        )
        assert depths[0] == pytest.approx(2.0, abs=0.01)

    def test_sparse_valid_pixels_with_threshold_fires_close_fallback(self):
        # Frame-299-like case (cs_robocup_2023): large bbox, ~1% valid pixels
        # likely background bleed-through. With ``near_clip_valid_frac_max =
        # 0.05``, the close-fallback fires instead of trusting unreliable
        # noise: depth = 0 (max proximity), safety-conservative.
        img = np.zeros((100, 100), dtype=np.float32)
        img[20, 20:60] = 2.0  # ~0.5% of the 90×90 bbox
        depths = extract_bbox_depths(
            img,
            [[5, 5, 95, 95]],
            depth_near_clip_m=0.6,
            near_clip_valid_frac_max=0.05,
        )
        assert depths[0] == 0.0

    def test_dense_valid_pixels_with_threshold_uses_percentile(self):
        # Mid-range cs_robocup_2023 case (e.g. RB_01 at 56% valid). When the
        # bbox really does have a measurable signal, the percentile path
        # must run regardless of the new threshold.
        img = np.full((100, 100), 2.0, dtype=np.float32)
        depths = extract_bbox_depths(
            img,
            [[5, 5, 95, 95]],
            depth_near_clip_m=0.6,
            near_clip_valid_frac_max=0.05,
        )
        assert depths[0] == pytest.approx(2.0, abs=0.01)

    def test_threshold_does_not_override_bbox_size_guard(self):
        # ``near_clip_bbox_min_frac`` still guards: a small bbox with sparse
        # valid signal is more plausibly noise than a close person, so the
        # close-fallback must not fire even with the new looser valid-frac
        # threshold.
        img = np.zeros((100, 100), dtype=np.float32)
        depths = extract_bbox_depths(
            img,
            [[0, 0, 10, 10]],  # 1% of frame, below the default 5%
            depth_near_clip_m=0.6,
            near_clip_valid_frac_max=0.05,
        )
        assert depths[0] == D_SAFE_DEFAULT




class TestDepthToVisualization:
    def test_close_is_bright(self):
        img = np.array([[0.1, 2.0]], dtype=np.float32)
        viz = depth_to_visualization(img, d_safe=1.5)
        # 0.1 m → bright; 2.0 m → dark
        assert viz[0, 0] > viz[0, 1]

    def test_zero_is_black(self):
        img = np.zeros((10, 10), dtype=np.float32)
        viz = depth_to_visualization(img)
        assert np.all(viz == 0)

    def test_output_dtype_and_range(self):
        img = np.random.uniform(0, 3.0, (50, 50)).astype(np.float32)
        img[0, 0] = 0.0
        viz = depth_to_visualization(img)
        assert viz.dtype == np.uint8
        assert viz.min() >= 0
        assert viz.max() <= 255

    def test_realsense_default_pinned_when_all_within_d_safe(self):
        # When every valid pixel is within d_safe, the auto-percentile upper
        # bound collapses back to d_safe — pre-T2.7 / RealSense behaviour
        # must be preserved bit-identical.
        img = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        viz_auto = depth_to_visualization(img, d_safe=1.5)
        # Manual reproduction of the old formula:
        upper = 1.5
        normalized = np.clip(img / upper, 0.0, 1.0)
        inverted = 1.0 - normalized
        inverted[img == 0] = 0.0
        expected = (inverted * 255).astype(np.uint8)
        np.testing.assert_array_equal(viz_auto, expected)

    def test_xtion_like_scene_produces_depth_varied_viz(self):
        # Xtion failure mode: all valid pixels well beyond d_safe, plus heavy
        # zero-fill. Pre-T2.7 viz cliffed at d_safe → near-uniform black.
        # Auto-percentile upper expands so the viz is depth-varied.
        img = np.zeros((100, 100), dtype=np.float32)
        img[:50, :] = 2.5
        img[50:, :] = 4.0
        viz = depth_to_visualization(img, d_safe=1.5)
        # Closer half is brighter than the farther half.
        assert viz[:50, :].mean() > viz[50:, :].mean()
        # Viz exhibits real depth contrast (>20 grey-levels), not a single shade.
        assert np.ptp(viz[img > 0]) > 20
