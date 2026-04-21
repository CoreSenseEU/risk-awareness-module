"""Unit tests for riskam.ml.depth."""

import numpy as np
import pytest

from riskam.ml.depth import (
    D_SAFE_DEFAULT,
    DEPTH_MIN_M,
    depth_mm_to_m,
    depth_to_visualization,
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
