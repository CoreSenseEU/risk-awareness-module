"""Unit tests for riskam.visualization.

Two pinned semantics:
- Bbox **border** is a continuous red↔white gradient by gaze score
  (introduced when T1.3 made gaze a continuous Gaussian; the previous
  strict-equality logic painted everything yellow).
- Bbox **interior** is a red overlay with opacity proportional to the
  proximity sub-score (red = close = danger, transparent = far = safe).
  Per-bbox uniform — driven by the per-bbox proximity value, not by the
  raw depth field. This sidesteps Xtion-class sensors' heavy zero-fill.
"""

import numpy as np

from riskam.visualization import visualize_risk


def _bbox_pixels(img, bbox):
    x1, y1, x2, y2 = bbox
    # Pixels strictly *on* the rectangle border (not the black outline beneath).
    border = []
    for x in range(x1, x2):
        border.append(img[y1, x])
        border.append(img[y2 - 1, x])
    for y in range(y1, y2):
        border.append(img[y, x1])
        border.append(img[y, x2 - 1])
    return np.array(border)


def _features(gaze, proximity=None):
    n = len(gaze)
    if proximity is None:
        proximity = np.zeros(n, dtype=float)
    return {
        "gaze": np.asarray(gaze, dtype=float),
        "proximity": np.asarray(proximity, dtype=float),
        "x_offset": np.zeros(n, dtype=float),
        "approach": np.full(n, 0.5, dtype=float),
    }


class TestBboxGazeGradient:
    def _blank(self):
        return np.full((100, 200, 3), 128, dtype=np.uint8)

    def test_gaze_zero_is_red(self):
        img = self._blank()
        bbox = (40, 30, 80, 70)
        out = visualize_risk(
            img, [bbox], risk_features=_features([0.0]),
            risk_score=0.0, max_risk_idx=-1,
        )
        pixels = _bbox_pixels(out, bbox)
        red = (pixels == np.array([0, 0, 255])).all(axis=1)
        assert red.any(), "gaze=0 must render at least one pure-red border pixel"

    def test_gaze_one_is_white(self):
        img = self._blank()
        bbox = (40, 30, 80, 70)
        out = visualize_risk(
            img, [bbox], risk_features=_features([1.0]),
            risk_score=0.0, max_risk_idx=-1,
        )
        pixels = _bbox_pixels(out, bbox)
        white = (pixels == np.array([255, 255, 255])).all(axis=1)
        assert white.any(), "gaze=1 must render at least one white border pixel"

    def test_gradient_monotonic(self):
        img = self._blank()
        bbox = (40, 30, 80, 70)
        out = visualize_risk(
            img, [bbox], risk_features=_features([0.5]),
            risk_score=0.0, max_risk_idx=-1,
        )
        pixels = _bbox_pixels(out, bbox)
        on_stroke = pixels[pixels[:, 2] == 255]
        assert on_stroke.size > 0
        # B and G should sit roughly at 127 (255*0.5 rounded).
        assert (on_stroke[:, 0] == 128).any() or (on_stroke[:, 0] == 127).any()

    def test_clipped_out_of_range_score(self):
        img = self._blank()
        bbox = (40, 30, 80, 70)
        for gaze, target in (
            (1.5, np.array([255, 255, 255])),
            (-0.2, np.array([0, 0, 255])),
        ):
            out = visualize_risk(
                img.copy(), [bbox],
                risk_features=_features([gaze]),
                risk_score=0.0, max_risk_idx=-1,
            )
            pixels = _bbox_pixels(out, bbox)
            assert (pixels == target).all(axis=1).any(), (
                f"gaze={gaze} should clamp to BGR {tuple(target)}"
            )

    def test_black_outline_is_drawn(self):
        img = np.full((100, 200, 3), 255, dtype=np.uint8)  # white background
        bbox = (40, 30, 80, 70)
        out = visualize_risk(
            img, [bbox], risk_features=_features([1.0]),
            risk_score=0.0, max_risk_idx=-1,
        )
        x1, y1, x2, y2 = bbox
        border_band = out[max(0, y1 - 2):y2 + 2, max(0, x1 - 2):x2 + 2]
        assert ((border_band == 0).all(axis=2)).any(), (
            "outline missing — white-on-white bbox would be invisible"
        )


class TestProximityRedInterior:
    """Bbox-interior alpha = proximity · 0.7. Red = close, translucent = far.

    Per-bbox uniform colour (driven by the proximity sub-score, not raw
    depth_viz pixels) — keeps Xtion-class zero-fill noise out of the overlay.
    """

    def _scene(self):
        # Mid-grey background — easy to detect any pull toward red.
        return np.full((100, 200, 3), 120, dtype=np.uint8)

    def _interior_pixel(self, bbox, inset=6):
        # Pixel a few px inside the bbox: clear of both the 4-px black outline
        # and the 2-px coloured border stroke.
        x1, y1, x2, y2 = bbox
        return (y1 + inset, x1 + inset)

    def test_proximity_zero_means_translucent(self):
        # No proximity → no red overlay → bbox interior pixels (well inside,
        # past the bbox border) match the original scene.
        img = self._scene()
        bbox = (60, 30, 140, 80)
        out = visualize_risk(
            img.copy(), [bbox],
            risk_features=_features([0.0], proximity=[0.0]),
            risk_score=0.0, max_risk_idx=-1,
        )
        py, px = self._interior_pixel(bbox)
        np.testing.assert_array_equal(out[py, px], np.array([120, 120, 120]))

    def test_proximity_one_pulls_strongly_red(self):
        # alpha = 1 * 0.7 = 0.7 → interior pixel = 0.3*120 + 0.7*(0,0,255)
        #                                       ≈ (36, 36, 215).
        img = self._scene()
        bbox = (60, 30, 140, 80)
        out = visualize_risk(
            img.copy(), [bbox],
            risk_features=_features([0.0], proximity=[1.0]),
            risk_score=0.0, max_risk_idx=-1,
        )
        py, px = self._interior_pixel(bbox)
        b, g, r = out[py, px]
        assert r > 200, "max-proximity bbox interior must be strongly red"
        assert b < 60 and g < 60

    def test_alpha_proportional_to_proximity(self):
        # Two bboxes with proximity 0.0 vs 0.5 → second one is more red than
        # the first (which is the original mid-grey).
        img = self._scene()
        bbox_far = (10, 10, 60, 60)
        bbox_close = (110, 10, 160, 60)
        out = visualize_risk(
            img.copy(), [bbox_far, bbox_close],
            risk_features=_features([0.0, 0.0], proximity=[0.0, 0.5]),
            risk_score=0.0, max_risk_idx=-1,
        )
        far_y, far_x = self._interior_pixel(bbox_far)
        close_y, close_x = self._interior_pixel(bbox_close)
        # Closer bbox: more red, less green/blue than the far one.
        assert out[close_y, close_x][2] > out[far_y, far_x][2]
        assert out[close_y, close_x][0] < out[far_y, far_x][0]

    def test_outside_bboxes_untouched(self):
        # Critical: the red overlay must stay scoped to bbox interiors —
        # outside, the original scene is preserved.
        img = self._scene()
        bbox = (60, 30, 140, 80)
        out = visualize_risk(
            img.copy(), [bbox],
            risk_features=_features([0.0], proximity=[1.0]),
            risk_score=0.0, max_risk_idx=-1,
        )
        # Sample bottom-left, well clear of the bbox and the score badge.
        np.testing.assert_array_equal(out[90, 10], np.array([120, 120, 120]))

    def test_no_bboxes_no_red_overlay(self):
        img = self._scene()
        out = visualize_risk(
            img.copy(), bboxes=[],
            risk_features={
                "gaze": np.zeros(0), "proximity": np.zeros(0),
                "x_offset": np.zeros(0), "approach": np.zeros(0),
            },
            risk_score=0.0, max_risk_idx=-1,
        )
        # Sample bottom-left, away from the score badge.
        np.testing.assert_array_equal(out[90, 10], np.array([120, 120, 120]))

    def test_close_fallback_renders_strongly_red(self):
        # T2.7 close-fallback: bbox depth assigned 0.0 → proximity 1.0 →
        # uniform red overlay. Pin this Xtion-critical path end-to-end.
        img = self._scene()
        bbox = (60, 30, 140, 80)
        out = visualize_risk(
            img.copy(), [bbox],
            risk_features=_features([0.0], proximity=[1.0]),
            risk_score=0.99, max_risk_idx=0,
        )
        py, px = self._interior_pixel(bbox)
        assert out[py, px][2] > 200  # R channel near max
