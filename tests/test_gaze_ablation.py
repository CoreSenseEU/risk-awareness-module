"""Tests for the T3.3.11 ablation-related pieces: eye-symmetry gaze,
algorithm selector, sweep-config `gaze_algorithm` axis, featextr threading.
"""

import math

import numpy as np
import pytest

from riskam.ml.humandet import (
    GAZE_ALGORITHM_DEFAULT,
    GAZE_ALGORITHMS,
    SIGMA_YAW_DEFAULT,
    _eye_symmetry_gaze,
    gaze_scores,
)


def _kpts(
    nose_x=100.0, nose_y=120.0,
    leye_x=115.0, leye_y=100.0,
    reye_x=85.0,  reye_y=100.0,
) -> np.ndarray:
    kpts = np.zeros((17, 2), dtype=float)
    kpts[0] = [nose_x, nose_y]
    kpts[1] = [leye_x, leye_y]
    kpts[2] = [reye_x, reye_y]
    return kpts


# ── _eye_symmetry_gaze ────────────────────────────────────────────────────────


class TestEyeSymmetryGaze:
    def test_frontal_face_high_score(self):
        # Symmetric around nose → yaw offset near zero → score near 1.
        kpts = _kpts()
        score = _eye_symmetry_gaze(kpts, sigma_yaw=SIGMA_YAW_DEFAULT)
        assert score > 0.95

    def test_strong_yaw_low_score(self):
        # Nose far off from eye midpoint → low score.
        kpts = _kpts(nose_x=150.0)  # eye midpoint is at 100; yaw offset = 50/30 ≈ 1.67
        score = _eye_symmetry_gaze(kpts, sigma_yaw=SIGMA_YAW_DEFAULT)
        assert score < 0.1

    def test_F2_weakness_retained(self):
        """
        The defining weakness of the eye-symmetry baseline: it scores the same
        whether the head is pitched neutrally OR looking straight up/down
        (because pitch is ignored). This test pins that behaviour — losing it
        would mean the baseline stopped being a faithful reconstruction.
        """
        # Build two faces identical in yaw (symmetric, nose at x=100) but with
        # drastically different pitch (nose well above vs well below eyes).
        neutral = _kpts(nose_x=100.0, nose_y=130.0)   # nose below eyes (normal)
        pitched_up = _kpts(nose_x=100.0, nose_y=60.0)  # nose well above eyes
        s_neutral = _eye_symmetry_gaze(neutral, sigma_yaw=SIGMA_YAW_DEFAULT)
        s_pitched = _eye_symmetry_gaze(pitched_up, sigma_yaw=SIGMA_YAW_DEFAULT)
        # Both should be near 1 — eye-symmetry can't tell them apart.
        assert s_neutral > 0.95
        assert s_pitched > 0.95

    def test_degenerate_keypoints_return_zero(self):
        kpts = np.zeros((17, 2), dtype=float)
        assert _eye_symmetry_gaze(kpts, sigma_yaw=SIGMA_YAW_DEFAULT) == 0.0


# ── gaze_scores algorithm selector ────────────────────────────────────────────


class TestGazeScoresAlgorithmSelector:
    def _batch(self, n: int = 3) -> np.ndarray:
        return np.stack([_kpts() for _ in range(n)])

    def test_default_is_head_pose(self):
        assert GAZE_ALGORITHM_DEFAULT == "head_pose"

    def test_head_pose_and_eye_symmetry_both_return_batch(self):
        batch = self._batch(n=4)
        hp = gaze_scores(batch, algorithm="head_pose")
        es = gaze_scores(batch, algorithm="eye_symmetry")
        assert len(hp) == 4
        assert len(es) == 4

    def test_head_pose_is_stricter_on_pitched_faces(self):
        """head_pose includes a pitch penalty; eye_symmetry does not. A face
        with strong pitch should score LOWER under head_pose than under
        eye_symmetry."""
        pitched = np.stack([_kpts(nose_x=100.0, nose_y=60.0)])
        hp = gaze_scores(pitched, algorithm="head_pose")
        es = gaze_scores(pitched, algorithm="eye_symmetry")
        assert es[0] > hp[0]

    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unknown gaze algorithm"):
            gaze_scores(self._batch(), algorithm="magic")

    def test_known_algorithms_constant_matches(self):
        assert set(GAZE_ALGORITHMS) == {"head_pose", "eye_symmetry"}


# ── SweepConfig gaze_algorithm axis ──────────────────────────────────────────


class TestSweepConfigGazeAlgorithm:
    def test_default_axis_is_head_pose_only(self):
        from riskam.sweep_config import load_sweep_config

        cfg = load_sweep_config()
        assert cfg.gaze_algorithm == ("head_pose",)

    def test_ablation_config_has_both_algorithms(self):
        from pathlib import Path
        from riskam.sweep_config import load_sweep_config

        cfg = load_sweep_config(
            Path(__file__).resolve().parent.parent / "configs" / "sweeps" / "ablation_gaze.yaml"
        )
        assert set(cfg.gaze_algorithm) == {"head_pose", "eye_symmetry"}
        # Sweep size doubles vs default: 6 × 3 × 2 × 2 = 72.
        assert len(cfg) == 72

    def test_cartesian_product_includes_algorithm(self):
        from pathlib import Path
        from riskam.sweep_config import load_sweep_config

        cfg = load_sweep_config(
            Path(__file__).resolve().parent.parent / "configs" / "sweeps" / "ablation_gaze.yaml"
        )
        algos_seen = {p["gaze_algorithm"] for p in cfg.iter_experiments()}
        assert algos_seen == {"head_pose", "eye_symmetry"}

    def test_string_algorithm_accepted_as_single_value(self, tmp_path):
        from riskam.sweep_config import load_sweep_config

        p = tmp_path / "sweep.yaml"
        p.write_text(
            """
weight_tuples:
  - {w_proximity: 1.0, w_gaze: 0.0, w_position: 0.0, w_approach: 0.0}
gaze_sigma_yaw: [0.3]
gaze_sigma_pitch: [0.5]
gaze_algorithm: eye_symmetry
"""
        )
        cfg = load_sweep_config(p)
        assert cfg.gaze_algorithm == ("eye_symmetry",)

    def test_unknown_algorithm_in_config_rejected(self, tmp_path):
        from riskam.sweep_config import load_sweep_config

        p = tmp_path / "sweep.yaml"
        p.write_text(
            """
weight_tuples:
  - {w_proximity: 1.0, w_gaze: 0.0, w_position: 0.0, w_approach: 0.0}
gaze_sigma_yaw: [0.3]
gaze_sigma_pitch: [0.5]
gaze_algorithm: [magic_new_thing]
"""
        )
        with pytest.raises(ValueError, match="Unknown gaze algorithm"):
            load_sweep_config(p)


# ── featextr threads the selector ────────────────────────────────────────────


class TestFeatextrThreadsAlgorithm:
    def test_extract_propagates_gaze_algorithm(self, monkeypatch):
        """Check that featextr.extract hands the algorithm to compute_gaze
        rather than always using the default."""
        from riskam.ml import featextr, subscores
        from riskam.ml.subscores import FrameInputs

        captured: dict = {}

        def fake_compute_gaze(keypoints_np, **kwargs):
            captured["algorithm"] = kwargs.get("algorithm")
            import numpy as np

            # Return a SubScoreResult with one value per detection.
            n = 0 if keypoints_np is None else len(keypoints_np)
            return subscores.SubScoreResult(
                values=np.zeros(n), status=subscores.SubScoreStatus.ACTIVE
            )

        monkeypatch.setattr(featextr, "compute_gaze", fake_compute_gaze)

        # Need a nonzero-person detection path for compute_gaze to be called.
        # Patch humandet.detect_humans to return one bbox + dummy keypoints.
        def fake_detect(image, track_bboxes):
            return (
                [[10, 10, 40, 40]],
                np.zeros((1, 17, 2), dtype=float),
                [None],
            )

        monkeypatch.setattr(featextr.humandet, "detect_humans", fake_detect)

        import numpy as np

        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.full((100, 100), 1.0, dtype=np.float32)
        featextr.extract(
            FrameInputs(rgb=rgb, depth_m=depth, cmd_vel=None),
            gaze_algorithm="eye_symmetry",
        )
        assert captured["algorithm"] == "eye_symmetry"
