"""Unit tests for riskam.sweep_config."""

from pathlib import Path

import pytest

from riskam.sweep_config import (
    DEFAULT_SWEEP_CONFIG_PATH,
    SweepConfig,
    WeightTuple,
    load_sweep_config,
)


# ── WeightTuple validation ───────────────────────────────────────────────────


class TestWeightTuple:
    def test_normalised_tuple_passes(self):
        WeightTuple(0.7, 0.2, 0.05, 0.05).validate()

    def test_tolerance_allows_small_drift(self):
        WeightTuple(0.70, 0.20, 0.05, 0.055).validate()  # sum 1.005 < 1.01 tol

    def test_off_sum_raises(self):
        with pytest.raises(ValueError, match="sums to"):
            WeightTuple(0.5, 0.5, 0.5, 0.0).validate()

    def test_zero_sum_raises(self):
        with pytest.raises(ValueError):
            WeightTuple(0.0, 0.0, 0.0, 0.0).validate()


# ── SweepConfig.iter_experiments ─────────────────────────────────────────────


class TestSweepConfigIter:
    def _cfg(self, tuples=1, yaws=2, pitches=3) -> SweepConfig:
        return SweepConfig(
            name="t",
            description="",
            weight_tuples=tuple(
                WeightTuple(0.6, 0.2, 0.1, 0.1) for _ in range(tuples)
            ),
            gaze_sigma_yaw=tuple(0.1 * (i + 1) for i in range(yaws)),
            gaze_sigma_pitch=tuple(0.1 * (i + 1) for i in range(pitches)),
        )

    def test_length_is_cartesian_product(self):
        cfg = self._cfg(tuples=3, yaws=4, pitches=2)
        assert len(cfg) == 24
        assert len(list(cfg.iter_experiments())) == 24

    def test_yielded_dict_has_expected_keys(self):
        cfg = self._cfg()
        params = next(iter(cfg.iter_experiments()))
        assert set(params.keys()) == {
            "w_prox",
            "w_gaze",
            "w_pos",
            "w_approach",
            "gaze_sigma_yaw",
            "gaze_sigma_pitch",
        }


# ── YAML round-trip / default ────────────────────────────────────────────────


class TestLoadSweepConfig:
    def test_default_config_loads(self):
        cfg = load_sweep_config()
        assert len(cfg) == 36  # 6 tuples × 3 yaw × 2 pitch
        assert cfg.name == "default"

    def test_explicit_path(self, tmp_path):
        p = tmp_path / "sweep.yaml"
        p.write_text(
            """
name: mini
weight_tuples:
  - {w_proximity: 1.0, w_gaze: 0.0, w_position: 0.0, w_approach: 0.0}
gaze_sigma_yaw: [0.3]
gaze_sigma_pitch: [0.5]
"""
        )
        cfg = load_sweep_config(p)
        assert cfg.name == "mini"
        assert len(cfg) == 1

    def test_missing_key_raises(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(
            """
weight_tuples:
  - {w_proximity: 1.0, w_gaze: 0.0, w_position: 0.0, w_approach: 0.0}
gaze_sigma_yaw: [0.3]
"""
        )  # missing gaze_sigma_pitch
        with pytest.raises(ValueError, match="gaze_sigma_pitch"):
            load_sweep_config(p)

    def test_weight_sum_validated(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(
            """
weight_tuples:
  - {w_proximity: 0.5, w_gaze: 0.5, w_position: 0.5, w_approach: 0.0}
gaze_sigma_yaw: [0.3]
gaze_sigma_pitch: [0.5]
"""
        )
        with pytest.raises(ValueError, match="sums to"):
            load_sweep_config(p)

    def test_unknown_weight_key_rejected(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(
            """
weight_tuples:
  - {w_proximity: 1.0, w_gaze: 0.0, w_position: 0.0, w_approach: 0.0, w_bonus: 0.5}
gaze_sigma_yaw: [0.3]
gaze_sigma_pitch: [0.5]
"""
        )
        with pytest.raises(ValueError, match="unknown keys"):
            load_sweep_config(p)

    def test_empty_weight_tuples_rejected(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(
            """
weight_tuples: []
gaze_sigma_yaw: [0.3]
gaze_sigma_pitch: [0.5]
"""
        )
        with pytest.raises(ValueError, match="non-empty"):
            load_sweep_config(p)


class TestDefaultConfigFile:
    def test_file_exists(self):
        assert DEFAULT_SWEEP_CONFIG_PATH.is_file()

    def test_all_tuples_normalised(self):
        cfg = load_sweep_config()
        for wt in cfg.weight_tuples:
            wt.validate()  # raises on failure

    def test_sweep_has_approach_off_and_on(self):
        cfg = load_sweep_config()
        approach_values = {wt.w_approach for wt in cfg.weight_tuples}
        assert approach_values == {0.0, 0.15}
