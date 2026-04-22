"""Unit tests for riskam.reeval."""

import json
from pathlib import Path

import pytest

from riskam.reeval import reevaluate_experiment, reevaluate_tree


def _make_experiment(
    exp_dir: Path,
    raw_predictions: dict,
    prev_results: dict | None = None,
) -> None:
    """Write a synthetic experiment dir with raw_predictions + optional prior results."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "raw_predictions.json").write_text(json.dumps(raw_predictions))
    if prev_results is not None:
        (exp_dir / "results.json").write_text(json.dumps(prev_results))


class TestReevaluateExperiment:
    def test_recomputes_correct_under_over(self, tmp_path):
        # gt=0 and pred=0 → correct; gt=2, pred=0.45 → correct (in bin); gt=3 pred=0.1 → underestimate
        raw = {"a.png": 0.0, "b.png": 0.45, "c.png": 0.1}
        gt = {"a.png": 0, "b.png": 2, "c.png": 3}
        _make_experiment(tmp_path / "exp", raw)
        result = reevaluate_experiment(tmp_path / "exp", gt)
        assert result["total"] == 3
        assert result["correct"] == 2
        assert result["underestimate"] == 1
        assert result["overestimate"] == 0

    def test_classification_report_populated(self, tmp_path):
        raw = {f"f_{i}.png": 0.8 for i in range(5)}
        gt = {f"f_{i}.png": 3 for i in range(5)}
        _make_experiment(tmp_path / "exp", raw)
        result = reevaluate_experiment(tmp_path / "exp", gt)
        c = result["classification"]
        assert c["micro"]["f1"] == pytest.approx(1.0)
        assert "confusion_matrix" in c

    def test_preserves_params_split_avg_time(self, tmp_path):
        raw = {"a.png": 0.5}
        gt = {"a.png": 2}
        prev = {
            "avg_time": 0.042,
            "params": {"w_prox": 0.7, "w_gaze": 0.25},
            "split": "val",
            "split_meta": {"scheme": "stratified_within_run", "seed": 42},
            "provenance": {"git_sha": "deadbeef"},
        }
        _make_experiment(tmp_path / "exp", raw, prev_results=prev)
        result = reevaluate_experiment(tmp_path / "exp", gt)
        assert result["avg_time"] == 0.042
        assert result["params"] == {"w_prox": 0.7, "w_gaze": 0.25}
        assert result["split"] == "val"
        assert result["split_meta"]["seed"] == 42
        assert result["provenance"]["reeval_of"] == "deadbeef"

    def test_ignores_images_without_ground_truth(self, tmp_path):
        raw = {"a.png": 0.5, "b.png": 0.3}
        gt = {"a.png": 2}  # b.png has no gt
        _make_experiment(tmp_path / "exp", raw)
        result = reevaluate_experiment(tmp_path / "exp", gt)
        assert result["total"] == 1

    def test_output_suffix_does_not_overwrite_originals(self, tmp_path):
        raw = {"a.png": 0.5}
        gt = {"a.png": 2}
        prev_results = {"total": 999, "correct": 999}  # sentinel
        _make_experiment(tmp_path / "exp", raw, prev_results=prev_results)
        reevaluate_experiment(tmp_path / "exp", gt, output_suffix="_v2")
        # Original should be untouched.
        original = json.loads((tmp_path / "exp" / "results.json").read_text())
        assert original["total"] == 999
        # New file has the recomputed metrics.
        new = json.loads((tmp_path / "exp" / "results_v2.json").read_text())
        assert new["total"] == 1

    def test_overwrites_by_default(self, tmp_path):
        raw = {"a.png": 0.5}
        gt = {"a.png": 2}
        _make_experiment(tmp_path / "exp", raw, prev_results={"total": 999})
        reevaluate_experiment(tmp_path / "exp", gt)  # default suffix ""
        result = json.loads((tmp_path / "exp" / "results.json").read_text())
        assert result["total"] == 1

    def test_predictions_json_written(self, tmp_path):
        raw = {"a.png": 0.5, "b.png": 0.8}
        gt = {"a.png": 2, "b.png": 3}
        _make_experiment(tmp_path / "exp", raw)
        reevaluate_experiment(tmp_path / "exp", gt)
        preds = json.loads((tmp_path / "exp" / "predictions.json").read_text())
        assert "a.png" in preds["correct"]
        assert "b.png" in preds["correct"]

    def test_missing_raw_predictions_raises(self, tmp_path):
        exp = tmp_path / "empty_exp"
        exp.mkdir()
        with pytest.raises(FileNotFoundError):
            reevaluate_experiment(exp, {})


class TestReevaluateTree:
    def test_walks_run_and_config_layout(self, tmp_path):
        # Layout: dataset_root / run / slug /
        _make_experiment(tmp_path / "RB_01" / "cfg_a", {"a.png": 0.5})
        _make_experiment(tmp_path / "RB_01" / "cfg_b", {"a.png": 0.8})
        _make_experiment(tmp_path / "RB_02" / "cfg_a", {"a.png": 0.2})
        gt_by_run = {
            "RB_01": {"a.png": 2},
            "RB_02": {"a.png": 1},
        }
        n = reevaluate_tree(tmp_path, gt_by_run)
        assert n == 3
        # Confirm each experiment now has a fresh results.json.
        for p in tmp_path.rglob("results.json"):
            r = json.loads(p.read_text())
            assert "classification" in r

    def test_skips_runs_without_ground_truth(self, tmp_path):
        _make_experiment(tmp_path / "RB_99" / "cfg_a", {"a.png": 0.5})
        assert reevaluate_tree(tmp_path, {}) == 0

    def test_missing_dataset_root_returns_zero(self, tmp_path):
        assert reevaluate_tree(tmp_path / "nonexistent", {}) == 0
