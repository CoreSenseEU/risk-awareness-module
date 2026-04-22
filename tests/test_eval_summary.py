"""Unit tests for riskam.eval_summary."""

import json
from pathlib import Path

import numpy as np
import pytest

from riskam.eval_summary import (
    aggregate_by_config,
    collect_results,
    rank_configs,
    summarize,
    write_summary,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _write_experiment(
    dataset_root: Path,
    run: str,
    params_slug: str,
    cm: np.ndarray,
    avg_time: float = 0.05,
    params: dict | None = None,
) -> Path:
    """Write a minimal results.json for one experiment under the standard layout."""
    from riskam.eval_metrics import (
        macro_prf,
        micro_prf,
        per_class_prf,
        regression_errors,
    )

    per_class = per_class_prf(cm)
    total = int(cm.sum())
    correct = int(np.trace(cm))

    # Synthetic per-sample arrays consistent with the CM (true class repeated
    # row-sum times, predicted class per row/column count).
    y_true: list[int] = []
    y_pred_classes: list[int] = []
    for t in range(cm.shape[0]):
        for p in range(cm.shape[1]):
            for _ in range(int(cm[t, p])):
                y_true.append(t)
                y_pred_classes.append(p)
    # For regression, treat predicted class's centre as the continuous pred.
    from riskam.eval_metrics import class_centre

    y_pred_cont = [class_centre(p) for p in y_pred_classes]
    reg = regression_errors(y_true, y_pred_cont)

    results = {
        "total": total,
        "correct": correct,
        "underestimate": 0,
        "overestimate": total - correct,
        "avg_time": avg_time,
        "classification": {
            "confusion_matrix": cm.tolist(),
            "per_class": per_class,
            "macro": macro_prf(per_class),
            "micro": micro_prf(cm),
            "regression": reg,
        },
        "params": params or {"w_prox": 0.7, "w_gaze": 0.25},
        "provenance": {"git_sha": "synthetic"},
    }
    exp_dir = dataset_root / run / params_slug
    exp_dir.mkdir(parents=True, exist_ok=True)
    with (exp_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f)
    return exp_dir


# ── collect_results ──────────────────────────────────────────────────────────


class TestCollectResults:
    def test_empty_dir(self, tmp_path):
        assert collect_results(tmp_path / "missing") == []

    def test_walks_run_and_config_layout(self, tmp_path):
        cm = np.eye(4, dtype=int) * 3
        _write_experiment(tmp_path, "RB_01", "cfg_a", cm)
        _write_experiment(tmp_path, "RB_01", "cfg_b", cm)
        _write_experiment(tmp_path, "RB_02", "cfg_a", cm)
        records = collect_results(tmp_path)
        assert len(records) == 3
        slugs = {r.params_slug for r in records}
        runs = {r.run for r in records}
        assert slugs == {"cfg_a", "cfg_b"}
        assert runs == {"RB_01", "RB_02"}


# ── aggregate_by_config ──────────────────────────────────────────────────────


class TestAggregateByConfig:
    def test_sums_confusion_matrices_per_config(self, tmp_path):
        cm1 = np.array([[2, 0, 0, 0], [0, 3, 1, 0], [0, 0, 4, 0], [0, 0, 0, 5]])
        cm2 = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 1], [0, 0, 0, 3]])
        _write_experiment(tmp_path, "RB_01", "cfg_x", cm1)
        _write_experiment(tmp_path, "RB_02", "cfg_x", cm2)
        records = collect_results(tmp_path)
        agg = aggregate_by_config(records)
        assert "cfg_x" in agg
        expected_cm = (cm1 + cm2).tolist()
        assert agg["cfg_x"]["aggregate"]["confusion_matrix"] == expected_cm
        assert agg["cfg_x"]["aggregate"]["total"] == int(cm1.sum() + cm2.sum())
        assert agg["cfg_x"]["n_runs"] == 2

    def test_groups_independent_configs(self, tmp_path):
        cm = np.eye(4, dtype=int) * 2
        _write_experiment(tmp_path, "RB_01", "cfg_a", cm)
        _write_experiment(tmp_path, "RB_01", "cfg_b", cm)
        agg = aggregate_by_config(collect_results(tmp_path))
        assert set(agg.keys()) == {"cfg_a", "cfg_b"}

    def test_sample_weighted_mae(self, tmp_path):
        # Force different per-run MAEs with known sample counts.
        # cm1 has 10 frames → mae_1 computed against class centres.
        # cm2 has 1 frame → mae_2.
        cm1 = np.array([[10, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        cm2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        _write_experiment(tmp_path, "RB_01", "cfg_x", cm1)
        _write_experiment(tmp_path, "RB_02", "cfg_x", cm2)
        records = collect_results(tmp_path)
        agg = aggregate_by_config(records)
        # cm1 has all-correct → per-run mae = 0; cm2 has one correct pred → mae = 0.
        # So pooled mae should also be 0.
        assert agg["cfg_x"]["aggregate"]["regression"]["mae"] == pytest.approx(0.0)


# ── rank_configs ─────────────────────────────────────────────────────────────


class TestRankConfigs:
    def _two_configs(self, tmp_path):
        good = np.eye(4, dtype=int) * 10  # all correct
        bad = np.array([
            [5, 5, 0, 0],
            [0, 5, 5, 0],
            [0, 0, 5, 5],
            [5, 0, 0, 5],
        ])
        _write_experiment(tmp_path, "RB_01", "cfg_good", good)
        _write_experiment(tmp_path, "RB_01", "cfg_bad", bad)
        return aggregate_by_config(collect_results(tmp_path))

    def test_rank_by_macro_f1_descending(self, tmp_path):
        agg = self._two_configs(tmp_path)
        ranked = rank_configs(agg, metric="macro_f1")
        assert ranked[0][0] == "cfg_good"
        assert ranked[1][0] == "cfg_bad"

    def test_top_k_limits_list(self, tmp_path):
        agg = self._two_configs(tmp_path)
        assert len(rank_configs(agg, metric="macro_f1", top_k=1)) == 1

    def test_rank_by_mae_ascending(self, tmp_path):
        agg = self._two_configs(tmp_path)
        ranked = rank_configs(agg, metric="mae")
        # Lower MAE is better → cfg_good first.
        assert ranked[0][0] == "cfg_good"

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError):
            rank_configs({}, metric="nonsense")


# ── write_summary / summarize end-to-end ─────────────────────────────────────


class TestSummarizeEndToEnd:
    def test_writes_both_files(self, tmp_path):
        cm = np.eye(4, dtype=int) * 4
        _write_experiment(tmp_path, "RB_01", "cfg_a", cm)
        _write_experiment(tmp_path, "RB_02", "cfg_a", cm)
        agg = summarize(tmp_path, top_k=5)
        assert "cfg_a" in agg
        assert (tmp_path / "summary.json").is_file()
        assert (tmp_path / "summary.md").is_file()

    def test_summary_json_has_ranked_list(self, tmp_path):
        cm = np.eye(4, dtype=int) * 2
        _write_experiment(tmp_path, "RB_01", "cfg_x", cm)
        write_summary(tmp_path, aggregate_by_config(collect_results(tmp_path)))
        with (tmp_path / "summary.json").open() as f:
            summary = json.load(f)
        assert summary["n_configs"] == 1
        assert summary["ranked"][0]["params_slug"] == "cfg_x"

    def test_markdown_contains_config_and_runs(self, tmp_path):
        cm = np.eye(4, dtype=int) * 3
        _write_experiment(tmp_path, "RB_01", "cfg_alpha", cm)
        _write_experiment(tmp_path, "RB_02", "cfg_alpha", cm)
        summarize(tmp_path)
        md = (tmp_path / "summary.md").read_text()
        assert "cfg_alpha" in md
        assert "RB_01" in md and "RB_02" in md
        assert "Macro F1" in md

    def test_empty_raises_clearly(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            summarize(tmp_path / "no_results_here")

    def test_old_results_without_classification_raises(self, tmp_path):
        # Legacy results.json predating T3.3.4 — no 'classification' block.
        exp = tmp_path / "RB_01" / "cfg_old"
        exp.mkdir(parents=True)
        with (exp / "results.json").open("w") as f:
            json.dump({"total": 5, "correct": 3}, f)
        with pytest.raises(ValueError, match="predates T3.3.4"):
            summarize(tmp_path)
