"""Unit tests for riskam.data.splits."""

import json
from pathlib import Path

import pytest

from riskam.data.splits import (
    Split,
    filter_by_split,
    load_split,
    make_stratified_split,
    save_split,
    split_summary,
)


def _fixture_gt() -> dict[str, dict[str, int]]:
    """Small two-run ground truth with mixed class distributions."""
    return {
        "RB_01": {f"f_{i}.png": (i % 4) for i in range(40)},
        "RB_02": {
            **{f"g_{i}.png": 0 for i in range(10)},
            **{f"g_{10 + i}.png": 3 for i in range(30)},
        },
    }


class TestMakeStratifiedSplit:
    def test_every_frame_assigned_to_val_or_test(self):
        gt = _fixture_gt()
        split = make_stratified_split(gt)
        for run, frames in gt.items():
            for fname in frames:
                assert split.bucket(run, fname) in ("val", "test")

    def test_deterministic_with_seed(self):
        gt = _fixture_gt()
        s1 = make_stratified_split(gt, seed=123)
        s2 = make_stratified_split(gt, seed=123)
        assert s1.assignments == s2.assignments

    def test_different_seeds_produce_different_splits(self):
        gt = _fixture_gt()
        s1 = make_stratified_split(gt, seed=1)
        s2 = make_stratified_split(gt, seed=2)
        assert s1.assignments != s2.assignments

    def test_stratification_preserves_class_ratios(self):
        # 40 frames evenly across 4 classes (10 each) in RB_01.
        gt = _fixture_gt()
        split = make_stratified_split(gt, test_fraction=0.3, seed=42)
        summary = split_summary(gt, split)

        # Each class should have ~3 frames in test per run (10 * 0.3 = 3).
        for label in (0, 1, 2, 3):
            assert summary["test"]["RB_01"]["classes"].get(label, 0) == 3
            assert summary["val"]["RB_01"]["classes"].get(label, 0) == 7

    def test_rejects_fraction_out_of_bounds(self):
        with pytest.raises(ValueError):
            make_stratified_split(_fixture_gt(), test_fraction=0.0)
        with pytest.raises(ValueError):
            make_stratified_split(_fixture_gt(), test_fraction=1.0)

    def test_handles_empty_run(self):
        gt = {"RB_01": {"a.png": 2}, "RB_07": {}}
        split = make_stratified_split(gt)
        assert split.assignments["RB_07"] == {}


class TestFilterBySplit:
    def test_returns_only_requested_bucket(self):
        gt = _fixture_gt()
        split = make_stratified_split(gt, test_fraction=0.25, seed=7)
        val = filter_by_split(gt["RB_01"], "RB_01", split, "val")
        test = filter_by_split(gt["RB_01"], "RB_01", split, "test")
        # Partition: no overlap, union covers the full run.
        assert set(val).isdisjoint(set(test))
        assert set(val) | set(test) == set(gt["RB_01"])


class TestLoadSave:
    def test_round_trip(self, tmp_path):
        gt = _fixture_gt()
        split = make_stratified_split(gt, test_fraction=0.2, seed=99, notes="test run")
        p = tmp_path / "split.json"
        save_split(p, split)
        assert p.is_file()
        reloaded = load_split(p)
        assert reloaded.test_fraction == pytest.approx(0.2)
        assert reloaded.seed == 99
        assert reloaded.notes == "test run"
        assert reloaded.assignments == split.assignments

    def test_unsupported_scheme_rejected(self, tmp_path):
        p = tmp_path / "split.json"
        p.write_text(json.dumps({"scheme": "something_else"}))
        with pytest.raises(ValueError, match="Unsupported split scheme"):
            load_split(p)


class TestSplitSummary:
    def test_counts_match_ground_truth_total(self):
        gt = _fixture_gt()
        split = make_stratified_split(gt, seed=5)
        summary = split_summary(gt, split)
        for run, frames in gt.items():
            got = (
                summary["val"][run]["frames"]
                + summary["test"][run]["frames"]
            )
            assert got == len(frames)
