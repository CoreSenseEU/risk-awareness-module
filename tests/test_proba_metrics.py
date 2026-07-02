"""Closed-form tests for riskam.proba_metrics (no GPU, no dataset)."""

import numpy as np
import pytest

from riskam.proba_metrics import (
    brier_score,
    cumulative_auc,
    log_loss_score,
    probabilistic_report,
    ranked_probability_score,
    reliability_table,
    spearman_rho,
)


def perfect_proba(y):
    p = np.zeros((len(y), 4))
    p[np.arange(len(y)), y] = 1.0
    return p


UNIFORM = np.full((8, 4), 0.25)
Y = np.array([0, 1, 2, 3, 3, 2, 1, 0])


class TestProperScoringRules:
    def test_perfect_forecast_scores_zero(self):
        p = perfect_proba(Y)
        assert brier_score(Y, p) == 0.0
        assert log_loss_score(Y, p) == pytest.approx(0.0, abs=1e-12)
        assert ranked_probability_score(Y, p) == 0.0

    def test_uniform_forecast_closed_forms(self):
        # Brier: Σ(0.25 − 1{k})² = 3·0.0625 + 0.5625 = 0.75.
        assert brier_score(Y, UNIFORM) == pytest.approx(0.75)
        # Log-loss: −log(0.25).
        assert log_loss_score(Y, UNIFORM) == pytest.approx(np.log(4))

    def test_rps_prefers_adjacent_class_miss(self):
        y = np.array([0])
        near = np.array([[0.0, 1.0, 0.0, 0.0]])  # off by one
        far = np.array([[0.0, 0.0, 0.0, 1.0]])   # off by three
        assert ranked_probability_score(y, near) < ranked_probability_score(y, far)
        # Brier can't tell them apart — that's why RPS is here.
        assert brier_score(y, near) == brier_score(y, far)

    def test_maximally_wrong_point_mass_brier(self):
        y = np.array([0])
        p = np.array([[0.0, 0.0, 0.0, 1.0]])
        assert brier_score(y, p) == pytest.approx(2.0)


class TestCumulativeAuc:
    def test_perfect_ordering_is_auc_one(self):
        y = np.array([0, 1, 2, 3])
        p = perfect_proba(y)
        aucs = cumulative_auc(y, p)
        assert aucs["auc_ge1"] == 1.0
        assert aucs["auc_ge2"] == 1.0
        assert aucs["auc_ge3"] == 1.0

    def test_single_class_threshold_is_none(self):
        y = np.array([3, 3, 3])
        aucs = cumulative_auc(y, perfect_proba(y))
        assert aucs["auc_ge1"] is None


class TestReliability:
    def test_bins_partition_all_samples(self):
        rel = reliability_table(Y, UNIFORM, n_bins=10)
        assert sum(rel["count"]) == len(Y)

    def test_perfect_forecast_has_zero_ece(self):
        rel = reliability_table(Y, perfect_proba(Y))
        assert rel["ece"] == pytest.approx(0.0)

    def test_confident_but_wrong_has_high_ece(self):
        y = np.array([0, 0, 0, 0])
        p = np.zeros((4, 4))
        p[:, 3] = 1.0  # fully confident, always wrong
        rel = reliability_table(y, p)
        assert rel["ece"] == pytest.approx(1.0)

    def test_full_confidence_lands_in_last_bin(self):
        rel = reliability_table(np.array([0]), perfect_proba(np.array([0])))
        assert rel["count"][-1] == 1


class TestSpearman:
    def test_monotone_scores(self):
        y = [0, 1, 2, 3]
        assert spearman_rho([0.1, 0.2, 0.3, 0.4], y) == pytest.approx(1.0)
        assert spearman_rho([0.4, 0.3, 0.2, 0.1], y) == pytest.approx(-1.0)


class TestReport:
    def test_bundle_keys_and_default_score(self):
        rep = probabilistic_report(Y, perfect_proba(Y))
        assert set(rep) == {
            "brier", "log_loss", "rps", "cumulative_auc",
            "reliability", "spearman_rho",
        }
        # Expected-class score of a perfect forecast is the class itself.
        assert rep["spearman_rho"] == pytest.approx(1.0)
