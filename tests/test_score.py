"""Unit tests for riskam.score.RiskScorer."""

import numpy as np
import pytest

from riskam.score import (
    CROWD_ALPHA_DEFAULT,
    VERY_SMALL_RISK_VALUE,
    RiskScorer,
)


def _features(proximity=0.8, gaze=0.0, x_offset=1.0):
    """Helper: single-person feature dict."""
    return {
        "proximity": np.array([proximity]),
        "gaze": np.array([gaze]),
        "x_offset": np.array([x_offset]),
        "approach": np.array([0.5]),
    }


class TestRiskScorerNoHumans:
    def test_returns_zero_on_none(self):
        scorer = RiskScorer()
        risk, idx, per = scorer.score(None)
        assert risk == 0.0
        assert idx == -1
        assert per == {}


class TestRiskScorerSinglePerson:
    def test_maximum_risk(self):
        scorer = RiskScorer(n_frames=1)
        # Worst case: very close (proximity=1), unaware (gaze=0), in-path (x_offset=1)
        risk, idx, _ = scorer.score(_features(proximity=1.0, gaze=0.0, x_offset=1.0))
        assert risk == pytest.approx(1.0, abs=0.01)
        assert idx == 0

    def test_minimum_risk(self):
        scorer = RiskScorer(n_frames=1)
        # Best case: far (proximity=0), fully aware (gaze=1), off-path (x_offset=0)
        risk, idx, _ = scorer.score(_features(proximity=0.0, gaze=1.0, x_offset=0.0))
        assert risk >= VERY_SMALL_RISK_VALUE
        assert risk < 0.1

    def test_risk_clamped_to_one(self):
        scorer = RiskScorer(n_frames=1)
        risk, _, _ = scorer.score(_features(proximity=2.0, gaze=0.0, x_offset=2.0))
        assert risk <= 1.0

    def test_temporal_smoothing(self):
        scorer = RiskScorer(n_frames=3)
        risk_high, _, _ = scorer.score(_features(proximity=1.0, gaze=0.0, x_offset=1.0))
        risk_low, _, _ = scorer.score(_features(proximity=0.0, gaze=1.0, x_offset=0.0))
        # With n_frames=3, second call is still smoothed with first high-risk frame.
        # Risk should be lower than the first call but not zero.
        assert risk_low < risk_high

    def test_per_person_dict_returned(self):
        scorer = RiskScorer(n_frames=1)
        _, _, per = scorer.score(_features(), track_ids=[42])
        assert 42 in per
        assert 0.0 <= per[42] <= 1.0


class TestRiskScorerMultiPerson:
    def test_crowd_penalty_increases_risk(self):
        scorer_single = RiskScorer(n_frames=1, crowd_alpha=CROWD_ALPHA_DEFAULT)
        scorer_crowd = RiskScorer(n_frames=1, crowd_alpha=CROWD_ALPHA_DEFAULT)

        feats_single = _features(proximity=0.5, gaze=0.0, x_offset=1.0)
        risk_single, _, _ = scorer_single.score(feats_single)

        feats_crowd = {
            "proximity": np.array([0.5, 0.5]),
            "gaze": np.array([0.0, 0.0]),
            "x_offset": np.array([1.0, 1.0]),
            "approach": np.array([0.5, 0.5]),
        }
        risk_crowd, _, _ = scorer_crowd.score(feats_crowd)

        assert risk_crowd > risk_single

    def test_max_risk_idx_correct(self):
        scorer = RiskScorer(n_frames=1)
        feats = {
            "proximity": np.array([0.1, 0.9]),
            "gaze": np.array([0.9, 0.0]),
            "x_offset": np.array([0.0, 1.0]),
            "approach": np.array([0.5, 0.5]),
        }
        _, idx, _ = scorer.score(feats)
        assert idx == 1  # second person is riskier

    def test_crowd_alpha_zero_no_penalty(self):
        scorer_no_penalty = RiskScorer(n_frames=1, crowd_alpha=0.0)
        scorer_penalty = RiskScorer(n_frames=1, crowd_alpha=0.5)

        feats = {
            "proximity": np.array([0.5, 0.5, 0.5]),
            "gaze": np.array([0.0, 0.0, 0.0]),
            "x_offset": np.array([1.0, 1.0, 1.0]),
            "approach": np.array([0.5, 0.5, 0.5]),
        }
        risk_no, _, _ = scorer_no_penalty.score(feats)
        risk_yes, _, _ = scorer_penalty.score(feats)
        assert risk_yes > risk_no


class TestRiskScorerReset:
    def test_reset_clears_history(self):
        scorer = RiskScorer(n_frames=5)
        scorer.score(_features(proximity=1.0, gaze=0.0, x_offset=1.0))
        scorer.reset()
        # After reset, a low-risk frame should not be smoothed with the old high-risk one.
        risk, _, _ = scorer.score(_features(proximity=0.0, gaze=1.0, x_offset=0.0))
        assert risk < 0.1
