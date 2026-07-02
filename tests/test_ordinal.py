"""Recovery tests for riskam.ordinal on simulated proportional-odds data.

statsmodels is only needed at fit time; these tests are skipped cleanly
when the ``experiments`` dependency group is not installed.
"""

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from riskam.ordinal import OrdinalFit, fit_ordinal, lr_test


def simulate(n=4000, seed=0):
    """Proportional-odds data: one strong feature, one pure-noise feature."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    noise = rng.normal(size=n)
    beta_true = 2.0
    alphas = np.array([-1.5, 0.0, 1.5])
    xb = beta_true * signal
    u = rng.logistic(size=n)
    y = (xb + u > alphas[:, None]).sum(axis=0)  # 0..3
    X = np.column_stack([signal, noise])
    return X, y


class TestFitRecovery:
    def setup_method(self):
        self.X, self.y = simulate()
        self.fit = fit_ordinal(self.X, self.y, ["signal", "noise"])

    def test_converged(self):
        assert self.fit.converged

    def test_recovers_true_coefficient(self):
        # signal has sd≈1 so the standardised coef ≈ β_true = 2.
        assert self.fit.coefs["signal"]["coef"] == pytest.approx(2.0, abs=0.2)
        assert self.fit.coefs["signal"]["p"] < 0.01

    def test_noise_feature_is_insignificant(self):
        assert self.fit.coefs["noise"]["p"] > 0.05
        assert abs(self.fit.coefs["noise"]["coef"]) < 0.1

    def test_thresholds_recovered_in_order(self):
        assert self.fit.thresholds == sorted(self.fit.thresholds)
        assert len(self.fit.thresholds) == 3
        assert self.fit.thresholds[1] == pytest.approx(0.0, abs=0.15)

    def test_better_than_null(self):
        assert self.fit.llf > self.fit.ll_null


class TestPredictProba:
    def setup_method(self):
        self.X, self.y = simulate()
        self.fit = fit_ordinal(self.X, self.y, ["signal", "noise"])

    def test_rows_sum_to_one(self):
        proba = self.fit.predict_proba(self.X[:100])
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
        assert (proba >= 0).all()

    def test_matches_statsmodels_predict(self):
        # The numpy reimplementation must agree with the fitted model.
        from statsmodels.miscmodels.ordinal_model import OrderedModel

        mu, sd = self.X.mean(0), self.X.std(0)
        Xz = (self.X - mu) / sd
        res = OrderedModel(self.y, Xz, distr="logit").fit(
            method="bfgs", maxiter=500, disp=False
        )
        np.testing.assert_allclose(
            self.fit.predict_proba(self.X), res.predict(Xz), atol=1e-6
        )

    def test_json_roundtrip(self, tmp_path):
        path = tmp_path / "fit.json"
        self.fit.to_json(path, extra={"note": "test"})
        loaded = OrdinalFit.from_json(path)
        np.testing.assert_allclose(
            loaded.predict_proba(self.X[:50]),
            self.fit.predict_proba(self.X[:50]),
            atol=1e-12,
        )
        assert loaded.feature_names == ["signal", "noise"]


class TestLrTest:
    def test_dropping_noise_is_insignificant(self):
        X, y = simulate()
        full = fit_ordinal(X, y, ["signal", "noise"])
        reduced = fit_ordinal(X[:, :1], y, ["signal"])
        out = lr_test(full, reduced)
        assert out["df"] == 1
        assert out["dropped"] == ["noise"]
        assert out["p"] > 0.05

    def test_dropping_signal_is_significant(self):
        X, y = simulate()
        full = fit_ordinal(X, y, ["signal", "noise"])
        reduced = fit_ordinal(X[:, 1:], y, ["noise"])
        out = lr_test(full, reduced)
        assert out["p"] < 1e-10

    def test_rejects_non_nested_models(self):
        X, y = simulate()
        full = fit_ordinal(X, y, ["signal", "noise"])
        other = fit_ordinal(X[:100], y[:100], ["signal", "noise"])
        with pytest.raises(ValueError):
            lr_test(full, other)  # different rows
        with pytest.raises(ValueError):
            lr_test(full, full)  # not a strict subset
