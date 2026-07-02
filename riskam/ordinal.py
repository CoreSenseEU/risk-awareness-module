"""
riskam.ordinal

Direction B1 of the metric redesign (``docs/private/paper-plan.md``): a
proportional-odds (ordered logit) model mapping features to the four risk
classes,

    logit P(class ≥ k) = x·β − α_k,

fit by maximum likelihood on the *val* bucket of the canonical split. The
β's play the role of the current hand-swept weights — but arrive with
standard errors, z-values, and p-values, and the fitted model emits class
*probabilities*, unlocking the proper scoring rules in
:mod:`riskam.proba_metrics`.

``statsmodels`` (the optional ``experiments`` dependency group) is imported
only inside :func:`fit_ordinal`; a persisted :class:`OrdinalFit` predicts
with numpy alone, so evaluation and downstream consumers stay lightweight.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class OrdinalFit:
    """A fitted proportional-odds model, self-contained for prediction.

    ``mu``/``sd`` are the fit-set standardisation applied before the
    linear predictor; ``thresholds`` are the transformed cutpoints
    α_1..α_{K−1} on the standardised scale.
    """

    feature_names: list
    mu: list
    sd: list
    coefs: dict           # name → {"coef", "se", "z", "p"}
    thresholds: list
    llf: float
    ll_null: float
    aic: float
    n_obs: int
    converged: bool

    @property
    def n_classes(self) -> int:
        return len(self.thresholds) + 1

    def linear_predictor(self, X: np.ndarray) -> np.ndarray:
        Xz = (np.asarray(X, dtype=float) - np.asarray(self.mu)) / np.asarray(self.sd)
        beta = np.array([self.coefs[n]["coef"] for n in self.feature_names])
        return Xz @ beta

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(n, K) class probabilities, reimplemented from stored params.

        P(y ≤ k) = σ(α_{k+1} − x·β); class probabilities are consecutive
        differences of the cumulative curve.
        """
        xb = self.linear_predictor(X)
        alphas = np.asarray(self.thresholds, dtype=float)
        cum = 1.0 / (1.0 + np.exp(-(alphas[None, :] - xb[:, None])))
        cum = np.concatenate(
            [np.zeros((len(xb), 1)), cum, np.ones((len(xb), 1))], axis=1
        )
        return np.diff(cum, axis=1)

    def to_json(self, path: Path, extra: dict | None = None) -> None:
        payload = asdict(self)
        if extra:
            payload.update(extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "OrdinalFit":
        data = json.loads(Path(path).read_text())
        fields = {f: data[f] for f in cls.__dataclass_fields__}
        return cls(**fields)


def fit_ordinal(X, y, feature_names: list) -> OrdinalFit:
    """Fit a proportional-odds logit on standardised features.

    BFGS first (matching the exploratory probe), L-BFGS retry if it fails
    to converge; the ``converged`` flag is persisted either way.
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if X.ndim != 2 or X.shape[1] != len(feature_names):
        raise ValueError(
            f"X has shape {X.shape}; expected (n, {len(feature_names)}) "
            f"for features {feature_names}"
        )

    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Xz = (X - mu) / sd

    model = OrderedModel(y, Xz, distr="logit")
    res = model.fit(method="bfgs", maxiter=500, disp=False)
    if not res.mle_retvals.get("converged", False):
        res = model.fit(method="lbfgs", maxiter=1000, disp=False)

    k = len(feature_names)
    params = np.asarray(res.params)
    bse = np.asarray(res.bse)
    zvals = np.asarray(res.tvalues)
    pvals = np.asarray(res.pvalues)
    coefs = {
        name: {
            "coef": float(params[i]),
            "se": float(bse[i]),
            "z": float(zvals[i]),
            "p": float(pvals[i]),
        }
        for i, name in enumerate(feature_names)
    }
    # transform_threshold_params returns [-inf, α_1..α_{K−1}, +inf].
    thresholds = model.transform_threshold_params(params)[1:-1].tolist()

    # Intercept-only (thresholds saturate the marginal) → the null
    # log-likelihood is the empirical class-frequency entropy sum.
    counts = np.bincount(y, minlength=int(y.max()) + 1).astype(float)
    counts = counts[counts > 0]
    ll_null = float(np.sum(counts * np.log(counts / len(y))))

    return OrdinalFit(
        feature_names=list(feature_names),
        mu=mu.tolist(),
        sd=sd.tolist(),
        coefs=coefs,
        thresholds=thresholds,
        llf=float(res.llf),
        ll_null=ll_null,
        aic=float(res.aic),
        n_obs=int(res.nobs),
        converged=bool(res.mle_retvals.get("converged", False)),
    )


def lr_test(full: OrdinalFit, reduced: OrdinalFit) -> dict:
    """Likelihood-ratio test of a full model against a nested reduction.

    The reduced model must have been fit on the same rows (same n_obs)
    with a strict subset of the full model's features.
    """
    from scipy.stats import chi2

    if reduced.n_obs != full.n_obs:
        raise ValueError(
            f"LR test needs identical fit rows: full n={full.n_obs}, "
            f"reduced n={reduced.n_obs}"
        )
    extra = set(full.feature_names) - set(reduced.feature_names)
    if not extra or not set(reduced.feature_names) <= set(full.feature_names):
        raise ValueError(
            "reduced model must use a strict subset of the full features"
        )
    df = len(extra)
    stat = 2.0 * (full.llf - reduced.llf)
    return {
        "stat": float(stat),
        "df": df,
        "p": float(chi2.sf(stat, df)),
        "dropped": sorted(extra),
    }
