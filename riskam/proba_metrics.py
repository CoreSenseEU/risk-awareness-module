"""
riskam.proba_metrics

Proper scoring rules and calibration diagnostics for probabilistic risk
predictions — the evaluation upgrade Direction B of the metric redesign
unlocks (``docs/private/paper-plan.md`` §2B).

Kept separate from :mod:`riskam.eval_metrics` on purpose: that module
defines the deployed pipeline's ``results.json`` schema and stays frozen;
this one serves the experimental ``metric_lab`` path.

All functions take ``y_true`` as integer classes in ``[0, N_CLASSES)`` and
``proba`` as an ``(n, N_CLASSES)`` array of predicted class probabilities.
"""

from __future__ import annotations

import numpy as np

N_CLASSES = 4


def _one_hot(y_true: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y_true), n_classes), dtype=float)
    out[np.arange(len(y_true)), y_true] = 1.0
    return out


def brier_score(y_true, proba) -> float:
    """Multiclass Brier score: mean over samples of Σ_k (p_k − 1{y=k})².

    0 = perfect; 2 = maximally wrong point mass.
    """
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    return float(np.mean(np.sum((proba - _one_hot(y_true, proba.shape[1])) ** 2, axis=1)))


def log_loss_score(y_true, proba, eps: float = 1e-15) -> float:
    """Mean negative log-likelihood of the true class."""
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    p_true = np.clip(proba[np.arange(len(y_true)), y_true], eps, 1.0)
    return float(-np.mean(np.log(p_true)))


def ranked_probability_score(y_true, proba) -> float:
    """RPS — the ordinal-proper scoring rule.

    Mean over samples of Σ_k (F̂_k − F_k)² / (K − 1), where F is the
    cumulative distribution over the *ordered* classes. Unlike Brier, an
    adjacent-class miss scores better than a far-class miss.
    """
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    k = proba.shape[1]
    cum_pred = np.cumsum(proba, axis=1)
    cum_true = np.cumsum(_one_hot(y_true, k), axis=1)
    return float(np.mean(np.sum((cum_pred - cum_true) ** 2, axis=1) / (k - 1)))


def cumulative_auc(y_true, proba) -> dict:
    """AUC of P(y ≥ k) as a score for the event {y ≥ k}, k = 1..K−1.

    The natural ROC decomposition for an ordinal target ("how well does
    the model rank at-least-medium-risk frames?"). Thresholds with a
    single-class truth return None for that k.
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    out: dict[str, float | None] = {}
    for k in range(1, proba.shape[1]):
        event = (y_true >= k).astype(int)
        score = proba[:, k:].sum(axis=1)
        if event.min() == event.max():
            out[f"auc_ge{k}"] = None
        else:
            out[f"auc_ge{k}"] = float(roc_auc_score(event, score))
    return out


def reliability_table(y_true, proba, n_bins: int = 10) -> dict:
    """Confidence-binned reliability data (max-probability convention).

    Bins predictions by their top-class probability, then compares mean
    confidence with empirical accuracy per bin. Returns the plot-ready
    table plus the expected calibration error (ECE, count-weighted mean
    |confidence − accuracy|).
    """
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-inclusive last bin so conf = 1.0 lands in bin n_bins − 1.
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)

    mean_conf, accuracy, count = [], [], []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        count.append(n)
        mean_conf.append(float(conf[mask].mean()) if n else None)
        accuracy.append(float(correct[mask].mean()) if n else None)

    total = len(y_true)
    ece = sum(
        (n / total) * abs(c - a)
        for n, c, a in zip(count, mean_conf, accuracy)
        if n > 0
    )
    return {
        "bin_edges": edges.tolist(),
        "mean_confidence": mean_conf,
        "accuracy": accuracy,
        "count": count,
        "ece": float(ece),
    }


def spearman_rho(scores, y_true) -> float:
    """Spearman rank correlation of a continuous score with the gt class."""
    from scipy.stats import spearmanr

    rho, _ = spearmanr(np.asarray(scores, dtype=float), np.asarray(y_true, dtype=int))
    return float(rho)


def probabilistic_report(y_true, proba, continuous_score=None) -> dict:
    """One-shot bundle of the proper scoring rules + calibration data.

    ``continuous_score`` (optional) is a per-sample scalar used for the
    Spearman rank check; defaults to the probability-weighted expected
    class, which is monotone-equivalent for a fitted ordinal model.
    """
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    if continuous_score is None:
        continuous_score = proba @ np.arange(proba.shape[1])
    return {
        "brier": brier_score(y_true, proba),
        "log_loss": log_loss_score(y_true, proba),
        "rps": ranked_probability_score(y_true, proba),
        "cumulative_auc": cumulative_auc(y_true, proba),
        "reliability": reliability_table(y_true, proba),
        "spearman_rho": spearman_rho(continuous_score, y_true),
    }
