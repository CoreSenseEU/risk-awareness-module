"""
riskam.eval_metrics

Classification and regression metrics for offline experiments.

Risk classes (0, 1, 2, 3) correspond to risk-score bins defined by
``RISK_SCORE_BREAKPOINTS`` in ``riskam.score``. Continuous predictions are
binned into classes for classification metrics. For regression-style metrics
the target is the midpoint ("centre") of each class's bin, treated as a
soft continuous target — a convenient proxy given integer-only ground truth.
"""

from __future__ import annotations

import numpy as np

from riskam.score import RISK_SCORE_BREAKPOINTS, VERY_SMALL_RISK_VALUE


N_CLASSES = 4


def predicted_class(risk_score: float) -> int:
    """Bin a continuous risk score into a class index in [0, N_CLASSES).

    Mirrors the semantics of ``experiments._eval_prediction``: class 0 is a
    point mass at 0; classes 1/2/3 cover the half-open bins defined by
    ``RISK_SCORE_BREAKPOINTS``.
    """
    if risk_score <= 0.0:
        return 0
    if risk_score < RISK_SCORE_BREAKPOINTS[1]:   # < 0.3
        return 1
    if risk_score < RISK_SCORE_BREAKPOINTS[2]:   # < 0.6
        return 2
    return 3


# Class centres used as soft regression targets.
_CLASS_CENTRES: tuple[float, ...] = (
    0.0,
    (VERY_SMALL_RISK_VALUE + RISK_SCORE_BREAKPOINTS[1]) / 2.0,
    (RISK_SCORE_BREAKPOINTS[1] + RISK_SCORE_BREAKPOINTS[2]) / 2.0,
    (RISK_SCORE_BREAKPOINTS[2] + RISK_SCORE_BREAKPOINTS[3]) / 2.0,
)


def class_centre(class_idx: int) -> float:
    return _CLASS_CENTRES[class_idx]


def confusion_matrix(y_true: list[int], y_pred: list[int]) -> np.ndarray:
    """``(N_CLASSES, N_CLASSES)`` matrix indexed ``[true, predicted]``."""
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_prf(cm: np.ndarray) -> dict:
    """Per-class precision / recall / F1 / support."""
    out: dict = {}
    for c in range(N_CLASSES):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum()) - tp
        fn = int(cm[c, :].sum()) - tp
        support = int(cm[c, :].sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        out[str(c)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }
    return out


def macro_prf(per_class: dict) -> dict:
    """Unweighted mean of per-class P/R/F1. Classes with support=0 are dropped."""
    supported = [v for v in per_class.values() if v["support"] > 0]
    if not supported:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "precision": float(np.mean([v["precision"] for v in supported])),
        "recall": float(np.mean([v["recall"] for v in supported])),
        "f1": float(np.mean([v["f1"] for v in supported])),
    }


def micro_prf(cm: np.ndarray) -> dict:
    """Micro-averaged P/R/F1 — equals accuracy for single-label classification."""
    total = int(cm.sum())
    if total == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    accuracy = float(np.trace(cm)) / total
    return {
        "precision": accuracy,
        "recall": accuracy,
        "f1": accuracy,
    }


def regression_errors(
    y_true: list[int], y_pred_continuous: list[float]
) -> dict:
    """MAE and RMSE of continuous predictions against class centres."""
    if not y_true:
        return {"mae": 0.0, "rmse": 0.0}
    targets = np.array([class_centre(t) for t in y_true], dtype=float)
    preds = np.array(y_pred_continuous, dtype=float)
    errors = preds - targets
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
    }


def classification_report(
    y_true: list[int],
    y_pred_continuous: list[float],
) -> dict:
    """One-shot classification + regression report.

    Returns a dict with keys ``confusion_matrix``, ``per_class``, ``macro``,
    ``micro``, and ``regression`` — ready to drop into an experiment's
    ``results.json`` under a single top-level key.
    """
    y_pred = [predicted_class(p) for p in y_pred_continuous]
    cm = confusion_matrix(y_true, y_pred)
    per_class = per_class_prf(cm)
    return {
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "macro": macro_prf(per_class),
        "micro": micro_prf(cm),
        "regression": regression_errors(y_true, y_pred_continuous),
    }
