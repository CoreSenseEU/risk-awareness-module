"""Unit tests for riskam.eval_metrics."""

import numpy as np
import pytest

from riskam.eval_metrics import (
    N_CLASSES,
    class_centre,
    classification_report,
    confusion_matrix,
    macro_prf,
    micro_prf,
    per_class_prf,
    predicted_class,
    regression_errors,
)
from riskam.score import RISK_SCORE_BREAKPOINTS, VERY_SMALL_RISK_VALUE


class TestPredictedClass:
    def test_zero_is_class_0(self):
        assert predicted_class(0.0) == 0
        assert predicted_class(-0.1) == 0

    def test_just_above_zero_is_class_1(self):
        assert predicted_class(VERY_SMALL_RISK_VALUE) == 1
        assert predicted_class(0.15) == 1

    def test_mid_risk_is_class_2(self):
        assert predicted_class(RISK_SCORE_BREAKPOINTS[1]) == 2  # 0.3
        assert predicted_class(0.45) == 2

    def test_high_risk_is_class_3(self):
        assert predicted_class(RISK_SCORE_BREAKPOINTS[2]) == 3  # 0.6
        assert predicted_class(0.95) == 3
        assert predicted_class(1.0) == 3


class TestClassCentre:
    def test_class_0_centre_is_zero(self):
        assert class_centre(0) == 0.0

    def test_class_3_centre_is_between_0_6_and_1(self):
        c = class_centre(3)
        assert 0.6 < c < 1.0
        assert c == pytest.approx(0.8)


class TestConfusionMatrix:
    def test_perfect_predictions(self):
        y_true = [0, 1, 2, 3, 1, 2]
        cm = confusion_matrix(y_true, y_true)
        assert cm.shape == (N_CLASSES, N_CLASSES)
        # All counts on the diagonal.
        assert cm[0, 0] == 1
        assert cm[1, 1] == 2
        assert cm[2, 2] == 2
        assert cm[3, 3] == 1
        assert cm.sum() == 6

    def test_misclassification_counted_off_diagonal(self):
        y_true = [1, 1, 2]
        y_pred = [1, 2, 1]  # one correct 1, one 1→2, one 2→1
        cm = confusion_matrix(y_true, y_pred)
        assert cm[1, 1] == 1
        assert cm[1, 2] == 1
        assert cm[2, 1] == 1


class TestPerClassPRF:
    def test_single_class_perfect(self):
        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        cm[1, 1] = 10
        out = per_class_prf(cm)
        assert out["1"]["precision"] == 1.0
        assert out["1"]["recall"] == 1.0
        assert out["1"]["f1"] == 1.0
        assert out["1"]["support"] == 10
        # Classes with no support → all zeros.
        assert out["0"]["support"] == 0
        assert out["0"]["precision"] == 0.0

    def test_known_precision_recall(self):
        # Class 2: 8 correct, 2 predicted as 2 but actually class 1
        # (FP=2), 4 actually class 2 but predicted as 3 (FN=4).
        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        cm[2, 2] = 8
        cm[1, 2] = 2   # FP for class 2
        cm[2, 3] = 4   # FN for class 2
        out = per_class_prf(cm)
        assert out["2"]["precision"] == pytest.approx(8 / 10)  # tp / (tp+fp)
        assert out["2"]["recall"] == pytest.approx(8 / 12)     # tp / (tp+fn)

    def test_zero_denominator_produces_zero(self):
        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        out = per_class_prf(cm)
        for c in range(N_CLASSES):
            assert out[str(c)]["f1"] == 0.0


class TestMacroMicroPRF:
    def test_macro_ignores_zero_support_classes(self):
        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        cm[1, 1] = 5  # only class 1 has support; precision=recall=1
        per_class = per_class_prf(cm)
        macro = macro_prf(per_class)
        # Only the one supported class contributes to macro average.
        assert macro["f1"] == 1.0

    def test_micro_equals_accuracy(self):
        cm = np.array(
            [
                [3, 1, 0, 0],
                [0, 4, 0, 0],
                [1, 0, 2, 0],
                [0, 0, 0, 5],
            ],
            dtype=int,
        )
        total = cm.sum()        # 16
        correct = np.trace(cm)   # 14
        micro = micro_prf(cm)
        assert micro["f1"] == pytest.approx(correct / total)


class TestRegressionErrors:
    def test_perfect_centres_zero_error(self):
        # Predictions exactly at class centres.
        y_true = [0, 1, 2, 3]
        preds = [class_centre(c) for c in y_true]
        out = regression_errors(y_true, preds)
        assert out["mae"] == pytest.approx(0.0)
        assert out["rmse"] == pytest.approx(0.0)

    def test_known_mae(self):
        # Single sample, class 3 centre ≈ 0.8; predict 0.5 → error 0.3.
        out = regression_errors([3], [0.5])
        assert out["mae"] == pytest.approx(0.3)
        assert out["rmse"] == pytest.approx(0.3)

    def test_empty_inputs(self):
        out = regression_errors([], [])
        assert out == {"mae": 0.0, "rmse": 0.0}


class TestClassificationReportEndToEnd:
    def test_perfect_predictions(self):
        y_true = [0, 1, 2, 3]
        preds = [class_centre(c) for c in y_true]
        report = classification_report(y_true, preds)
        assert report["micro"]["f1"] == 1.0
        assert report["macro"]["f1"] == 1.0
        assert report["regression"]["mae"] == pytest.approx(0.0, abs=1e-9)

    def test_report_has_expected_keys(self):
        report = classification_report([1, 2], [0.2, 0.5])
        assert set(report.keys()) == {
            "confusion_matrix",
            "per_class",
            "macro",
            "micro",
            "regression",
        }
        assert len(report["confusion_matrix"]) == N_CLASSES
