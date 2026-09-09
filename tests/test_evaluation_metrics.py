import pytest

from backend.ml.evaluation.metrics import (
    compute_multitask_metrics,
    mcnemar_test,
    paired_bootstrap_ci,
)


def test_compute_multitask_metrics():
    y_true = ["A+", "A-", "B+", "O+", "AB-"]
    y_pred = ["A+", "A+", "B+", "O-", "AB-"]

    metrics = compute_multitask_metrics(y_true, y_pred)

    assert "flat_8class_accuracy" in metrics
    assert "abo_4class_accuracy" in metrics
    assert "rh_2class_accuracy" in metrics

    # ABO accuracy:
    # true: A, A, B, O, AB
    # pred: A, A, B, O, AB -> all 5 correct!
    assert metrics["abo_4class_accuracy"] == 1.0

    # Rh accuracy:
    # true: +, -, +, +, -
    # pred: +, +, +, -, - -> indices 1 and 3 mismatched -> 3/5 correct
    assert metrics["rh_2class_accuracy"] == 0.6


def test_mcnemar_test():
    y_true = ["A+", "B+", "O+", "AB+"] * 10
    # Model A gets all correct
    y_pred_a = list(y_true)
    # Model B gets half wrong
    y_pred_b = ["A-", "B-", "O-", "AB-"] * 5 + list(y_true[:20])

    res = mcnemar_test(y_true, y_pred_a, y_pred_b)
    assert "mcnemar_statistic" in res
    assert "p_value" in res
    assert res["model_a_only_correct"] == 20
    assert res["model_b_only_correct"] == 0
    assert res["p_value"] < 0.01
    assert res["statistically_significant_difference"] is True


def test_paired_bootstrap_ci():
    y_true = ["A+"] * 50 + ["B+"] * 50
    y_pred_a = ["A+"] * 50 + ["B+"] * 50  # 100%
    y_pred_b = ["A-"] * 50 + ["B+"] * 50  # 50%

    ci = paired_bootstrap_ci(y_true, y_pred_a, y_pred_b, n_bootstraps=200)
    assert ci["mean_accuracy_difference"] == pytest.approx(0.5, abs=0.05)
    assert ci["ci_lower"] > 0.35
    assert ci["significant_at_level"] is True
