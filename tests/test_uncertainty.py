import numpy as np
import pytest

from backend.ml.uncertainty.calibration import (
    TemperatureScaling,
    compute_brier_score,
    compute_expected_calibration_error,
)
from backend.ml.uncertainty.conformal import SplitConformalPredictor


def test_split_conformal_calibration_and_prediction():
    # 8 classes
    class_labels = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O-", "O+"]
    conformal = SplitConformalPredictor(alpha=0.10, class_labels=class_labels)

    # 100 calibration samples where the true class has probability 0.85
    n = 100
    cal_probs = np.zeros((n, 8))
    cal_labels = np.random.randint(0, 8, size=n)
    for i in range(n):
        cal_probs[i, cal_labels[i]] = 0.85
        # spread remaining probability
        remaining = (1.0 - 0.85) / 7.0
        for c in range(8):
            if c != cal_labels[i]:
                cal_probs[i, c] = remaining

    q_hat = conformal.calibrate(cal_probs, cal_labels)
    assert 0.0 <= q_hat <= 1.0

    # Test sample with clear winner
    sample_probs = np.zeros(8)
    sample_probs[0] = 0.90
    sample_probs[1:] = 0.10 / 7.0

    pred_set = conformal.predict_set(sample_probs)
    assert "A+" in pred_set
    assert len(pred_set) <= 2


def test_conformal_abstention_decision():
    conformal = SplitConformalPredictor(alpha=0.10)
    conformal.q_hat = 0.70  # threshold p >= 0.30

    # High confidence test
    confident_probs = np.array([0.85, 0.05, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01])
    eval_accepted = conformal.evaluate_decision(confident_probs)
    assert eval_accepted["status"] == "ACCEPTED"
    assert eval_accepted["abstention_reason"] is None
    assert eval_accepted["top_class"] == "A+"

    # Uniform/ambiguous test
    ambiguous_probs = np.full(8, 0.125)
    eval_withheld = conformal.evaluate_decision(ambiguous_probs)
    assert eval_withheld["status"] == "PREDICTION_WITHHELD"
    assert eval_withheld["abstention_reason"] is not None


def test_temperature_scaler_and_ece():
    scaler = TemperatureScaling(temperature=1.5)
    logits = np.array([[2.0, 0.5, -1.0], [0.1, 3.0, 0.5]])
    scaled = scaler.calibrate(logits, is_logit=True)
    assert scaled.shape == logits.shape
    assert np.allclose(np.sum(scaled, axis=-1), [1.0, 1.0])

    # Calibration error check
    confidences = np.array([0.9, 0.8, 0.7, 0.95])
    accuracies = np.array([1.0, 1.0, 1.0, 1.0])
    res = compute_expected_calibration_error(confidences, accuracies, n_bins=5)
    assert "ece" in res
    assert "mce" in res

    # Brier score check
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    one_hot = np.array([[1.0, 0.0], [0.0, 1.0]])
    brier = compute_brier_score(probs, one_hot)
    assert brier < 0.10

