from typing import Optional

import numpy as np


class TemperatureScaling:
    """Post-hoc probability calibration via temperature scaling (Guo et al., 2017).

    Learns a single scalar temperature T > 0 on validation logits to align confidence
    with true empirical accuracy without altering class rank order.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = max(float(temperature), 1e-4)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Optimizes temperature using negative log-likelihood on validation logits."""
        from scipy.optimize import minimize

        def nll_loss(t):
            temp = max(float(t[0]), 1e-3)
            scaled = logits / temp
            exp_scaled = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
            probs = exp_scaled / np.sum(exp_scaled, axis=-1, keepdims=True)
            probs = np.clip(probs, 1e-7, 1.0 - 1e-7)

            if labels.ndim == 1:
                nll = -np.mean(np.log(probs[np.arange(len(labels)), labels]))
            else:
                nll = -np.mean(np.sum(labels * np.log(probs), axis=-1))
            return nll

        res = minimize(nll_loss, [1.0], method="Nelder-Mead")
        self.temperature = max(float(res.x[0]), 1e-3)
        return self.temperature

    def calibrate(self, logits_or_probs: np.ndarray, is_logit: bool = True) -> np.ndarray:
        """Applies learned temperature to scale logits or log-probabilities."""
        if not is_logit:
            logits = np.log(np.clip(logits_or_probs, 1e-7, 1.0))
        else:
            logits = logits_or_probs

        scaled = logits / self.temperature
        exp_scaled = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=-1, keepdims=True)


def compute_expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(confidences)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            diff = abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += diff * prop_in_bin
            mce = max(mce, diff)

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
    }


def compute_brier_score(probabilities: np.ndarray, y_true_one_hot: np.ndarray) -> float:
    """Computes Brier Score: mean squared error between forecast probabilities and actual outcomes."""
    return float(np.mean(np.sum((probabilities - y_true_one_hot) ** 2, axis=-1)))
