from typing import Optional

import numpy as np


class SplitConformalPredictor:
    """Split Conformal Prediction for Fingerprint Biometric Classification.

    Provides distribution-free, finite-sample marginal coverage guarantees:
        P(Y_{test} in C(X_{test})) >= 1 - alpha
    Converts a forced overconfident guess into a rigorous prediction set with
    a formal abstention / rejection policy.
    """

    def __init__(self, alpha: float = 0.10, class_labels: Optional[list[str]] = None) -> None:
        self.alpha = float(alpha)
        self.q_hat: float = 1.0
        self.class_labels = class_labels or ["A+", "A-", "AB+", "AB-", "B+", "B-", "O-", "O+"]

    def calibrate(self, cal_probabilities: np.ndarray, cal_labels: np.ndarray) -> float:
        """Computes non-conformity threshold quantile q_hat on held-out calibration set.

        Non-conformity score: s_i = 1 - P(Y_i | X_i)
        """
        n = len(cal_probabilities)
        if n == 0:
            self.q_hat = 1.0
            return self.q_hat

        # Calculate scores for true label
        if cal_labels.ndim == 1:
            true_probs = cal_probabilities[np.arange(n), cal_labels]
        else:
            true_probs = np.sum(cal_probabilities * cal_labels, axis=-1)

        scores = 1.0 - true_probs

        # Finite sample adjusted quantile index
        p_index = int(np.ceil((n + 1) * (1.0 - self.alpha))) / n
        p_index = min(max(p_index, 0.0), 1.0)
        self.q_hat = float(np.quantile(scores, p_index, method="higher"))
        return self.q_hat

    def predict_set(self, probabilities: np.ndarray) -> list[str]:
        """Generates conformal prediction set for a single sample probability distribution."""
        probs = np.asarray(probabilities)
        # Include all classes whose non-conformity score <= q_hat
        # i.e., 1 - p <= q_hat <=> p >= 1 - q_hat
        threshold = max(0.0, 1.0 - self.q_hat)
        included = [
            self.class_labels[i]
            for i, p in enumerate(probs)
            if p >= threshold
        ]
        if not included:
            # Fallback: include argmax if set would otherwise be empty
            included = [self.class_labels[int(np.argmax(probs))]]
        return included

    def evaluate_decision(
        self,
        probabilities: np.ndarray,
        max_set_size_for_acceptance: int = 2,
        min_top_confidence: float = 0.50,
    ) -> dict:
        """Evaluates whether prediction meets safety criteria or should be withheld."""
        prediction_set = self.predict_set(probabilities)
        top_class = self.class_labels[int(np.argmax(probabilities))]
        top_confidence = float(np.max(probabilities))

        # Abstention rule
        withhold = (len(prediction_set) > max_set_size_for_acceptance) or (top_confidence < min_top_confidence)

        return {
            "prediction_set": prediction_set,
            "set_size": len(prediction_set),
            "top_class": top_class,
            "top_confidence": round(top_confidence, 4),
            "status": "PREDICTION_WITHHELD" if withhold else "ACCEPTED",
            "coverage_guarantee": f"{int((1 - self.alpha) * 100)}%",
            "abstention_reason": (
                "Anatomical ambiguity exceeds safety threshold; set size > 2"
                if len(prediction_set) > max_set_size_for_acceptance
                else "Confidence below screening margin"
                if top_confidence < min_top_confidence
                else None
            ),
        }
