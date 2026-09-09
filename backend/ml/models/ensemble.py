from typing import Optional

import numpy as np


class LearnedEnsembleFusion:
    """Learned Stacking / Weighted Averaging for Multi-Scale Model Ensembles.

    Replaces fixed 0.5/0.5 weighting with optimal blending weights determined
    via validation simplex optimization or logistic stacking.
    """

    def __init__(self, initial_weights: Optional[list[float]] = None) -> None:
        if initial_weights is not None:
            w = np.array(initial_weights, dtype=np.float32)
            self.weights = w / np.sum(w)
        else:
            self.weights = np.array([0.5, 0.5], dtype=np.float32)

    def fit(self, predictions_list: list[np.ndarray], y_true_one_hot: np.ndarray) -> np.ndarray:
        """Optimizes ensemble weights using Nelder-Mead / projected gradient on validation log-loss."""
        from scipy.optimize import minimize

        def loss_fn(raw_weights):
            # Softmax to enforce positive weights summing to 1
            w = np.exp(raw_weights) / np.sum(np.exp(raw_weights))
            blended = np.zeros_like(predictions_list[0])
            for i, p in enumerate(predictions_list):
                blended += w[i] * p
            blended = np.clip(blended, 1e-7, 1.0 - 1e-7)
            # Categorical cross-entropy
            return -np.mean(np.sum(y_true_one_hot * np.log(blended), axis=-1))

        init_raw = np.zeros(len(predictions_list))
        res = minimize(loss_fn, init_raw, method="Nelder-Mead")
        opt_w = np.exp(res.x) / np.sum(np.exp(res.x))
        self.weights = opt_w.astype(np.float32)
        return self.weights

    def predict_proba(self, predictions_list: list[np.ndarray]) -> np.ndarray:
        """Applies learned weights to combine model probability distributions."""
        blended = np.zeros_like(predictions_list[0])
        for weight, preds in zip(self.weights, predictions_list):
            blended += weight * preds
        # Re-normalize to ensure exact simplex membership
        sums = np.sum(blended, axis=-1, keepdims=True)
        return blended / np.where(sums > 0, sums, 1.0)
