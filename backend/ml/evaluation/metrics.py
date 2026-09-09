from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_multitask_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Computes comprehensive classification metrics across ABO, Rh, and Flat 8-way targets."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # ABO-specific metrics
    y_true_abo = [y.replace("+", "").replace("-", "") for y in y_true]
    y_pred_abo = [y.replace("+", "").replace("-", "") for y in y_pred]
    abo_acc = accuracy_score(y_true_abo, y_pred_abo)
    abo_f1 = f1_score(y_true_abo, y_pred_abo, average="macro", zero_division=0)

    # Rh-specific metrics
    y_true_rh = ["+" if "+" in y else "-" for y in y_true]
    y_pred_rh = ["+" if "+" in y else "-" for y in y_pred]
    rh_acc = accuracy_score(y_true_rh, y_pred_rh)
    rh_f1 = f1_score(y_true_rh, y_pred_rh, average="binary", pos_label="+", zero_division=0)

    return {
        "flat_8class_accuracy": round(float(acc), 4),
        "flat_8class_macro_f1": round(float(macro_f1), 4),
        "flat_8class_precision": round(float(macro_prec), 4),
        "flat_8class_recall": round(float(macro_rec), 4),
        "abo_4class_accuracy": round(float(abo_acc), 4),
        "abo_4class_macro_f1": round(float(abo_f1), 4),
        "rh_2class_accuracy": round(float(rh_acc), 4),
        "rh_2class_f1": round(float(rh_f1), 4),
    }


def mcnemar_test(y_true: list[str], y_pred_a: list[str], y_pred_b: list[str]) -> dict:
    """Performs McNemar's test with continuity correction to test whether model A

    and model B have statistically distinct error distributions on the same test set.
    """
    from scipy.stats import chi2

    n = len(y_true)
    correct_a = np.array([y_true[i] == y_pred_a[i] for i in range(n)])
    correct_b = np.array([y_true[i] == y_pred_b[i] for i in range(n)])

    # Contingency cells: b (A correct, B wrong), c (A wrong, B correct)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    # McNemar's statistic with Edward's continuity correction
    if (b + c) == 0:
        stat = 0.0
        p_val = 1.0
    else:
        stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
        p_val = 1.0 - chi2.cdf(stat, df=1)

    return {
        "model_a_only_correct": b,
        "model_b_only_correct": c,
        "mcnemar_statistic": round(float(stat), 4),
        "p_value": round(float(p_val), 5),
        "statistically_significant_difference": bool(p_val < 0.05),
    }


def paired_bootstrap_ci(
    y_true: list[str],
    y_pred_a: list[str],
    y_pred_b: list[str],
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
) -> dict:
    """Computes paired bootstrap confidence interval for difference in accuracy (Model A - Model B)."""
    n = len(y_true)
    y_t = np.array(y_true)
    p_a = np.array(y_pred_a)
    p_b = np.array(y_pred_b)

    rng = np.random.default_rng(seed=42)
    deltas = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        acc_a = np.mean(y_t[idx] == p_a[idx])
        acc_b = np.mean(y_t[idx] == p_b[idx])
        deltas.append(acc_a - acc_b)

    alpha = 1.0 - confidence_level
    lower = float(np.percentile(deltas, 100 * (alpha / 2.0)))
    upper = float(np.percentile(deltas, 100 * (1.0 - alpha / 2.0)))
    mean_diff = float(np.mean(deltas))

    return {
        "mean_accuracy_difference": round(mean_diff, 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "confidence_level": confidence_level,
        "significant_at_level": bool(lower > 0 or upper < 0),
    }
