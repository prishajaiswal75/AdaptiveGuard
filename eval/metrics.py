"""
Metrics and cost calculations shared across Model A, Model B, and the
fair-budget ablation (Phase 4).

Cost-sensitive thresholding is required by Razorpay's grading bar
("honest metrics including false-positive cost"). It is implemented here as
a standard evaluation/business layer -- not claimed as a novel technique.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def confusion_counts(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return tp, fp, fn, tn


def compute_metrics(y_true, y_score, threshold, cost_fp, cost_fn):
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "threshold": float(threshold), "n": len(y_true),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
        "fp_cost": fp * cost_fp, "fn_cost": fn * cost_fn,
        "total_cost": fp * cost_fp + fn * cost_fn,
    }


def best_threshold_by_cost(y_true, y_score, cost_fp, cost_fn, n_steps=199):
    """Sweep thresholds on a VALIDATION set only; caller applies the chosen
    threshold to a separate test/eval set. Never fit the threshold on the
    same data it's scored against."""
    thresholds = np.linspace(0.01, 0.99, n_steps)
    best = None
    for t in thresholds:
        m = compute_metrics(y_true, y_score, t, cost_fp, cost_fn)
        if best is None or m["total_cost"] < best["total_cost"]:
            best = m
    return best


def supplementary_metrics(y_true, y_score):
    out = {}
    try:
        out["auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        out["auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    except ValueError:
        out["pr_auc"] = float("nan")
    return out
