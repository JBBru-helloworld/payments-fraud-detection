"""Ranking metrics for imbalanced fraud scores."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

REPORTED_PRECISIONS: tuple[float, ...] = (0.50, 0.80)


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision, the step-wise area under the precision-recall curve."""
    return float(average_precision_score(y_true, y_score))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve."""
    return float(roc_auc_score(y_true, y_score))


def recall_at_precision(
    y_true: np.ndarray, y_score: np.ndarray, target: float
) -> float:
    """Maximum recall over thresholds with precision >= target, or 0.0 if unreachable."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    # The final (precision=1, recall=0) point has no threshold, so it can only yield 0.0.
    reachable = recall[precision >= target]
    return float(reachable.max()) if reachable.size else 0.0


def precision_key(target: float) -> str:
    """Metric name for recall at a given precision, for example recall_at_p50."""
    return f"recall_at_p{round(target * 100):02d}"


def evaluate(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Return PR-AUC, ROC-AUC and recall at each reported precision as a plain dict."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    results = {"pr_auc": pr_auc(y_true, y_score), "roc_auc": roc_auc(y_true, y_score)}
    for target in REPORTED_PRECISIONS:
        results[precision_key(target)] = recall_at_precision(y_true, y_score, target)
    return results
