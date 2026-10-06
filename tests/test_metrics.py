"""Hand-checked cases for the evaluation metrics."""

import numpy as np
import pytest

from fraud.metrics import evaluate, pr_auc, recall_at_precision

# Ranked by score: 0.8 (fraud), 0.4 (legit), 0.35 (fraud), 0.1 (legit).
Y = np.array([0, 0, 1, 1])
S = np.array([0.1, 0.4, 0.35, 0.8])


def test_average_precision_hand_checked() -> None:
    # Precision 1.0 at recall 0.5, then 2/3 at recall 1.0: 0.5 * 1 + 0.5 * 2/3.
    assert pr_auc(Y, S) == pytest.approx(5 / 6)


def test_average_precision_of_constant_score_is_prevalence() -> None:
    y = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    assert pr_auc(y, np.full(10, 0.2)) == pytest.approx(0.2)


def test_recall_at_precision_hand_checked() -> None:
    # Threshold 0.35 gives precision 2/3 and recall 1.0; threshold 0.8 gives 1.0 and 0.5.
    assert recall_at_precision(Y, S, 0.50) == pytest.approx(1.0)
    assert recall_at_precision(Y, S, 0.80) == pytest.approx(0.5)


def test_recall_at_precision_unreachable_returns_zero() -> None:
    # The only fraud has the lowest score, so the best precision is 1/4.
    y = np.array([1, 0, 0, 0])
    s = np.array([0.1, 0.9, 0.8, 0.7])
    assert recall_at_precision(y, s, 0.50) == 0.0


def test_evaluate_returns_plain_floats() -> None:
    result = evaluate(Y, S)
    assert set(result) == {"pr_auc", "roc_auc", "recall_at_p50", "recall_at_p80"}
    assert all(type(v) is float for v in result.values())
    assert result["roc_auc"] == pytest.approx(0.75)
