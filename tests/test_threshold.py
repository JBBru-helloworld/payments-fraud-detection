"""Hand-checked cost cases and grid search checks for threshold selection."""

import numpy as np
import pytest

from fraud.threshold import (
    best_threshold,
    flag_everything,
    flag_nothing,
    grid_costs,
    policy_cost,
    threshold_grid,
)

Y = np.array([1, 0, 1, 0])
S = np.array([0.9, 0.8, 0.3, 0.1])
AMT = np.array([100.0, 50.0, 20.0, 10.0])


def test_policy_cost_hand_checked() -> None:
    # Threshold 0.5 flags 0.9 (fraud, caught) and 0.8 (false alarm); misses the 20.00 fraud.
    result = policy_cost(Y, S, AMT, threshold=0.5, review_fee=5.0)
    assert result["cost"] == pytest.approx(20.0 + 5.0)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 1)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)


def test_score_equal_to_threshold_is_flagged() -> None:
    result = policy_cost(Y, S, AMT, threshold=0.3, review_fee=5.0)
    assert (result["tp"], result["fp"], result["fn"]) == (2, 1, 0)
    assert result["cost"] == pytest.approx(5.0)


def test_reference_policies() -> None:
    assert flag_nothing(Y, S, AMT, 5.0)["cost"] == pytest.approx(120.0)
    assert flag_nothing(Y, S, AMT, 5.0)["precision"] == 0.0
    assert flag_everything(Y, S, AMT, 5.0)["cost"] == pytest.approx(10.0)


def test_amounts_are_rounded_before_use() -> None:
    stored = np.array([49.99], dtype=np.float32)
    result = policy_cost(np.array([1]), np.array([0.1]), stored, threshold=0.5, review_fee=5.0)
    assert result["cost"] == 49.99


def test_grid_matches_direct_cost_and_best_is_not_beaten() -> None:
    rng = np.random.default_rng(42)
    y = rng.binomial(1, 0.1, size=2_000)
    scores = np.clip(0.3 * y + rng.random(2_000) * 0.7, 0, 1)
    amounts = rng.uniform(1, 500, size=2_000).astype(np.float32)
    grid = threshold_grid(scores)
    table = grid_costs(y, scores, amounts, grid, review_fee=5.0)

    direct = np.array([policy_cost(y, scores, amounts, t, 5.0)["cost"] for t in grid])
    np.testing.assert_allclose(table["cost"].to_numpy(), direct, rtol=1e-9)

    best = best_threshold(table)
    assert best["cost"] <= direct.min() + 1e-6
    assert best["cost"] == pytest.approx(policy_cost(y, scores, amounts, best["threshold"], 5.0)["cost"])


def test_final_eval_guard_refuses_second_run(tmp_path, monkeypatch) -> None:
    import json

    from fraud import final_eval

    metrics = tmp_path / "metrics.json"
    assert final_eval.previous_runs(metrics) == 0
    metrics.write_text(json.dumps({"final_test": {"runs": 1}}))
    assert final_eval.previous_runs(metrics) == 1

    monkeypatch.setattr(final_eval, "previous_runs", lambda: 1)
    with pytest.raises(SystemExit):
        final_eval.run(force=False)
