"""Final evaluation on the held-out test split. Run once.

This is the only module that loads test features or test labels. The model
(Phase 4) and the decision thresholds (fraud.threshold) are fixed beforehand on
train and validation, and nothing here is adjusted using test results. A second
run is refused unless --force is passed, which records that the test set has been
used more than once.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from fraud.config import INTERIM_DIR, RANDOM_SEED
from fraud.data import ID_COL, TARGET, TIME_COL
from fraud.features import FEATURE_PATHS
from fraud.metrics import evaluate, pr_auc
from fraud.split import read_test
from fraud.threshold import (
    THRESHOLD_JSON,
    flag_everything,
    flag_nothing,
    policy_cost,
    write_cost_report,
)
from fraud.train import (
    FINAL_MODEL_PATH,
    FINAL_PARAMS_JSON,
    METRICS_JSON,
    to_category_codes,
    update_metrics,
)

FINAL_KEY = "final_test"
TEST_SCORES_PATH: Path = INTERIM_DIR / "test_scores.parquet"
N_BOOTSTRAP = 200


def previous_runs(path: Path = METRICS_JSON) -> int:
    """Number of earlier final evaluations recorded in metrics.json (0 if none)."""
    metrics = json.loads(path.read_text()) if path.exists() else {}
    return int(metrics[FINAL_KEY].get("runs", 1)) if FINAL_KEY in metrics else 0


def score_features(features: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
    """Score a FeatureBuilder output with the saved final model."""
    X = to_category_codes(features, params["categorical_as_codes"])
    assert list(X.columns) == params["feature_names"], "feature columns differ from training"
    booster = lgb.Booster(model_file=FINAL_MODEL_PATH)
    return np.asarray(booster.predict(X), dtype=np.float64)


def bootstrap_pr_auc(
    y_true: np.ndarray, scores: np.ndarray, n: int = N_BOOTSTRAP, seed: int = RANDOM_SEED
) -> dict[str, float]:
    """Percentile 95% interval for PR-AUC from resampling rows with replacement."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        values.append(pr_auc(y_true[idx], scores[idx]))
    low, high = np.percentile(values, [2.5, 97.5])
    return {"low": float(low), "high": float(high), "resamples": n, "seed": seed}


def test_report(
    y: np.ndarray, scores: np.ndarray, amounts: np.ndarray, thresholds: dict[str, Any]
) -> dict[str, Any]:
    """Every test result, using thresholds chosen on validation only."""
    chosen = thresholds["chosen"]
    fee = chosen["review_fee"]
    sensitivity = []
    for row in thresholds["sensitivity"]:
        result = policy_cost(y, scores, amounts, row["threshold"], row["review_fee"])
        sensitivity.append(
            {
                "review_fee": row["review_fee"],
                "threshold": row["threshold"],
                "cost": result["cost"],
                "precision": result["precision"],
                "recall": result["recall"],
                "flag_nothing_cost": flag_nothing(y, scores, amounts, row["review_fee"])["cost"],
                "flag_everything_cost": flag_everything(y, scores, amounts, row["review_fee"])["cost"],
            }
        )
    return {
        "rows": int(len(y)),
        "frauds": int(y.sum()),
        "fraud_rate": float(y.mean()),
        **evaluate(y, scores),
        "pr_auc_ci95": bootstrap_pr_auc(y, scores),
        "threshold": chosen["threshold"],
        "review_fee": fee,
        "at_threshold": policy_cost(y, scores, amounts, chosen["threshold"], fee),
        "flag_nothing": flag_nothing(y, scores, amounts, fee),
        "flag_everything": flag_everything(y, scores, amounts, fee),
        "sensitivity": sensitivity,
    }


def run(force: bool = False) -> dict[str, Any]:
    """Score the test split once, report metrics and costs, and save test scores."""
    runs = previous_runs()
    if runs and not force:
        print(
            f"Refusing to run: reports/metrics.json already contains '{FINAL_KEY}'. "
            "The test set is evaluated once. Pass --force only if you accept that the "
            "test set will have been used more than once.",
            file=sys.stderr,
        )
        sys.exit(1)
    if runs:
        print(f"WARNING: --force used. This is test evaluation number {runs + 1}; "
              "the test set has now been used more than once.", file=sys.stderr)

    thresholds = json.loads(THRESHOLD_JSON.read_text())
    params = json.loads(FINAL_PARAMS_JSON.read_text())
    assert thresholds["assumptions"]["chosen_on"] == "validation"

    # First and only point at which test data is loaded.
    features = pd.read_parquet(FEATURE_PATHS["test"])
    labels = read_test([ID_COL, TARGET, "TransactionAmt"])
    assert len(features) == len(labels)

    y = labels[TARGET].to_numpy()
    scores = score_features(features, params)
    report = test_report(y, scores, labels["TransactionAmt"].to_numpy(), thresholds)
    report["runs"] = runs + 1

    pd.DataFrame(
        {ID_COL: labels[ID_COL], TIME_COL: labels[TIME_COL], "label": y, "score": scores}
    ).to_parquet(TEST_SCORES_PATH, index=False)
    write_cost_report(thresholds["sensitivity"], report["sensitivity"])
    update_metrics({FINAL_KEY: report})
    print_summary(report)
    return report


def print_summary(report: dict[str, Any]) -> None:
    """Print validation against test metrics and the test cost comparison."""
    metrics = json.loads(METRICS_JSON.read_text())
    valid = metrics[json.loads(FINAL_PARAMS_JSON.read_text())["model"]]
    ci = report["pr_auc_ci95"]
    print(f"Test rows {report['rows']:,}, frauds {report['frauds']:,} ({report['fraud_rate']:.4%})")
    print(f"{'metric':<16}{'validation':>12}{'test':>12}")
    for key in ("pr_auc", "roc_auc", "recall_at_p50", "recall_at_p80"):
        print(f"{key:<16}{valid[key]:>12.4f}{report[key]:>12.4f}")
    print(f"Test PR-AUC 95% bootstrap CI: [{ci['low']:.4f}, {ci['high']:.4f}] ({ci['resamples']} resamples)")
    at = report["at_threshold"]
    print(f"At threshold {report['threshold']:.6g}: precision {at['precision']:.4f}, recall {at['recall']:.4f}")
    print(f"Test cost at fee {report['review_fee']:g}: chosen {at['cost']:,.2f}, "
          f"flag nothing {report['flag_nothing']['cost']:,.2f}, "
          f"flag everything {report['flag_everything']['cost']:,.2f}")
    for r in report["sensitivity"]:
        print(f"  fee {r['review_fee']:>4g}: threshold {r['threshold']:.6g}, test cost {r['cost']:,.2f} "
              f"(nothing {r['flag_nothing_cost']:,.2f}, everything {r['flag_everything_cost']:,.2f})")
    print(f"Wrote {FINAL_KEY} to {METRICS_JSON} and {TEST_SCORES_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="re-run even though the test set has already been evaluated")
    run(force=parser.parse_args().force)
