"""Baseline models fitted on train and evaluated on validation only."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud.config import MERGED_PARQUET, RANDOM_SEED, REPORTS_DIR
from fraud.data import TARGET
from fraud.features import ENGINEERED_NUMERIC, FEATURE_PATHS
from fraud.metrics import evaluate
from fraud.split import TIME_COL, split_masks

METRICS_JSON: Path = REPORTS_DIR / "metrics.json"
C_COLS: list[str] = [f"C{i}" for i in range(1, 15)]
D_COLS: list[str] = ["D1", "D4", "D10", "D15"]
AMOUNT_COLS: list[str] = ["TransactionAmt", "log_amt"]
BASELINE_COLS: list[str] = ["TransactionAmt", *C_COLS, *D_COLS]


def load_train_validation(path: Path = MERGED_PARQUET) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load baseline columns and return the train and validation frames. Test rows are dropped."""
    df = pd.read_parquet(path, columns=[TIME_COL, TARGET, *BASELINE_COLS])
    masks = split_masks(df[TIME_COL])
    return df.loc[masks["train"]].copy(), df.loc[masks["validation"]].copy()


def add_log_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with log1p(TransactionAmt) added as log_amt."""
    return df.assign(log_amt=np.log1p(df["TransactionAmt"].astype(np.float64)))


def prevalence_scores(y_train: pd.Series, n_rows: int) -> np.ndarray:
    """Constant score equal to the training fraud rate."""
    return np.full(n_rows, float(y_train.mean()))


def logreg_pipeline(
    dense_cols: list[str] | None = None, indicator_cols: list[str] | None = None
) -> Pipeline:
    """Median imputation, missing indicators and scaling, then logistic regression.

    Defaults reproduce the Phase 2 baseline: indicators for the D columns only.
    """
    dense_cols = dense_cols if dense_cols is not None else [*AMOUNT_COLS, *C_COLS]
    indicator_cols = indicator_cols if indicator_cols is not None else D_COLS
    preprocess = ColumnTransformer(
        [
            ("dense", SimpleImputer(strategy="median"), dense_cols),
            ("indicated", SimpleImputer(strategy="median", add_indicator=True), indicator_cols),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ]
    )


def update_metrics(results: dict[str, dict], path: Path = METRICS_JSON) -> None:
    """Merge results into metrics.json, keeping keys written by other phases."""
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2) + "\n")


def run_baselines() -> dict[str, dict]:
    """Fit both baselines on train, score validation and write metrics.json."""
    train, valid = load_train_validation()
    y_train, y_valid = train[TARGET], valid[TARGET]

    pipeline = logreg_pipeline()
    feature_cols = [*AMOUNT_COLS, *C_COLS, *D_COLS]
    pipeline.fit(add_log_amount(train)[feature_cols], y_train)
    logreg_scores = pipeline.predict_proba(add_log_amount(valid)[feature_cols])[:, 1]

    results = {
        "baseline_prevalence": evaluate(y_valid, prevalence_scores(y_train, len(valid))),
        "baseline_logreg": evaluate(y_valid, logreg_scores),
    }
    update_metrics(results)

    print(f"Validation rows: {len(valid):,}, fraud rate: {y_valid.mean():.4%}")
    print_table(results)
    return results


def print_table(results: dict[str, dict]) -> None:
    """Print a metrics table for the given models."""
    names = list(next(iter(results.values())))
    print(f"{'model':<26}" + "".join(f"{n:>15}" for n in names))
    for model, scores in results.items():
        print(f"{model:<26}" + "".join(f"{scores[n]:>15.4f}" for n in names))
    print(f"Wrote {METRICS_JSON}")


def load_labels(path: Path = MERGED_PARQUET) -> tuple[pd.Series, pd.Series]:
    """Train and validation labels in split order. Test labels are never returned."""
    df = pd.read_parquet(path, columns=[TIME_COL, TARGET])
    masks = split_masks(df[TIME_COL])
    return df.loc[masks["train"], TARGET], df.loc[masks["validation"], TARGET]


def run_engineered_check() -> dict[str, dict]:
    """Leakage sanity check: Phase 2 logistic regression plus engineered numeric features.

    Fitted on train, scored on validation. Not a final model.
    """
    raw_cols = ["TransactionAmt", *C_COLS]
    indicated = [*D_COLS, *[c for c in ENGINEERED_NUMERIC if c != "log_amt"]]
    cols = [*raw_cols, "log_amt", *indicated]
    train = pd.read_parquet(FEATURE_PATHS["train"], columns=cols)
    valid = pd.read_parquet(FEATURE_PATHS["validation"], columns=cols)
    y_train, y_valid = load_labels()
    assert len(train) == len(y_train) and len(valid) == len(y_valid)

    pipeline = logreg_pipeline([*raw_cols, "log_amt"], indicated)
    pipeline.fit(train, y_train.to_numpy())
    scores = pipeline.predict_proba(valid)[:, 1]
    results = {"logreg_engineered_check": evaluate(y_valid, scores)}
    update_metrics(results)

    existing = json.loads(METRICS_JSON.read_text())
    print_table({k: existing[k] for k in ("baseline_logreg", "logreg_engineered_check")})
    pr_auc = results["logreg_engineered_check"]["pr_auc"]
    if pr_auc > 0.6:
        raise RuntimeError(f"PR-AUC {pr_auc:.4f} exceeds 0.6: audit for leakage before continuing")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engineered-check", action="store_true", help="run the Phase 3 leakage sanity check"
    )
    if parser.parse_args().engineered_check:
        run_engineered_check()
    else:
        run_baselines()
