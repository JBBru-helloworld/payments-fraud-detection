"""Baseline models fitted on train and evaluated on validation only."""

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


def logreg_pipeline() -> Pipeline:
    """Median imputation, D-column missing indicators and scaling, then logistic regression."""
    preprocess = ColumnTransformer(
        [
            ("dense", SimpleImputer(strategy="median"), [*AMOUNT_COLS, *C_COLS]),
            ("d", SimpleImputer(strategy="median", add_indicator=True), D_COLS),
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
    names = list(next(iter(results.values())))
    print(f"{'model':<22}" + "".join(f"{n:>15}" for n in names))
    for model, scores in results.items():
        print(f"{model:<22}" + "".join(f"{scores[n]:>15.4f}" for n in names))
    print(f"Wrote {METRICS_JSON}")
    return results


if __name__ == "__main__":
    run_baselines()
