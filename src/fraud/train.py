"""Baselines and model comparison: every model is fitted on train and evaluated on validation only.

Test features and test labels are never loaded here.
"""

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from fraud.config import INTERIM_DIR, MODELS_DIR, RANDOM_SEED, REPORTS_DIR
from fraud.data import ID_COL, TARGET
from fraud.features import ENGINEERED_NUMERIC, FEATURE_PATHS
from fraud.metrics import evaluate, pr_auc
from fraud.split import read_train_validation

METRICS_JSON: Path = REPORTS_DIR / "metrics.json"
C_COLS: list[str] = [f"C{i}" for i in range(1, 15)]
D_COLS: list[str] = ["D1", "D4", "D10", "D15"]
AMOUNT_COLS: list[str] = ["TransactionAmt", "log_amt"]
BASELINE_COLS: list[str] = ["TransactionAmt", *C_COLS, *D_COLS]

FINAL_MODEL_PATH: Path = MODELS_DIR / "final_model.txt"
FINAL_PARAMS_JSON: Path = REPORTS_DIR / "final_model_params.json"
VALID_SCORES_PATH: Path = INTERIM_DIR / "valid_scores.parquet"
OPTUNA_TRIALS_CSV: Path = REPORTS_DIR / "optuna_trials.csv"
OPTUNA_STORAGE = f"sqlite:///{MODELS_DIR / 'optuna_study.db'}"

ONE_HOT_MAX_LEVELS = 20
HIGH_CARDINALITY_LEVELS = 100
MAX_ROUNDS = 2_000
PATIENCE = 100
N_TRIALS = 40
LGBM_BASE_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "average_precision",
    "seed": RANDOM_SEED,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}
# LightGBM defaults for the tuned parameters; Optuna evaluates this point first.
TUNING_START: dict[str, float] = {
    "num_leaves": 31,
    "learning_rate": 0.1,
    "min_child_samples": 20,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "lambda_l1": 1e-8,
    "lambda_l2": 1e-8,
}


# Data loading


def load_train_validation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Baseline columns and labels for the train and validation windows."""
    return read_train_validation([TARGET, *BASELINE_COLS])


def load_labels() -> tuple[pd.Series, pd.Series]:
    """Train and validation labels in split order. Test labels are never read."""
    train, valid = read_train_validation([TARGET])
    return train[TARGET], valid[TARGET]


def load_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """FeatureBuilder output for train and validation. The test parquet is not read."""
    return (
        pd.read_parquet(FEATURE_PATHS["train"]),
        pd.read_parquet(FEATURE_PATHS["validation"]),
    )


# Phase 2 baselines


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
    names = [n for n in next(iter(results.values())) if n not in ("params",)]
    print(f"{'model':<26}" + "".join(f"{n:>15}" for n in names))
    for model, scores in results.items():
        cells = []
        for n in names:
            value = scores.get(n)
            cells.append(f"{'n/a':>15}" if value is None else f"{value:>15.4f}")
        print(f"{model:<26}" + "".join(cells))
    print(f"Wrote {METRICS_JSON}")


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
    pr = results["logreg_engineered_check"]["pr_auc"]
    if pr > 0.6:
        raise RuntimeError(f"PR-AUC {pr:.4f} exceeds 0.6: audit for leakage before continuing")
    return results


# Phase 4 model comparison


def categorical_levels(X: pd.DataFrame) -> dict[str, int]:
    """Number of train-fitted levels for each categorical column."""
    return {c: len(X[c].cat.categories) for c in X.select_dtypes("category").columns}


def to_category_codes(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Replace categorical columns with float32 integer codes (NaN when missing)."""
    codes = {c: X[c].cat.codes.astype(np.float32).replace(-1, np.nan) for c in cols}
    return X.assign(**codes)


def timed(fit: Callable[[], Any]) -> tuple[Any, float]:
    """Run a fitting function and return its result and wall-clock seconds."""
    start = time.perf_counter()
    result = fit()
    return result, round(time.perf_counter() - start, 1)


def logreg_full_frame(X: pd.DataFrame, one_hot: list[str], dropped: list[str]) -> pd.DataFrame:
    """Categoricals to plain strings with 'missing' for NaN; high-cardinality raw columns removed."""
    out = X.drop(columns=dropped)
    for col in one_hot:
        out[col] = out[col].astype(object).where(out[col].notna(), "missing")
    return out


def logreg_full_pipeline(numeric: list[str], one_hot: list[str]) -> Pipeline:
    """Imputation with indicators, one-hot for low-cardinality categoricals, scaling, logistic regression."""
    numeric_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("float32", FunctionTransformer(lambda a: a.astype(np.float32))),
        ]
    )
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    return Pipeline(
        [
            ("preprocess", ColumnTransformer([("num", numeric_steps, numeric), ("cat", encoder, one_hot)])),
            ("scale", StandardScaler(copy=False)),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ]
    )


def fit_logreg_full(
    X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame
) -> tuple[np.ndarray, float]:
    """Fit the full logistic regression and return validation scores and training seconds."""
    levels = categorical_levels(X_train)
    one_hot = [c for c, n in levels.items() if n <= ONE_HOT_MAX_LEVELS]
    dropped = [c for c, n in levels.items() if n > ONE_HOT_MAX_LEVELS]
    numeric = [c for c in X_train.columns if c not in levels]
    pipeline = logreg_full_pipeline(numeric, one_hot)
    train = logreg_full_frame(X_train, one_hot, dropped)
    _, seconds = timed(lambda: pipeline.fit(train, y_train.to_numpy()))
    scores = pipeline.predict_proba(logreg_full_frame(X_valid, one_hot, dropped))[:, 1]
    return scores, seconds


def fit_isolation_forest(
    X_train: pd.DataFrame, X_valid: pd.DataFrame
) -> tuple[np.ndarray, float]:
    """Unsupervised reference: IsolationForest on median-imputed numeric features."""
    numeric = [c for c in X_train.columns if c not in categorical_levels(X_train)]
    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("model", IsolationForest(random_state=RANDOM_SEED, n_jobs=-1)),
        ]
    )
    _, seconds = timed(lambda: pipeline.fit(X_train[numeric]))
    return -pipeline.score_samples(X_valid[numeric]), seconds


def lgb_datasets(
    X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series
) -> tuple[lgb.Dataset, lgb.Dataset]:
    """Train and validation Datasets sharing bins and pandas category mappings."""
    params = {"feature_pre_filter": False, "verbosity": -1}
    dtrain = lgb.Dataset(X_train, y_train.to_numpy(), params=params, free_raw_data=False)
    dvalid = lgb.Dataset(X_valid, y_valid.to_numpy(), reference=dtrain, params=params)
    return dtrain, dvalid


def train_lgbm(
    params: dict[str, Any], dtrain: lgb.Dataset, dvalid: lgb.Dataset
) -> tuple[lgb.Booster, float]:
    """Train with early stopping on validation average precision; return booster and seconds."""
    callbacks = [lgb.early_stopping(PATIENCE, verbose=False)]
    return timed(
        lambda: lgb.train(
            {**LGBM_BASE_PARAMS, **params}, dtrain, MAX_ROUNDS, valid_sets=[dvalid], callbacks=callbacks
        )
    )


def lgbm_result(
    booster: lgb.Booster, seconds: float, X_valid: pd.DataFrame, y_valid: pd.Series
) -> tuple[dict[str, Any], np.ndarray]:
    """Validation metrics, training time and best iteration for a booster, plus its scores."""
    scores = booster.predict(X_valid, num_iteration=booster.best_iteration)
    result = {**evaluate(y_valid, scores), "train_seconds": seconds, "best_iteration": booster.best_iteration}
    return result, scores


def tuning_params(trial: optuna.Trial, start_params: dict[str, Any]) -> dict[str, Any]:
    """Search space on top of the starting setup chosen from models 2 to 4."""
    return {
        **start_params,
        "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 500, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }


def tune_lgbm(
    start_params: dict[str, Any], dtrain: lgb.Dataset, dvalid: lgb.Dataset,
    X_valid: pd.DataFrame, y_valid: pd.Series,
) -> tuple[dict[str, Any], optuna.Study]:
    """Optuna TPE search on validation average precision; returns the best params and the study.

    Trials are stored in SQLite so an interrupted run resumes where it stopped.
    """

    def objective(trial: optuna.Trial) -> float:
        booster, seconds = train_lgbm(tuning_params(trial, start_params), dtrain, dvalid)
        score = pr_auc(y_valid, booster.predict(X_valid, num_iteration=booster.best_iteration))
        trial.set_user_attr("best_iteration", booster.best_iteration)
        trial.set_user_attr("train_seconds", seconds)
        print(f"  trial {trial.number:>2}: pr_auc {score:.4f}, iter {booster.best_iteration}, {seconds}s", flush=True)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"lgbm_{json.dumps(start_params, sort_keys=True)}",
        storage=OPTUNA_STORAGE,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not study.trials:
        study.enqueue_trial(TUNING_START)
    print(f"  {len(done)} trials already complete", flush=True)
    study.optimize(objective, n_trials=max(N_TRIALS - len(done), 0))
    study.trials_dataframe().to_csv(OPTUNA_TRIALS_CSV, index=False)
    best = optuna.trial.FixedTrial(study.best_params)
    return tuning_params(best, start_params), study


def plot_bars(values: pd.Series, title: str, xlabel: str, name: str, fmt: str) -> Path:
    """Horizontal single-series bar chart, first value at the top, with direct value labels."""
    from fraud.eda import LEGIT_COLOUR, INK_SECONDARY, plt, save, style_axes

    fig, ax = plt.subplots(figsize=(7, 0.35 * len(values) + 1.4))
    bars = ax.barh(values.index[::-1], values.to_numpy()[::-1], color=LEGIT_COLOUR, height=0.65)
    ax.bar_label(bars, labels=[format(v, fmt) for v in values.to_numpy()[::-1]],
                 padding=3, color=INK_SECONDARY, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.15)
    return save(fig, name)


def save_final_model(
    name: str, booster: lgb.Booster, params: dict[str, Any], coded: list[str],
    result: dict[str, Any], X_valid: pd.DataFrame,
) -> None:
    """Write the native model file and its parameters."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(FINAL_MODEL_PATH, num_iteration=booster.best_iteration)
    info = {
        "model": name,
        "params": {**LGBM_BASE_PARAMS, **params},
        "best_iteration": booster.best_iteration,
        "validation_pr_auc": result["pr_auc"],
        "categorical_as_codes": coded,
        "n_features": X_valid.shape[1],
        "feature_names": list(X_valid.columns),
    }
    FINAL_PARAMS_JSON.write_text(json.dumps(info, indent=2) + "\n")


def run_models() -> dict[str, dict]:
    """Fit and compare every Phase 4 model on train, evaluate on validation, save the winner."""
    X_train, X_valid = load_features()
    train_ids, valid_ids = read_train_validation([ID_COL, TARGET])
    y_train, y_valid = train_ids[TARGET], valid_ids[TARGET]
    assert len(X_train) == len(y_train) and len(X_valid) == len(y_valid)
    assert list(X_train.columns) == list(X_valid.columns) and X_train.dtypes.equals(X_valid.dtypes)

    spw = float((y_train == 0).sum() / (y_train == 1).sum())
    high_card = [c for c, n in categorical_levels(X_train).items() if n > HIGH_CARDINALITY_LEVELS]
    print(f"scale_pos_weight {spw:.4f}; high-cardinality categoricals: {high_card}", flush=True)

    results: dict[str, dict] = {}
    candidates: dict[str, dict[str, Any]] = {}

    scores, seconds = fit_logreg_full(X_train, y_train, X_valid)
    results["logreg_full"] = {**evaluate(y_valid, scores), "train_seconds": seconds, "best_iteration": None}
    print(f"logreg_full done: {results['logreg_full']['pr_auc']:.4f}", flush=True)

    scores, seconds = fit_isolation_forest(X_train, X_valid)
    results["isolation_forest"] = {**evaluate(y_valid, scores), "train_seconds": seconds, "best_iteration": None}

    coded_train, coded_valid = to_category_codes(X_train, high_card), to_category_codes(X_valid, high_card)
    datasets = {
        "native": lgb_datasets(X_train, y_train, X_valid, y_valid),
        "coded": lgb_datasets(coded_train, y_train, coded_valid, y_valid),
    }
    variants = {
        "lgbm_default": ({}, "native"),
        "lgbm_scale_pos_weight": ({"scale_pos_weight": spw}, "native"),
        "lgbm_no_highcard_cats": ({}, "coded"),
    }
    for name, (params, data) in variants.items():
        booster, seconds = train_lgbm(params, *datasets[data])
        X_eval = X_valid if data == "native" else coded_valid
        results[name], scores = lgbm_result(booster, seconds, X_eval, y_valid)
        candidates[name] = {"booster": booster, "params": params, "data": data, "scores": scores}
        print(f"{name} done: {results[name]['pr_auc']:.4f} at {booster.best_iteration}", flush=True)

    start = max(variants, key=lambda n: results[n]["pr_auc"])
    start_params, start_data = variants[start]
    X_eval = X_valid if start_data == "native" else coded_valid
    print(f"Tuning from {start} ({N_TRIALS} trials)", flush=True)
    params, study = tune_lgbm(start_params, *datasets[start_data], X_eval, y_valid)
    booster, seconds = train_lgbm(params, *datasets[start_data])
    results["lgbm_tuned"], scores = lgbm_result(booster, seconds, X_eval, y_valid)
    assert np.isclose(results["lgbm_tuned"]["pr_auc"], study.best_value), "tuned refit does not match best trial"
    results["lgbm_tuned"]["tuned_from"] = start
    results["lgbm_tuned"]["n_trials"] = len(study.trials)
    results["lgbm_tuned"]["tuning_seconds"] = round(
        sum(t.user_attrs.get("train_seconds", 0.0) for t in study.trials), 1
    )
    candidates["lgbm_tuned"] = {"booster": booster, "params": params, "data": start_data, "scores": scores}

    for name, result in results.items():
        if result["pr_auc"] > 0.95:
            raise RuntimeError(f"{name} validation PR-AUC {result['pr_auc']:.4f} > 0.95: audit for leakage")
    update_metrics(results)

    best = max(results, key=lambda n: results[n]["pr_auc"])
    if best not in candidates:
        raise RuntimeError(f"Best model {best} is not LightGBM; final model format assumes LightGBM")
    final = candidates[best]
    coded = high_card if final["data"] == "coded" else []
    X_final = X_valid if not coded else coded_valid
    save_final_model(best, final["booster"], final["params"], coded, results[best], X_final)
    pd.DataFrame(
        {ID_COL: valid_ids[ID_COL].to_numpy(), "label": y_valid.to_numpy(), "score": final["scores"]}
    ).to_parquet(VALID_SCORES_PATH, index=False)

    verify_saved_model(X_final, y_valid, results[best]["pr_auc"])
    plot_comparison()
    importance = pd.Series(
        final["booster"].feature_importance("gain", iteration=final["booster"].best_iteration),
        index=final["booster"].feature_name(),
    ).nlargest(25)
    plot_bars(importance, f"Top 25 features by gain ({best})", "total gain", "feature_importance.png", ",.0f")

    print_table(results)
    print(f"Final model: {best}. Wrote {FINAL_MODEL_PATH}, {FINAL_PARAMS_JSON}, {VALID_SCORES_PATH}")
    return results


def verify_saved_model(X_valid: pd.DataFrame, y_valid: pd.Series, expected: float) -> None:
    """Reload the saved model and confirm it reproduces the recorded validation PR-AUC."""
    reloaded = lgb.Booster(model_file=FINAL_MODEL_PATH)
    observed = pr_auc(y_valid, reloaded.predict(X_valid))
    assert abs(observed - expected) < 1e-9, f"reloaded PR-AUC {observed} != recorded {expected}"
    print(f"Reloaded model PR-AUC {observed:.6f} matches recorded value")


def plot_comparison() -> Path:
    """Bar chart of validation PR-AUC for every model in metrics.json, best first."""
    metrics = json.loads(METRICS_JSON.read_text())
    keys = [
        "baseline_prevalence", "baseline_logreg", "logreg_full", "isolation_forest",
        "lgbm_default", "lgbm_scale_pos_weight", "lgbm_no_highcard_cats", "lgbm_tuned",
    ]
    values = pd.Series({k: metrics[k]["pr_auc"] for k in keys if k in metrics}).sort_values(ascending=False)
    return plot_bars(values, "Validation PR-AUC by model", "PR-AUC (average precision)", "model_comparison.png", ".3f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--engineered-check", action="store_true", help="run the Phase 3 leakage sanity check")
    group.add_argument("--models", action="store_true", help="run the Phase 4 model comparison")
    args = parser.parse_args()
    if args.engineered_check:
        run_engineered_check()
    elif args.models:
        run_models()
    else:
        run_baselines()
