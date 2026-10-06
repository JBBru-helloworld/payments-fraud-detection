"""Explainability, error analysis and stability over time. Descriptive only.

Nothing computed here changes the model, the features or the threshold.
SHAP values are on the model's raw (log-odds) scale. Anonymised columns
(C, D, M, V, id) are referred to by name and group only.
Test scores are read for the stability description only.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shap

from fraud.config import INTERIM_DIR, MERGED_PARQUET, RANDOM_SEED, REPORTS_DIR
from fraud.data import ID_COL, TIME_COL, column_group, column_groups
from fraud.eda import (
    FRAUD_COLOUR,
    INK_SECONDARY,
    LEGIT_COLOUR,
    MUTED,
    has_identity,
    plt,
    save,
    style_axes,
)
from fraud.features import ENGINEERED_CATEGORICAL, ENGINEERED_NUMERIC, FEATURE_PATHS
from fraud.metrics import pr_auc
from fraud.split import SPLIT_INFO_JSON, read_train_validation
from fraud.threshold import THRESHOLD_JSON, clean_amounts
from fraud.train import FINAL_MODEL_PATH, FINAL_PARAMS_JSON, to_category_codes

SHAP_SAMPLE_ROWS = 5_000
TOP_N = 20
BUCKET_DAYS = 7
MIN_BUCKET_FRAUDS = 300
SECONDS_PER_DAY = 86_400

SHAP_TOP_CSV: Path = REPORTS_DIR / "shap_top_features.csv"
LOCAL_EXAMPLES_JSON: Path = REPORTS_DIR / "local_examples.json"
ERROR_ANALYSIS_CSV: Path = REPORTS_DIR / "error_analysis.csv"
STABILITY_CSV: Path = REPORTS_DIR / "stability.csv"
SUMMARY_JSON: Path = REPORTS_DIR / "explain_summary.json"
FINDINGS_MD: Path = REPORTS_DIR / "findings.md"
VALID_SCORES_PATH: Path = INTERIM_DIR / "valid_scores.parquet"
TEST_SCORES_PATH: Path = INTERIM_DIR / "test_scores.parquet"
ENGINEERED = set(ENGINEERED_NUMERIC) | set(ENGINEERED_CATEGORICAL)


# Pure helpers (unit tested)


def feature_group(name: str) -> str:
    """Column group for a feature name; engineered features get their own group."""
    return "engineered" if name in ENGINEERED else (column_group(name) or "other")


def error_rates(df: pd.DataFrame, dimension: str, group_col: str) -> pd.DataFrame:
    """False negative and false positive rates per group, with counts.

    Expects boolean columns 'label' (fraud) and 'flagged'. FNR = missed frauds / frauds,
    FPR = flagged legitimate / legitimate; NaN when the denominator is zero.
    """
    rows = []
    for group, part in df.groupby(group_col, observed=True, sort=True, dropna=False):
        fraud, flagged = part["label"].astype(bool), part["flagged"].astype(bool)
        frauds, legit = int(fraud.sum()), int((~fraud).sum())
        fn, fp = int((fraud & ~flagged).sum()), int((~fraud & flagged).sum())
        rows.append(
            {
                "dimension": dimension,
                "group": "missing" if pd.isna(group) else str(group),
                "rows": len(part),
                "frauds": frauds,
                "legit": legit,
                "false_negatives": fn,
                "false_positives": fp,
                "fnr": fn / frauds if frauds else np.nan,
                "fpr": fp / legit if legit else np.nan,
            }
        )
    return pd.DataFrame(rows)


def time_buckets(
    dt: pd.Series, labels: pd.Series, start: float,
    days: int = BUCKET_DAYS, min_frauds: int = MIN_BUCKET_FRAUDS,
) -> np.ndarray:
    """Bucket index per row in fixed windows of `days` from `start`.

    While the final bucket has fewer than `min_frauds` frauds it is merged into the previous one.
    """
    bucket = ((dt.to_numpy(dtype=np.float64) - start) // (days * SECONDS_PER_DAY)).astype(int)
    frauds = pd.Series(labels.to_numpy()).groupby(bucket).sum()
    while len(frauds) > 1 and frauds.iloc[-1] < min_frauds:
        last, previous = frauds.index[-1], frauds.index[-2]
        bucket[bucket == last] = previous
        frauds = pd.Series(labels.to_numpy()).groupby(bucket).sum()
    return bucket


def median_example(group: pd.DataFrame) -> pd.Series:
    """Lower-median row by score, ties broken by TransactionID, so the choice is reproducible."""
    ordered = group.sort_values(["score", ID_COL], kind="stable")
    return ordered.iloc[(len(ordered) - 1) // 2]


# Data loading


def load_validation() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Validation features, a row-aligned frame of scores and descriptors, and train amount quartile edges."""
    features = pd.read_parquet(FEATURE_PATHS["validation"])
    scores = pd.read_parquet(VALID_SCORES_PATH)
    identity_cols = column_groups(pq.read_schema(MERGED_PARQUET).names)["identity"]
    train, valid = read_train_validation([ID_COL, "ProductCD", "TransactionAmt", *identity_cols])
    assert (valid[ID_COL].to_numpy() == scores[ID_COL].to_numpy()).all(), "validation rows misaligned"
    edges = np.quantile(clean_amounts(train["TransactionAmt"]), [0.25, 0.5, 0.75])
    frame = scores.assign(
        **{TIME_COL: valid[TIME_COL].to_numpy(), "ProductCD": valid["ProductCD"].to_numpy(),
           "TransactionAmt": clean_amounts(valid["TransactionAmt"]),
           "identity": np.where(has_identity(valid, identity_cols), "present", "absent")}
    )
    return features, frame, edges


def amount_band(amounts: np.ndarray, edges: np.ndarray) -> pd.Series:
    """Quartile band labels using train-split edges."""
    labels = [
        f"Q1 (<= {edges[0]:.2f})", f"Q2 ({edges[0]:.2f} to {edges[1]:.2f}]",
        f"Q3 ({edges[1]:.2f} to {edges[2]:.2f}]", f"Q4 (> {edges[2]:.2f})",
    ]
    return pd.Series(pd.cut(amounts, [-np.inf, *edges, np.inf], labels=labels))


# SHAP


def shap_explanation(explainer: shap.TreeExplainer, X: pd.DataFrame) -> shap.Explanation:
    """SHAP values with numeric display data (category codes) so plots can colour by value."""
    raw = explainer(X)
    codes = to_category_codes(X, list(X.select_dtypes("category").columns))
    return shap.Explanation(
        values=raw.values, base_values=raw.base_values,
        data=codes.to_numpy(dtype=np.float64), feature_names=list(X.columns),
    )


def global_shap(explainer: shap.TreeExplainer, features: pd.DataFrame) -> pd.DataFrame:
    """Top features by mean |SHAP| on a seeded validation sample; writes the CSV and two plots."""
    sample = features.sample(n=SHAP_SAMPLE_ROWS, random_state=RANDOM_SEED)
    exp = shap_explanation(explainer, sample)
    mean_abs = pd.Series(np.abs(exp.values).mean(axis=0), index=exp.feature_names)
    top = mean_abs.nlargest(TOP_N)
    table = pd.DataFrame(
        {"rank": range(1, TOP_N + 1), "feature": top.index,
         "group": [feature_group(f) for f in top.index], "mean_abs_shap": top.to_numpy()}
    )
    table.to_csv(SHAP_TOP_CSV, index=False)

    shap.plots.bar(exp, max_display=TOP_N + 1, show=False)
    finish_shap_figure(f"Mean |SHAP| (log-odds), {SHAP_SAMPLE_ROWS:,} validation rows", "shap_bar.png")
    shap.plots.beeswarm(exp, max_display=TOP_N + 1, show=False)
    finish_shap_figure(
        f"SHAP values (log-odds), {SHAP_SAMPLE_ROWS:,} validation rows\n"
        "categorical features coloured by category code (order not meaningful)", "shap_beeswarm.png"
    )
    return table


def finish_shap_figure(title: str, name: str) -> Path:
    """Title and save the current SHAP figure."""
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    fig.suptitle(title, x=0.02, ha="left", fontsize=11)
    return save(fig, name)


def local_examples(
    explainer: shap.TreeExplainer, features: pd.DataFrame, frame: pd.DataFrame, threshold: float
) -> dict[str, Any]:
    """Waterfall plots for the median-scoring caught fraud, missed fraud and false alarm."""
    fraud, flagged = frame["label"] == 1, frame["score"] >= threshold
    groups = {
        "caught_fraud": frame[fraud & flagged],
        "missed_fraud": frame[fraud & ~flagged],
        "false_alarm": frame[~fraud & flagged],
    }
    examples: dict[str, Any] = {"threshold": threshold, "rule": "lower-median score within group, ties by TransactionID"}
    for name, group in groups.items():
        row = median_example(group)
        position = int(np.flatnonzero(frame[ID_COL].to_numpy() == row[ID_COL])[0])
        X = features.iloc[[position]]
        raw = explainer(X)
        exp = shap.Explanation(
            values=raw.values[0], base_values=float(np.ravel(raw.base_values)[0]),
            data=X.iloc[0].astype(object).to_numpy(), feature_names=list(X.columns),
        )
        shap.plots.waterfall(exp, max_display=12, show=False)
        finish_shap_figure(f"{name.replace('_', ' ')}: TransactionID {int(row[ID_COL])}, "
                           f"score {row['score']:.3g}\nSHAP contributions in log-odds", f"waterfall_{name}.png")
        examples[name] = {
            ID_COL: int(row[ID_COL]), "score": float(row["score"]), "label": int(row["label"]),
            "group_size": int(len(group)), "base_value_log_odds": float(exp.base_values),
            "top_contributions": {
                f: float(v) for f, v in sorted(zip(exp.feature_names, exp.values), key=lambda t: -abs(t[1]))[:5]
            },
        }
    LOCAL_EXAMPLES_JSON.write_text(json.dumps(examples, indent=2) + "\n")
    return examples


# Error analysis and stability


def error_analysis(frame: pd.DataFrame, edges: np.ndarray, threshold: float) -> pd.DataFrame:
    """Error rates by ProductCD, train-quartile amount band and identity presence."""
    df = frame.assign(flagged=frame["score"] >= threshold,
                      amount_band=amount_band(frame["TransactionAmt"].to_numpy(), edges).to_numpy())
    table = pd.concat(
        [error_rates(df, "ProductCD", "ProductCD"),
         error_rates(df, "amount_quartile", "amount_band"),
         error_rates(df, "identity", "identity")],
        ignore_index=True,
    )
    table.to_csv(ERROR_ANALYSIS_CSV, index=False)
    plot_error_rates(table)
    return table


def plot_error_rates(table: pd.DataFrame) -> Path:
    """Two panels (FNR, FPR) of horizontal bars per group with counts in the labels."""
    labels = [f"{d}: {g}" for d, g in zip(table["dimension"], table["group"])][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.38 * len(table) + 1.6), sharey=True)
    panels = (("fnr", "frauds", "False negative rate (share of frauds missed)", FRAUD_COLOUR),
              ("fpr", "legit", "False positive rate (share of legitimate flagged)", LEGIT_COLOUR))
    for ax, (rate, count, title, colour) in zip(axes, panels):
        values = table[rate].to_numpy()[::-1]
        bars = ax.barh(labels, np.nan_to_num(values), color=colour, height=0.65)
        texts = [f"{v:.1%} (n={n:,})" if not np.isnan(v) else f"n/a (n={n:,})"
                 for v, n in zip(values, table[count].to_numpy()[::-1])]
        ax.bar_label(bars, labels=texts, padding=3, color=INK_SECONDARY, fontsize=7)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_title(title, loc="left", fontsize=10)
        style_axes(ax)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.35)
    return save(fig, "error_analysis.png")


def stability(frame: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """PR-AUC and fraud rate per ~7-day bucket in the validation and test windows."""
    split = json.loads(SPLIT_INFO_JSON.read_text())
    day0 = split["train"]["dt_min"] // SECONDS_PER_DAY
    rows = []
    for window, df in (("validation", frame), ("test", test)):
        buckets = time_buckets(df[TIME_COL], df["label"], start=split[window]["dt_min"])
        for b, part in df.groupby(buckets, sort=True):
            rows.append(
                {
                    "window": window, "bucket": int(b),
                    "day_start": int(part[TIME_COL].min() // SECONDS_PER_DAY - day0),
                    "day_end": int(part[TIME_COL].max() // SECONDS_PER_DAY - day0),
                    "rows": len(part), "frauds": int(part["label"].sum()),
                    "fraud_rate": float(part["label"].mean()),
                    "pr_auc": pr_auc(part["label"].to_numpy(), part["score"].to_numpy()),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(STABILITY_CSV, index=False)
    plot_stability(table, split, day0)
    return table


def plot_stability(table: pd.DataFrame, split: dict, day0: int) -> Path:
    """Stacked panels on a shared day axis: PR-AUC and fraud rate per bucket, split boundaries marked."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    boundaries = {
        "train | validation": split["validation"]["dt_min"] / SECONDS_PER_DAY - day0,
        "validation | test": split["test"]["dt_min"] / SECONDS_PER_DAY - day0,
    }
    for ax, col, ylabel in ((axes[0], "pr_auc", "PR-AUC"), (axes[1], "fraud_rate", "fraud rate")):
        for window, colour in (("validation", LEGIT_COLOUR), ("test", FRAUD_COLOUR)):
            part = table[table["window"] == window]
            mid = (part["day_start"] + part["day_end"] + 1) / 2
            ax.plot(mid, part[col], color=colour, linewidth=2, marker="o", markersize=5, label=window)
        for name, day in boundaries.items():
            ax.axvline(day, color=MUTED, linewidth=1, linestyle="--")
        ax.set_ylabel(ylabel)
        style_axes(ax)
    for name, day in boundaries.items():
        axes[0].annotate(name, (day, 1.0), xycoords=("data", "axes fraction"), xytext=(-4, -10),
                         textcoords="offset points", color=INK_SECONDARY, fontsize=8, ha="right")
    axes[1].yaxis.set_major_formatter(lambda v, _: f"{v:.1%}")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    axes[1].set_xlabel(f"day since first transaction ({BUCKET_DAYS}-day buckets; reference date unknown)")
    axes[0].set_title("PR-AUC and fraud rate over time", loc="left")
    return save(fig, "stability.png")


def threshold_summary(frame: pd.DataFrame, test: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """Counts, flag rate and precision at the chosen threshold, per window."""
    summary: dict[str, Any] = {"threshold": threshold, "score_scale": "raw model score, not calibrated"}
    for window, df in (("validation", frame), ("test", test)):
        fraud, flagged = df["label"] == 1, df["score"] >= threshold
        tp, fp = int((fraud & flagged).sum()), int((~fraud & flagged).sum())
        summary[window] = {
            "rows": len(df), "tp": tp, "fp": fp,
            "fn": int((fraud & ~flagged).sum()), "tn": int((~fraud & ~flagged).sum()),
            "flagged": int(flagged.sum()), "flag_rate": float(flagged.mean()),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(int(fraud.sum()), 1),
        }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run() -> None:
    """Compute every descriptive output, then write findings.md from the saved files."""
    threshold = float(json.loads(THRESHOLD_JSON.read_text())["chosen"]["threshold"])
    features, frame, edges = load_validation()
    test = pd.read_parquet(TEST_SCORES_PATH)
    assert json.loads(FINAL_PARAMS_JSON.read_text())["categorical_as_codes"] == []

    summary = threshold_summary(frame, test, threshold)
    print(f"Flag rate: validation {summary['validation']['flag_rate']:.4%}, test {summary['test']['flag_rate']:.4%}")
    errors = error_analysis(frame, edges, threshold)
    print(errors.to_string(index=False))
    print(stability(frame, test).to_string(index=False))

    explainer = shap.TreeExplainer(lgb.Booster(model_file=FINAL_MODEL_PATH))
    print(json.dumps(local_examples(explainer, features, frame, threshold), indent=2))
    print("Computing global SHAP (slow) ...", flush=True)
    print(global_shap(explainer, features).to_string(index=False))
    write_findings()


def write_findings() -> Path:
    """Build findings.md. Defined in fraud.findings so its numbers come only from saved files."""
    from fraud.findings import write

    path = write()
    print(f"Wrote {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--findings-only", action="store_true", help="rebuild findings.md from saved outputs")
    if parser.parse_args().findings_only:
        write_findings()
    else:
        run()
