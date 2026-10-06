"""Exploratory data analysis: figures in reports/figures/ and a summary in reports/eda_summary.md.

Descriptive only. No statistic computed here is used as a model input.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraud.config import FIGURES_DIR, MERGED_PARQUET, REPORTS_DIR, SPLIT_FRACTIONS
from fraud.data import ID_COL, TARGET, TIME_COL, column_group, column_groups

SECONDS_PER_DAY = 86_400
MIN_EMAIL_COUNT = 500
TOP_EMAIL_DOMAINS = 15
HIGH_MISSING = 0.90
CATEGORICAL_COLS = ("ProductCD", "card4", "card6", "DeviceType")

# Reference palette (light mode): categorical slots 1 and 2, plus chart chrome.
LEGIT_COLOUR = "#2a78d6"
FRAUD_COLOUR = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame of already-formatted values as a Markdown table."""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def pct(x: float, digits: int = 2) -> str:
    """Format a share as a percentage string."""
    return f"{x:.{digits}%}"


def fraud_rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Row count and fraud rate per value of col, with missing values labelled."""
    keys = df[col].astype("object").where(df[col].notna(), "missing")
    out = df.groupby(keys)[TARGET].agg(rows="size", fraud_rate="mean")
    return out.sort_values("rows", ascending=False)


def has_identity(df: pd.DataFrame, identity_cols: list[str]) -> pd.Series:
    """True where any identity column is populated, which marks a matched identity row."""
    return df[identity_cols].notna().any(axis=1)


def day_index(df: pd.DataFrame) -> pd.Series:
    """Whole days since the first transaction. The calendar reference date is unknown."""
    days = df[TIME_COL] // SECONDS_PER_DAY
    return (days - days.min()).astype(int)


def chronological_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Preview the 70/15/15 split by position after sorting on TransactionDT."""
    ordered = df.sort_values(TIME_COL, kind="stable")
    cuts = np.cumsum([0, *SPLIT_FRACTIONS])
    bounds = np.rint(cuts * len(ordered)).astype(int)
    days = day_index(ordered)
    rows = []
    for name, lo, hi in zip(("train", "validation", "test"), bounds[:-1], bounds[1:]):
        window = ordered.iloc[lo:hi]
        rows.append(
            {
                "window": name,
                "rows": len(window),
                "frauds": int(window[TARGET].sum()),
                "fraud_rate": window[TARGET].mean(),
                "first_day": int(days.iloc[lo]),
                "last_day": int(days.iloc[hi - 1]),
                "dt_min": int(window[TIME_COL].min()),
                "dt_max": int(window[TIME_COL].max()),
            }
        )
    return pd.DataFrame(rows)


def style_axes(ax: plt.Axes) -> None:
    """Apply recessive grid and axis styling."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def save(fig: plt.Figure, name: str) -> Path:
    """Write a figure to the figures directory and close it."""
    path = FIGURES_DIR / name
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_class_balance(df: pd.DataFrame) -> Path:
    """Bar chart of transaction counts per class."""
    counts = df[TARGET].value_counts().reindex([0, 1])
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(["Legitimate", "Fraud"], counts.values, color=[LEGIT_COLOUR, FRAUD_COLOUR], width=0.6)
    ax.bar_label(bars, labels=[f"{v:,}" for v in counts.values], padding=3, color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Transactions")
    ax.set_title("Class balance", loc="left")
    style_axes(ax)
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    return save(fig, "class_balance.png")


def plot_daily_fraud_rate(daily: pd.DataFrame, windows: pd.DataFrame) -> Path:
    """Daily fraud rate with a 7-day rolling mean and the preview split boundaries."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(daily.index, daily["fraud_rate"], color=LEGIT_COLOUR, alpha=0.35, linewidth=1, label="Daily")
    ax.plot(daily.index, daily["rolling_7d"], color=LEGIT_COLOUR, linewidth=2, label="7-day rolling mean")
    for day in windows["first_day"].iloc[1:]:
        ax.axvline(day, color=MUTED, linestyle="--", linewidth=1)
    for _, w in windows.iterrows():
        ax.text((w["first_day"] + w["last_day"]) / 2, 1.0, w["window"], transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", color=INK_SECONDARY, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Days since first transaction")
    ax.set_ylabel("Fraud rate")
    ax.set_title("Daily fraud rate", loc="left", pad=18)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper right")
    style_axes(ax)
    return save(fig, "fraud_rate_daily.png")


def plot_amount_by_class(df: pd.DataFrame) -> Path:
    """Density of TransactionAmt per class on a log-scaled x axis."""
    amounts = df["TransactionAmt"].astype(float)
    bins = np.logspace(np.log10(amounts[amounts > 0].min()), np.log10(amounts.max()), 80)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for label, value, colour in (("Legitimate", 0, LEGIT_COLOUR), ("Fraud", 1, FRAUD_COLOUR)):
        ax.hist(amounts[df[TARGET] == value], bins=bins, density=True, histtype="step",
                linewidth=2, color=colour, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("TransactionAmt (log scale)")
    ax.set_ylabel("Density")
    ax.set_title("Transaction amount by class", loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    style_axes(ax)
    return save(fig, "amount_by_class.png")


def plot_horizontal_rates(series: pd.Series, title: str, xlabel: str, name: str, as_pct: bool = True) -> Path:
    """Horizontal bar chart of one value per category, largest category at the top."""
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(series) + 1.4))
    bars = ax.barh(series.index.astype(str)[::-1], series.values[::-1], color=LEGIT_COLOUR, height=0.65)
    labels = [pct(v, 1) if as_pct else f"{v:,.0f}" for v in series.values[::-1]]
    ax.bar_label(bars, labels=labels, padding=3, color=INK_SECONDARY, fontsize=8)
    if as_pct:
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.15)
    return save(fig, name)


def plot_categorical_rates(tables: dict[str, pd.DataFrame]) -> Path:
    """Small multiples of fraud rate per category for each categorical column."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for ax, (col, table) in zip(axes.flat, tables.items()):
        bars = ax.barh(table.index.astype(str)[::-1], table["fraud_rate"].values[::-1], color=LEGIT_COLOUR, height=0.65)
        ax.bar_label(bars, labels=[pct(v, 1) for v in table["fraud_rate"].values[::-1]],
                     padding=3, color=INK_SECONDARY, fontsize=8)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(col, loc="left", fontsize=10)
        style_axes(ax)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.2)
    fig.suptitle("Fraud rate by category (rows sorted by count)", x=0.01, ha="left", color=INK)
    return save(fig, "fraud_rate_by_category.png")


def audit(df: pd.DataFrame, groups: dict[str, list[str]]) -> dict:
    """Missingness per group, constant and mostly-missing columns, ID and time checks."""
    missing = df.isna().mean()
    unique = df.nunique(dropna=True)
    feature_cols = [c for cols in groups.values() for c in cols]
    group_missing = pd.DataFrame(
        {
            "group": list(groups),
            "columns": [len(cols) for cols in groups.values()],
            "mean_missing": [missing[cols].mean() for cols in groups.values()],
            "fully_populated": [int((missing[cols] == 0).sum()) for cols in groups.values()],
            "over_90pct_missing": [int((missing[cols] > HIGH_MISSING).sum()) for cols in groups.values()],
        }
    )
    dt = df[TIME_COL]
    return {
        "group_missing": group_missing,
        "constant": [c for c in feature_cols if unique[c] <= 1],
        "high_missing": missing[feature_cols][missing[feature_cols] > HIGH_MISSING].sort_values(ascending=False),
        "id_unique": bool(df[ID_COL].is_unique),
        "dt_min": int(dt.min()),
        "dt_max": int(dt.max()),
        "dt_span_days": (dt.max() - dt.min()) / SECONDS_PER_DAY,
        "dt_sorted": bool(dt.is_monotonic_increasing),
    }


def write_summary(sections: list[str], path: Path) -> Path:
    """Write the Markdown summary."""
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return path


def run() -> None:
    """Run the full EDA and write figures and the summary report."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(MERGED_PARQUET)
    groups = column_groups(df.columns)
    checks = audit(df, groups)
    n_rows, n_cols = df.shape
    fraud_rate = df[TARGET].mean()
    n_fraud = int(df[TARGET].sum())

    identity_mask = has_identity(df, groups["identity"])
    identity_table = (
        df.groupby(identity_mask.map({True: "with identity", False: "without identity"}))[TARGET]
        .agg(rows="size", fraud_rate="mean")
        .reindex(["with identity", "without identity"])
    )

    days = day_index(df)
    daily = df.groupby(days)[TARGET].agg(rows="size", fraud_rate="mean")
    daily["rolling_7d"] = daily["fraud_rate"].rolling(7, min_periods=1).mean()
    windows = chronological_windows(df)

    cat_tables = {col: fraud_rate_by(df, col) for col in CATEGORICAL_COLS}
    email = fraud_rate_by(df, "P_emaildomain").drop(index="missing", errors="ignore")
    email = email[email["rows"] >= MIN_EMAIL_COUNT].head(TOP_EMAIL_DOMAINS)

    figures = [
        plot_class_balance(df),
        plot_daily_fraud_rate(daily, windows),
        plot_amount_by_class(df),
        plot_horizontal_rates(
            checks["group_missing"].set_index("group")["mean_missing"],
            "Mean missing share by column group", "Mean share of missing values", "missingness_by_group.png",
        ),
        plot_horizontal_rates(
            identity_table["fraud_rate"], "Fraud rate with and without identity data", "Fraud rate",
            "fraud_rate_by_identity.png",
        ),
        plot_categorical_rates(cat_tables),
        plot_horizontal_rates(
            email["fraud_rate"], f"Fraud rate for the {len(email)} most frequent P_emaildomain values",
            "Fraud rate", "fraud_rate_by_email_domain.png",
        ),
    ]

    amt = df.groupby(TARGET)["TransactionAmt"].describe(percentiles=[0.25, 0.5, 0.75])
    amt_table = pd.DataFrame(
        {
            "class": ["legitimate", "fraud"],
            "rows": [f"{int(amt.loc[k, 'count']):,}" for k in (0, 1)],
            "median": [f"{amt.loc[k, '50%']:.2f}" for k in (0, 1)],
            "mean": [f"{amt.loc[k, 'mean']:.2f}" for k in (0, 1)],
            "p25": [f"{amt.loc[k, '25%']:.2f}" for k in (0, 1)],
            "p75": [f"{amt.loc[k, '75%']:.2f}" for k in (0, 1)],
            "max": [f"{amt.loc[k, 'max']:.2f}" for k in (0, 1)],
        }
    )

    def rate_table(table: pd.DataFrame, label: str) -> str:
        return markdown_table(
            pd.DataFrame(
                {
                    label: table.index.astype(str),
                    "rows": [f"{v:,}" for v in table["rows"]],
                    "fraud rate": [pct(v) for v in table["fraud_rate"]],
                }
            )
        )

    gm = checks["group_missing"]
    hm = checks["high_missing"]
    peak_day, low_day = daily["fraud_rate"].idxmax(), daily["fraud_rate"].idxmin()
    sections = [
        "# EDA summary\n\nGenerated by `python -m fraud.eda` from `data/interim/train_merged.parquet`. "
        "Every number below is computed by the script. Anonymised columns (C, D, M, V, id) are described by name only.",
        "## Data audit\n\n"
        f"- The merged table has {n_rows:,} rows and {n_cols} columns.\n"
        f"- TransactionID unique: {checks['id_unique']}.\n"
        f"- TransactionDT runs from {checks['dt_min']:,} to {checks['dt_max']:,} seconds, "
        f"spanning {checks['dt_span_days']:.1f} days. Rows are sorted by TransactionDT: {checks['dt_sorted']}.\n"
        f"- Constant columns: {', '.join(checks['constant']) or 'none'}.\n"
        f"- Columns more than 90% missing: {len(hm)}.",
        "### Missingness by column group\n\n"
        + markdown_table(
            gm.assign(mean_missing=gm["mean_missing"].map(pct)).rename(
                columns={"mean_missing": "mean missing", "fully_populated": "fully populated",
                         "over_90pct_missing": "over 90% missing"}
            )
        )
        + "\n\n![Missingness by group](figures/missingness_by_group.png)",
        "### Columns more than 90% missing\n\n"
        + markdown_table(pd.DataFrame({"column": hm.index, "missing": [pct(v) for v in hm.values]})),
        "## 1. Class balance\n\n"
        f"{n_fraud:,} of {n_rows:,} transactions are fraud, a fraud rate of {pct(fraud_rate)}.\n\n"
        "![Class balance](figures/class_balance.png)",
        "## 2. Fraud rate over time\n\n"
        f"Daily buckets run from day 0 to day {int(daily.index.max())} ({len(daily)} days). "
        f"The daily fraud rate ranges from {pct(daily['fraud_rate'].min())} (day {low_day}) "
        f"to {pct(daily['fraud_rate'].max())} (day {peak_day}), with a median of {pct(daily['fraud_rate'].median())}. "
        f"Daily row counts range from {int(daily['rows'].min()):,} to {int(daily['rows'].max()):,}. "
        "Days are relative to the first transaction because the reference date is unknown.\n\n"
        "![Daily fraud rate](figures/fraud_rate_daily.png)",
        "## 3. Transaction amount by class\n\n"
        + markdown_table(amt_table)
        + "\n\n![Amount by class](figures/amount_by_class.png)",
        "## 4. Identity data\n\n"
        f"{pct(identity_mask.mean())} of rows have identity data.\n\n"
        + rate_table(identity_table, "identity")
        + "\n\n![Fraud rate by identity](figures/fraud_rate_by_identity.png)",
        "## 5. Fraud rate by category\n\n"
        + "\n\n".join(f"### {col}\n\n" + rate_table(t, col) for col, t in cat_tables.items())
        + "\n\n![Fraud rate by category](figures/fraud_rate_by_category.png)",
        f"## 6. Top {len(email)} P_emaildomain values (at least {MIN_EMAIL_COUNT} rows)\n\n"
        + rate_table(email, "P_emaildomain")
        + "\n\n![Fraud rate by email domain](figures/fraud_rate_by_email_domain.png)",
        "## 7. Chronological split preview\n\n"
        "Rows sorted by TransactionDT and cut by position at "
        f"{', '.join(pct(f, 0) for f in SPLIT_FRACTIONS)}. No model is trained here.\n\n"
        + markdown_table(
            windows.assign(
                rows=windows["rows"].map("{:,}".format),
                frauds=windows["frauds"].map("{:,}".format),
                fraud_rate=windows["fraud_rate"].map(pct),
                dt_min=windows["dt_min"].map("{:,}".format),
                dt_max=windows["dt_max"].map("{:,}".format),
            ).rename(columns={"fraud_rate": "fraud rate", "first_day": "first day", "last_day": "last day",
                              "dt_min": "TransactionDT min", "dt_max": "TransactionDT max"})
        ),
    ]
    summary = write_summary(sections, REPORTS_DIR / "eda_summary.md")

    for path in figures:
        print(f"Wrote {path.relative_to(REPORTS_DIR.parent)}")
    print(f"Wrote {summary.relative_to(REPORTS_DIR.parent)}")


if __name__ == "__main__":
    run()
