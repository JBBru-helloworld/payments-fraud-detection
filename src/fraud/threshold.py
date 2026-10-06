"""Cost-based decision threshold, chosen on the validation split only.

Rule: a transaction is flagged when its score is at or above the threshold.
Cost = sum of TransactionAmt over missed frauds + review_fee * number of false alarms.
Flagged frauds are blocked at no further cost. All assumptions are illustrative (see config.COSTS).
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud.config import COSTS, INTERIM_DIR, REPORTS_DIR, SENSITIVITY_FEES
from fraud.data import ID_COL
from fraud.split import read_train_validation

THRESHOLD_JSON: Path = REPORTS_DIR / "threshold.json"
COST_REPORT_MD: Path = REPORTS_DIR / "cost_analysis.md"
VALID_SCORES_PATH: Path = INTERIM_DIR / "valid_scores.parquet"
N_LINEAR = 1_000
N_QUANTILES = 2_000


def clean_amounts(amounts: Any) -> np.ndarray:
    """Amounts as float64 rounded to 2 dp, removing float32 storage noise."""
    return np.round(np.asarray(amounts, dtype=np.float64), 2)


def policy_cost(
    y_true: Any, scores: Any, amounts: Any, threshold: float, review_fee: float
) -> dict[str, float]:
    """Total cost, precision, recall and confusion counts for one threshold.

    Precision is 0.0 when nothing is flagged.
    """
    y = np.asarray(y_true).astype(bool)
    flagged = np.asarray(scores) >= threshold
    amt = clean_amounts(amounts)
    tp = int((flagged & y).sum())
    fp = int((flagged & ~y).sum())
    fn = int((~flagged & y).sum())
    return {
        "threshold": float(threshold),
        "cost": float(amt[~flagged & y].sum() + review_fee * fp),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def flag_nothing(y_true: Any, scores: Any, amounts: Any, review_fee: float) -> dict[str, float]:
    """Reference policy that never flags: every fraud is missed."""
    return policy_cost(y_true, scores, amounts, np.inf, review_fee)


def flag_everything(y_true: Any, scores: Any, amounts: Any, review_fee: float) -> dict[str, float]:
    """Reference policy that flags every transaction: every legitimate one is a false alarm."""
    return policy_cost(y_true, scores, amounts, -np.inf, review_fee)


def threshold_grid(scores: Any) -> np.ndarray:
    """Sorted unique union of 1,000 evenly spaced values in [0, 1] and 2,000 score quantiles."""
    linear = np.linspace(0.0, 1.0, N_LINEAR)
    quantiles = np.quantile(np.asarray(scores, dtype=np.float64), np.linspace(0.0, 1.0, N_QUANTILES))
    return np.unique(np.concatenate([linear, quantiles]))


def grid_costs(
    y_true: Any, scores: Any, amounts: Any, thresholds: np.ndarray, review_fee: float
) -> pd.DataFrame:
    """Cost, precision, recall and counts for every threshold, via sorting and cumulative sums."""
    order = np.argsort(scores, kind="stable")
    s = np.asarray(scores, dtype=np.float64)[order]
    y = np.asarray(y_true).astype(bool)[order]
    fraud_amt = np.where(y, clean_amounts(amounts)[order], 0.0)
    cum_pos = np.concatenate([[0], np.cumsum(y)])
    cum_amt = np.concatenate([[0.0], np.cumsum(fraud_amt)])

    below = np.searchsorted(s, thresholds, side="left")  # rows with score < threshold
    fn = cum_pos[below]
    tp = int(y.sum()) - fn
    fp = (len(s) - below) - tp
    flagged = tp + fp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(flagged > 0, tp / np.maximum(flagged, 1), 0.0)
        recall = tp / max(int(y.sum()), 1)
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "cost": cum_amt[below] + review_fee * fp,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    )


def best_threshold(table: pd.DataFrame) -> dict[str, float]:
    """Row with the lowest cost; ties go to the lowest threshold."""
    row = table.loc[table["cost"].idxmin()]
    return {k: (int(v) if k in ("tp", "fp", "fn") else float(v)) for k, v in row.items()}


def sensitivity(y_true: Any, scores: Any, amounts: Any, fees: tuple[float, ...]) -> list[dict]:
    """Cost-minimising threshold and reference policy costs for each review fee."""
    grid = threshold_grid(scores)
    rows = []
    for fee in fees:
        best = best_threshold(grid_costs(y_true, scores, amounts, grid, fee))
        rows.append(
            {
                "review_fee": fee,
                "threshold": best["threshold"],
                "cost": best["cost"],
                "precision": best["precision"],
                "recall": best["recall"],
                "flag_nothing_cost": flag_nothing(y_true, scores, amounts, fee)["cost"],
                "flag_everything_cost": flag_everything(y_true, scores, amounts, fee)["cost"],
            }
        )
    return rows


def load_validation_scores() -> pd.DataFrame:
    """Validation scores from Phase 4 joined to TransactionAmt by TransactionID."""
    scores = pd.read_parquet(VALID_SCORES_PATH)
    _, valid = read_train_validation([ID_COL, "TransactionAmt"])
    merged = scores.merge(valid[[ID_COL, "TransactionAmt"]], on=ID_COL, how="left", validate="one_to_one")
    assert merged["TransactionAmt"].notna().all(), "validation scores without an amount"
    return merged


def assumptions() -> dict[str, Any]:
    """The cost assumptions recorded alongside every threshold."""
    return {
        "missed_fraud_cost": "TransactionAmt of the transaction (simplified chargeback loss)",
        "false_alarm_cost": "fixed review fee per flagged legitimate transaction",
        "caught_fraud_cost": 0.0,
        "review_fee": COSTS["review_fee"],
        "sensitivity_fees": list(SENSITIVITY_FEES),
        "amounts": "float64, rounded to 2 decimal places",
        "rule": "flag when score >= threshold",
        "grid": f"{N_LINEAR} evenly spaced values in [0, 1] plus {N_QUANTILES} validation score quantiles",
        "chosen_on": "validation",
        "illustrative": True,
    }


def money(value: float) -> str:
    """Currency units with thousands separators and 2 dp."""
    return f"{value:,.2f}"


def write_cost_report(valid_rows: list[dict], test_rows: list[dict] | None = None) -> Path:
    """Markdown sensitivity table on validation, plus test costs once the final evaluation has run."""
    lines = [
        "# Cost analysis",
        "",
        "Generated by `python -m fraud.threshold` and `python -m fraud.final_eval`. Costs are in the "
        "currency units of TransactionAmt and rest on illustrative assumptions: a missed fraud costs its "
        "amount, a false alarm costs a fixed review fee, and a caught fraud costs nothing. Thresholds are "
        "chosen on validation only.",
        "",
        "## Validation",
        "",
        "| review fee | threshold | precision | recall | chosen policy cost | flag nothing | flag everything |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in valid_rows:
        lines.append(
            f"| {r['review_fee']:g} | {r['threshold']:.6g} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{money(r['cost'])} | {money(r['flag_nothing_cost'])} | {money(r['flag_everything_cost'])} |"
        )
    if test_rows is not None:
        lines += [
            "",
            "## Test (threshold for each fee taken from validation)",
            "",
            "| review fee | threshold | precision | recall | chosen policy cost | flag nothing | flag everything |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in test_rows:
            lines.append(
                f"| {r['review_fee']:g} | {r['threshold']:.6g} | {r['precision']:.4f} | {r['recall']:.4f} | "
                f"{money(r['cost'])} | {money(r['flag_nothing_cost'])} | {money(r['flag_everything_cost'])} |"
            )
    COST_REPORT_MD.write_text("\n".join(lines) + "\n")
    return COST_REPORT_MD


def plot_cost(table: pd.DataFrame, chosen: dict[str, float], fee: float) -> Path:
    """Validation cost against threshold (log scale), chosen threshold marked."""
    from fraud.eda import FRAUD_COLOUR, INK_SECONDARY, LEGIT_COLOUR, SURFACE, plt, save, style_axes

    shown = table[table["threshold"] > 0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(shown["threshold"], shown["cost"], color=LEGIT_COLOUR, linewidth=2)
    ax.axvline(chosen["threshold"], color=FRAUD_COLOUR, linewidth=1.2, linestyle="--")
    ax.scatter([chosen["threshold"]], [chosen["cost"]], s=40, color=FRAUD_COLOUR, zorder=3,
               edgecolors=SURFACE, linewidths=2)
    ax.text(0.03, 0.06, f"chosen threshold {chosen['threshold']:.3g}, cost {money(chosen['cost'])}",
            transform=ax.transAxes, color=INK_SECONDARY, fontsize=8)
    ax.set_xscale("log")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.set_xlabel("threshold (log scale)")
    ax.set_ylabel("total cost (currency units)")
    ax.set_title(f"Validation cost by threshold, review fee {fee:g}", loc="left")
    style_axes(ax)
    return save(fig, "cost_vs_threshold.png")


def plot_precision_recall(table: pd.DataFrame, chosen: dict[str, float]) -> Path:
    """Validation precision and recall against threshold (log scale), chosen threshold marked."""
    from fraud.eda import FRAUD_COLOUR, INK_SECONDARY, LEGIT_COLOUR, MUTED, plt, save, style_axes

    shown = table[table["threshold"] > 0]
    fig, ax = plt.subplots(figsize=(7, 4))
    first = shown["threshold"].iloc[0]
    for col, colour, dy in (("precision", LEGIT_COLOUR, 6), ("recall", FRAUD_COLOUR, -12)):
        ax.plot(shown["threshold"], shown[col], color=colour, linewidth=2, label=col)
        ax.annotate(col, (first, shown[col].iloc[0]), xytext=(2, dy),
                    textcoords="offset points", color=INK_SECONDARY, fontsize=8)
    ax.axvline(chosen["threshold"], color=MUTED, linewidth=1.2, linestyle="--")
    ax.annotate(f"chosen {chosen['threshold']:.3g}", (chosen["threshold"], 0.5), xytext=(-6, 0),
                textcoords="offset points", color=INK_SECONDARY, fontsize=8, ha="right")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("threshold (log scale)")
    ax.set_ylabel("validation value")
    ax.set_title("Validation precision and recall by threshold", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.62))
    style_axes(ax)
    return save(fig, "precision_recall_vs_threshold.png")


def run() -> dict[str, Any]:
    """Choose the threshold on validation, run the sensitivity analysis and write the outputs."""
    df = load_validation_scores()
    y, s, amt = df["label"].to_numpy(), df["score"].to_numpy(), df["TransactionAmt"].to_numpy()
    fee = COSTS["review_fee"]
    grid = threshold_grid(s)
    table = grid_costs(y, s, amt, grid, fee)
    chosen = best_threshold(table)
    rows = sensitivity(y, s, amt, SENSITIVITY_FEES)

    result = {
        "assumptions": assumptions(),
        "grid_size": int(len(grid)),
        "chosen": {"review_fee": fee, **chosen},
        "validation_flag_nothing": flag_nothing(y, s, amt, fee),
        "validation_flag_everything": flag_everything(y, s, amt, fee),
        "sensitivity": rows,
    }
    THRESHOLD_JSON.write_text(json.dumps(result, indent=2, default=float) + "\n")
    write_cost_report(rows)
    plot_cost(table, chosen, fee)
    plot_precision_recall(table, chosen)

    print(f"Grid size {len(grid):,}. Chosen threshold at fee {fee:g}: {chosen['threshold']:.6g}")
    print(f"{'fee':>5}{'threshold':>14}{'cost':>15}{'nothing':>15}{'everything':>15}{'precision':>11}{'recall':>9}")
    for r in rows:
        print(f"{r['review_fee']:>5g}{r['threshold']:>14.6g}{r['cost']:>15,.2f}{r['flag_nothing_cost']:>15,.2f}"
              f"{r['flag_everything_cost']:>15,.2f}{r['precision']:>11.4f}{r['recall']:>9.4f}")
    print(f"Wrote {THRESHOLD_JSON}, {COST_REPORT_MD} and two figures")
    return result


if __name__ == "__main__":
    run()
