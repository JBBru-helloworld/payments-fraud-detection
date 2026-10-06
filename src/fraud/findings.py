"""Generate reports/findings.md. Every number is read from a file written by another script.

Sources: reports/metrics.json, reports/shap_top_features.csv, reports/error_analysis.csv,
reports/stability.csv, reports/explain_summary.json and reports/local_examples.json.
"""

import json
from pathlib import Path

import pandas as pd

from fraud.config import REPORTS_DIR

FINDINGS_MD: Path = REPORTS_DIR / "findings.md"
ANONYMISED_GROUPS = ("C", "D", "M", "V", "identity")


def pct(x: float) -> str:
    """Percentage with one decimal place."""
    return f"{x:.1%}"


def load() -> dict:
    """Read every source file."""
    final_model = json.loads((REPORTS_DIR / "final_model_params.json").read_text())["model"]
    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text())
    return {
        "valid": metrics[final_model],
        "test": metrics["final_test"],
        "shap": pd.read_csv(REPORTS_DIR / "shap_top_features.csv"),
        "errors": pd.read_csv(REPORTS_DIR / "error_analysis.csv"),
        "stability": pd.read_csv(REPORTS_DIR / "stability.csv"),
        "summary": json.loads((REPORTS_DIR / "explain_summary.json").read_text()),
        "split": json.loads((REPORTS_DIR / "split_info.json").read_text()),
        "local": json.loads((REPORTS_DIR / "local_examples.json").read_text()),
    }


def drivers(src: dict) -> str:
    """Finding on the strongest drivers of the predictions."""
    shap = src["shap"]
    top5 = ", ".join(f"{r.feature} ({r.mean_abs_shap:.3f})" for r in shap.head(5).itertuples())
    counts = shap["group"].value_counts()
    anonymised = int(counts.reindex(ANONYMISED_GROUPS).fillna(0).sum())
    engineered = shap.loc[shap["group"] == "engineered", "feature"].tolist()
    card_features = [f for f in shap["feature"] if f.startswith(("card", "freq_card"))]
    both = [f for f in engineered if f in card_features]
    is_anonymised = shap["group"].isin(ANONYMISED_GROUPS)
    neither = [
        f for f, anon in zip(shap["feature"], is_anonymised)
        if not anon and f not in engineered and f not in card_features
    ]
    return (
        "## 1. The strongest drivers are a mix of anonymised columns and card-level features\n\n"
        f"On 5,000 validation rows, the five features with the largest mean absolute SHAP value "
        f"(log-odds) are {top5}. Of the top {len(shap)}, {anonymised} are anonymised columns "
        f"(C, D, M, V or id groups, named here only). The engineered and card-based groups overlap: "
        f"{len(engineered)} are features engineered in Phase 3 ({', '.join(engineered)}) and "
        f"{len(card_features)} are card columns or card-level aggregates ({', '.join(card_features)}), "
        f"with {len(both)} features in both groups ({', '.join(both)}). The remaining {len(neither)} are "
        f"neither anonymised, engineered nor card-based ({', '.join(neither)}).\n\n"
        "Hypothesis: the weight on card-level aggregates and frequencies suggests the model partly "
        "recognises card-level patterns seen in the training window. This would help on cards that recur "
        "and may help less on new cards. It is not tested here.\n"
        "Source: `reports/shap_top_features.csv`, `reports/figures/shap_bar.png`, "
        "`reports/figures/shap_beeswarm.png`.\n"
    )


def failures(src: dict) -> str:
    """Finding on the largest error-rate gaps."""
    errors = src["errors"]
    product = errors[errors["dimension"] == "ProductCD"].set_index("group")
    identity = errors[errors["dimension"] == "identity"].set_index("group")
    worst, best = product["fnr"].idxmax(), product["fnr"].idxmin()
    fpr_worst = product["fpr"].idxmax()
    w, b, f = product.loc[worst], product.loc[best], product.loc[fpr_worst]
    absent, present = identity.loc["absent"], identity.loc["present"]
    return (
        "## 2. Missed fraud concentrates in transactions without identity data and in one product code\n\n"
        f"At the chosen threshold on validation, ProductCD {worst} has the highest false negative rate: "
        f"{pct(w.fnr)} of its {int(w.frauds):,} frauds are missed ({int(w.false_negatives):,}), against "
        f"{pct(b.fnr)} for ProductCD {best} ({int(b.false_negatives):,} of {int(b.frauds):,}). Transactions "
        f"without identity data have a false negative rate of {pct(absent.fnr)} ({int(absent.false_negatives):,} "
        f"of {int(absent.frauds):,} frauds) against {pct(present.fnr)} with identity data "
        f"({int(present.false_negatives):,} of {int(present.frauds):,}).\n\n"
        f"False alarms run the other way: the false positive rate is {pct(present.fpr)} with identity data and "
        f"{pct(absent.fpr)} without, and ProductCD {fpr_worst} has the highest false positive rate at "
        f"{pct(f.fpr)} ({int(f.false_positives):,} of {int(f.legit):,} legitimate transactions).\n\n"
        f"Hypothesis: ProductCD {worst} and the rows without identity data may overlap heavily, so these "
        "two gaps may be one gap seen twice. The overlap is not measured here.\n"
        "Source: `reports/error_analysis.csv`, `reports/figures/error_analysis.png`.\n"
    )


def over_time(src: dict) -> str:
    """Finding on stability over time and the validation-to-test gap."""
    st = src["stability"]
    valid, test = st[st["window"] == "validation"], st[st["window"] == "test"]
    first, last = valid.iloc[0], valid.iloc[-1]
    ci = src["test"]["pr_auc_ci95"]
    return (
        "## 3. PR-AUC is lower on test than on validation and does not decline steadily within the test window\n\n"
        f"Validation PR-AUC is {first.pr_auc:.3f} in the first bucket (days {int(first.day_start)} to "
        f"{int(first.day_end)}) and {last.pr_auc:.3f} in the last (days {int(last.day_start)} to "
        f"{int(last.day_end)}). Test buckets range from {test.pr_auc.min():.3f} to {test.pr_auc.max():.3f}. "
        f"Overall PR-AUC is {src['valid']['pr_auc']:.4f} on validation and {src['test']['pr_auc']:.4f} on "
        f"test, and the test 95% bootstrap interval ({ci['low']:.4f} to {ci['high']:.4f}, "
        f"{ci['resamples']} resamples) does not include the validation value.\n\n"
        f"Test buckets do not decline steadily: in time order they score "
        f"{', '.join(f'{v:.3f}' for v in test.pr_auc)}. The bucket fraud rate ranges from "
        f"{pct(st.fraud_rate.min())} to {pct(st.fraud_rate.max())}. The overall fraud rate is "
        f"{src['split']['validation']['fraud_rate']:.2%} on validation and {src['test']['fraud_rate']:.2%} on "
        "test, so a lower overall fraud rate does not account for the lower test PR-AUC.\n\n"
        "Hypothesis: two effects may contribute. First, the fraud patterns "
        "may shift over time. Second, the validation score is optimistic because validation was used for "
        "early stopping, model choice and tuning. Because the test buckets show no steady decline, the data "
        "does not clearly favour drift over validation optimism, and the two are not separated here.\n"
        "Source: `reports/stability.csv`, `reports/figures/stability.png`, `reports/metrics.json`, `reports/split_info.json`.\n"
    )


def operating_point(src: dict) -> str:
    """Finding on the cost of the chosen policy on test."""
    t = src["test"]
    at = t["at_threshold"]
    return (
        "## 4. The cost-based threshold beats both reference policies on test\n\n"
        f"With a review fee of {t['review_fee']:g}, the threshold chosen on validation costs "
        f"{at['cost']:,.2f} on test, against {t['flag_nothing']['cost']:,.2f} for flagging nothing and "
        f"{t['flag_everything']['cost']:,.2f} for flagging everything. It catches {pct(at['recall'])} of test "
        f"frauds at a precision of {pct(at['precision'])}. These costs rest on illustrative assumptions.\n"
        "Source: `reports/metrics.json` (`final_test`), `reports/cost_analysis.md`.\n"
    )


def limitations(src: dict) -> str:
    """Limitations paragraph, including the flag rate and precision at the threshold."""
    s = src["summary"]
    v, t = s["validation"], s["test"]
    base = src["local"]["caught_fraud"]["base_value_log_odds"]
    return (
        "## Limitations\n\n"
        f"The threshold ({s['threshold']:.3g}) is on the raw model score scale. The scores are not calibrated "
        f"probabilities: the SHAP base value is {base:.2f} in log-odds, so most scores sit very close to zero. "
        f"At this threshold the model flags {pct(v['flag_rate'])} of validation transactions "
        f"({v['flagged']:,} of {v['rows']:,}) at a precision of {pct(v['precision'])}, and "
        f"{pct(t['flag_rate'])} of test transactions ({t['flagged']:,} of {t['rows']:,}) at a precision of "
        f"{pct(t['precision'])}. Most flagged transactions are therefore legitimate, which the review-fee "
        "assumption accepts but a real review team may not. The costs are illustrative. Validation was reused "
        "for early stopping, tuning and threshold choice, and there is a single chronological split. SHAP "
        "values come from a 5,000-row sample. They describe how the model uses each feature, not why fraud "
        "happens. Anonymised columns cannot be interpreted.\n"
    )


def write(path: Path = FINDINGS_MD) -> Path:
    """Assemble and write findings.md."""
    src = load()
    sections = [
        "# Findings\n\nGenerated by `python -m fraud.explain`. Every number is read from a file produced by a "
        "script in this repository. Statements marked as hypotheses are not tested here.\n",
        drivers(src), failures(src), over_time(src), operating_point(src), limitations(src),
    ]
    path.write_text("\n".join(sections))
    return path


if __name__ == "__main__":
    print(f"Wrote {write()}")
