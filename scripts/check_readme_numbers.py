"""Check that every number in README.md comes from a file in reports/, the Makefile or the code.

Numbers inside fenced code blocks are skipped: those blocks hold shell commands and an
example request with invented values. Digits that are part of an identifier (C13, card1,
SHA-256) are not numbers. Run from the repository root: python scripts/check_readme_numbers.py
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud.config import COSTS, RANDOM_SEED, SENSITIVITY_FEES, SPLIT_FRACTIONS  # noqa: E402
from fraud.explain import BUCKET_DAYS, SHAP_SAMPLE_ROWS, TOP_N  # noqa: E402
from fraud.metrics import REPORTED_PRECISIONS  # noqa: E402
from fraud.train import N_TRIALS  # noqa: E402

REPORTS = ROOT / "reports"
NUMBER = re.compile(r"(?<![\w.-])\d[\d,]*(?:\.\d+)?(?:e-?\d+)?%?(?![\w-])")

# Numbers that are not results, with the reason they are allowed.
EXCEPTIONS = {
    "1": "list numbering in Findings",
    "2": "list numbering in Findings",
    "3": "list numbering in Findings",
    "12": "length of the model version hash (app/main.py, hexdigest()[:12])",
    "422": "HTTP status code for invalid input (also the output column count in feature_report.md)",
}


def leaves(obj: object):
    """Every numeric leaf of a JSON object."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from leaves(v)
    elif isinstance(obj, int | float) and not isinstance(obj, bool):
        yield float(obj)


def formats(v: float) -> set[str]:
    """Every way the README may print a source value."""
    out = {f"{v:.4f}", f"{v:.3f}", f"{v:.2f}", f"{v:,.2f}", f"{v:,.1f}", f"{v:.3g}", f"{v:g}",
           f"{v:.1%}", f"{v:.2%}"}
    if v.is_integer():
        out |= {f"{int(v):,}", str(int(v))}
    return out


def source_values() -> dict[str, set[str]]:
    """Formatted values from reports/, the Makefile, .python-version and code constants."""
    values: dict[str, set[str]] = {}

    def add(name: str, numbers) -> None:
        for v in numbers:
            for f in formats(float(v)):
                values.setdefault(f, set()).add(name)

    for path in REPORTS.glob("*.json"):
        add(path.name, leaves(json.loads(path.read_text())))
    for path in REPORTS.glob("*.csv"):
        frame = pd.read_csv(path).select_dtypes("number")
        add(path.name, [v for v in frame.to_numpy().ravel() if pd.notna(v)])
    for path in [*REPORTS.glob("*.md"), ROOT / "Makefile", ROOT / ".python-version"]:
        for token in NUMBER.findall(path.read_text()):
            values.setdefault(token, set()).add(path.name)

    metrics = json.loads((REPORTS / "metrics.json").read_text())
    final = json.loads((REPORTS / "final_model_params.json").read_text())["model"]
    add("derived: validation minus test PR-AUC", [metrics[final]["pr_auc"] - metrics["final_test"]["pr_auc"]])
    cumulative = [SPLIT_FRACTIONS[0], SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1]]
    add("config.SPLIT_FRACTIONS (as percentages)", [round(f * 100, 6) for f in [*SPLIT_FRACTIONS, *cumulative]])
    add("config", [RANDOM_SEED, COSTS["review_fee"], *SENSITIVITY_FEES])
    add("code constants", [SHAP_SAMPLE_ROWS, TOP_N, BUCKET_DAYS, N_TRIALS, *REPORTED_PRECISIONS])
    return values


def readme_numbers(text: str) -> list[str]:
    """Numbers in README prose and tables, skipping fenced code blocks."""
    prose = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    return [t.rstrip(",") for t in NUMBER.findall(prose)]


def main() -> int:
    values = source_values()
    tokens = readme_numbers((ROOT / "README.md").read_text())
    unmatched = sorted({t for t in tokens if t not in values and t not in EXCEPTIONS})
    excepted = sorted({t for t in tokens if t not in values and t in EXCEPTIONS})
    print(f"{len(tokens)} numbers in README.md prose and tables, {len(set(tokens))} distinct")
    for token in excepted:
        print(f"  exception {token}: {EXCEPTIONS[token]}")
    if unmatched:
        print(f"UNMATCHED ({len(unmatched)}): {unmatched}")
        return 1
    print("All numbers matched a source.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
