"""Chronological train, validation and test split on TransactionDT quantiles."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraud.config import MERGED_PARQUET, REPORTS_DIR, SPLIT_FRACTIONS
from fraud.data import ID_COL, TARGET, TIME_COL

SPLIT_NAMES: tuple[str, str, str] = ("train", "validation", "test")
SPLIT_INFO_JSON: Path = REPORTS_DIR / "split_info.json"


def cut_points(
    time: pd.Series, fractions: tuple[float, float, float] = SPLIT_FRACTIONS
) -> tuple[float, float]:
    """Return the TransactionDT quantiles that close the train and validation windows."""
    first = fractions[0]
    second = fractions[0] + fractions[1]
    lo, hi = np.quantile(time.to_numpy(dtype=np.float64), [first, second])
    return float(lo), float(hi)


def split_masks(
    time: pd.Series, fractions: tuple[float, float, float] = SPLIT_FRACTIONS
) -> dict[str, np.ndarray]:
    """Boolean masks per split. Equal timestamps always fall in the same split."""
    lo, hi = cut_points(time, fractions)
    values = time.to_numpy()
    return {
        "train": values <= lo,
        "validation": (values > lo) & (values <= hi),
        "test": values > hi,
    }


def split_info(df: pd.DataFrame, masks: dict[str, np.ndarray]) -> dict[str, dict]:
    """Row count, fraud count, fraud rate and TransactionDT range for each split."""
    info: dict[str, dict] = {}
    for name in SPLIT_NAMES:
        part = df.loc[masks[name]]
        info[name] = {
            "rows": int(len(part)),
            "frauds": int(part[TARGET].sum()),
            "fraud_rate": float(part[TARGET].mean()),
            "dt_min": int(part[TIME_COL].min()),
            "dt_max": int(part[TIME_COL].max()),
        }
    return info


def read_train_validation(
    columns: list[str], path: Path = MERGED_PARQUET
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read columns for the train and validation windows only.

    Rows after the validation cut point are filtered out by the parquet reader,
    so test rows and their labels are never returned. TransactionDT is always included.
    """
    time = pd.read_parquet(path, columns=[TIME_COL])[TIME_COL]
    lo, hi = cut_points(time)
    cols = list(dict.fromkeys([TIME_COL, *columns]))
    df = pd.read_parquet(path, columns=cols, filters=[(TIME_COL, "<=", hi)])
    assert np.array_equal(df[TIME_COL].to_numpy(), time[time <= hi].to_numpy())
    is_train = (df[TIME_COL] <= lo).to_numpy()
    return df.loc[is_train].reset_index(drop=True), df.loc[~is_train].reset_index(drop=True)


def read_test(columns: list[str], path: Path = MERGED_PARQUET) -> pd.DataFrame:
    """Read columns for the test window only. Call this from the final evaluation step alone."""
    time = pd.read_parquet(path, columns=[TIME_COL])[TIME_COL]
    _, hi = cut_points(time)
    cols = list(dict.fromkeys([TIME_COL, *columns]))
    df = pd.read_parquet(path, columns=cols, filters=[(TIME_COL, ">", hi)])
    assert np.array_equal(df[TIME_COL].to_numpy(), time[time > hi].to_numpy())
    return df.reset_index(drop=True)


def load_split_frame(path: Path = MERGED_PARQUET) -> pd.DataFrame:
    """Load only the columns needed to build and describe the split."""
    return pd.read_parquet(path, columns=[ID_COL, TIME_COL, TARGET])


def write_split_info(
    df: pd.DataFrame, path: Path = SPLIT_INFO_JSON
) -> dict[str, dict]:
    """Compute masks, write split_info.json and return its contents."""
    masks = split_masks(df[TIME_COL])
    lo, hi = cut_points(df[TIME_COL])
    info: dict = {
        "fractions": list(SPLIT_FRACTIONS),
        "cut_points": {"train_max_dt": lo, "validation_max_dt": hi},
        **split_info(df, masks),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2) + "\n")
    return info


def run() -> None:
    """Build the split from the merged parquet, check ordering and write split_info.json."""
    df = load_split_frame()
    masks = split_masks(df[TIME_COL])
    time = df[TIME_COL]
    assert time[masks["train"]].max() < time[masks["validation"]].min()
    assert time[masks["validation"]].max() < time[masks["test"]].min()
    ids = [set(df.loc[masks[name], ID_COL]) for name in SPLIT_NAMES]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sum(int(m.sum()) for m in masks.values()) == len(df)

    info = write_split_info(df)
    print(f"Cut points: {info['cut_points']}")
    print(f"{'split':<11}{'rows':>9}{'frauds':>8}{'rate':>8}{'dt_min':>12}{'dt_max':>12}")
    for name in SPLIT_NAMES:
        s = info[name]
        print(
            f"{name:<11}{s['rows']:>9,}{s['frauds']:>8,}{s['fraud_rate']:>8.2%}"
            f"{s['dt_min']:>12,}{s['dt_max']:>12,}"
        )
    print(f"Wrote {SPLIT_INFO_JSON}")


if __name__ == "__main__":
    run()
