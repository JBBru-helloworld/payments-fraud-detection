"""Load the IEEE-CIS training CSVs, merge them, downcast numerics and save parquet."""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud.config import MERGED_PARQUET, TRAIN_IDENTITY_CSV, TRAIN_TRANSACTION_CSV

TARGET = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"
GROUP_ORDER = ("transaction", "C", "D", "M", "V", "identity")


def column_group(col: str) -> str | None:
    """Return the column group name, or None for ID, target and time columns."""
    if col in (ID_COL, TARGET, TIME_COL):
        return None
    for prefix in ("C", "D", "M", "V"):
        if re.fullmatch(rf"{prefix}\d+", col):
            return prefix
    if col.startswith("id_") or col in ("DeviceType", "DeviceInfo"):
        return "identity"
    return "transaction"


def column_groups(columns: Sequence[str]) -> dict[str, list[str]]:
    """Map each group name to its columns, in GROUP_ORDER."""
    groups: dict[str, list[str]] = {g: [] for g in GROUP_ORDER}
    for col in columns:
        group = column_group(col)
        if group is not None:
            groups[group].append(col)
    return groups


def load_raw(
    transaction_path: Path = TRAIN_TRANSACTION_CSV,
    identity_path: Path = TRAIN_IDENTITY_CSV,
) -> tuple[pd.DataFrame, float]:
    """Left-join identity onto transactions. Return the merged frame and the share with identity."""
    for path in (transaction_path, identity_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Place the Kaggle CSV in data/raw/.")
    transactions = pd.read_csv(transaction_path)
    identity = pd.read_csv(identity_path)
    identity_share = float(transactions[ID_COL].isin(identity[ID_COL]).mean())
    merged = transactions.merge(identity, on=ID_COL, how="left", validate="one_to_one")
    return merged, identity_share


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Cast float64 to float32 and int64 to the smallest integer type that holds the values."""
    dtypes: dict[str, Any] = {
        col: np.float32 for col in df.select_dtypes(include="float64").columns
    }
    for col in df.select_dtypes(include="int64").columns:
        dtypes[col] = pd.to_numeric(df[col], downcast="integer").dtype
    return df.astype(dtypes)


def memory_mb(df: pd.DataFrame) -> float:
    """Return the deep memory usage of a DataFrame in megabytes."""
    return df.memory_usage(deep=True).sum() / 1024**2


def build_merged(output_path: Path = MERGED_PARQUET) -> pd.DataFrame:
    """Load, merge, downcast and write the merged training table to parquet."""
    merged, identity_share = load_raw()
    before = memory_mb(merged)
    merged = downcast(merged)
    after = memory_mb(merged)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)

    print(f"Merged shape: {merged.shape}")
    print(f"Memory before downcast: {before:,.1f} MB")
    print(f"Memory after downcast:  {after:,.1f} MB")
    print(f"Fraud rate: {merged[TARGET].mean():.4%}")
    print(f"Rows with identity data: {identity_share:.4%}")
    print(f"Wrote {output_path}")
    return merged


if __name__ == "__main__":
    build_merged()
