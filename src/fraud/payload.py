"""Build /predict request bodies from raw dataset rows.

Used by `make sample-request` and the API parity test. Payloads written to disk stay
under data/interim/ (gitignored) because they contain rows of the Kaggle data.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from fraud.config import INTERIM_DIR, MERGED_PARQUET
from fraud.data import ID_COL, TARGET, TIME_COL
from fraud.split import cut_points

SAMPLE_PAYLOAD_PATH: Path = INTERIM_DIR / "sample_payload.json"

# Raw columns accepted as top-level fields by /predict; every other raw column goes under `extra`.
CORE_NUMERIC: tuple[str, ...] = (
    "TransactionAmt", TIME_COL, "card1", "card2", "card3", "card5", "addr1", "addr2", "dist1",
)
CORE_TEXT: tuple[str, ...] = (
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo",
)
CORE_FIELDS: tuple[str, ...] = (*CORE_NUMERIC, *CORE_TEXT)


def json_value(value: Any) -> float | int | str | None:
    """Plain JSON value; NaN and None become None. float32 values convert exactly to float64."""
    if value is None or (isinstance(value, float | np.floating) and math.isnan(value)):
        return None
    if isinstance(value, bool | np.bool_):
        return str(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return float(value)
    return value if isinstance(value, str) else str(value)


def row_to_payload(row: pd.Series) -> dict[str, Any]:
    """Core fields at the top level, every other non-missing raw column under `extra`."""
    skip = {ID_COL, TARGET}
    payload: dict[str, Any] = {f: json_value(row.get(f)) for f in CORE_FIELDS}
    payload["extra"] = {
        col: json_value(value)
        for col, value in row.items()
        if col not in skip and col not in CORE_FIELDS and json_value(value) is not None
    }
    return payload


def read_validation_rows(path: Path = MERGED_PARQUET) -> pd.DataFrame:
    """Raw validation rows (all columns except the label), in TransactionDT order."""
    time = pd.read_parquet(path, columns=[TIME_COL])[TIME_COL]
    lo, hi = cut_points(time)
    columns = [c for c in pq.read_schema(path).names if c != TARGET]
    rows = pd.read_parquet(path, columns=columns, filters=[(TIME_COL, ">", lo), (TIME_COL, "<=", hi)])
    return rows.reset_index(drop=True)


def write_sample_payload(transaction_id: int | None = None, path: Path = SAMPLE_PAYLOAD_PATH) -> Path:
    """Write one validation row as a /predict body (the first row unless an ID is given)."""
    rows = read_validation_rows()
    row = rows.iloc[0] if transaction_id is None else rows.loc[rows[ID_COL] == transaction_id].iloc[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row_to_payload(row), indent=2) + "\n")
    print(f"Wrote {path} for TransactionID {int(row[ID_COL])} ({len(row_to_payload(row)['extra'])} extra fields)")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transaction-id", type=int, default=None, help="validation TransactionID to use")
    write_sample_payload(parser.parse_args().transaction_id)
