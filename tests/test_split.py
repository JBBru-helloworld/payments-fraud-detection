"""Tests for the chronological split."""

import numpy as np
import pandas as pd

from fraud.split import SPLIT_NAMES, split_info, split_masks


def make_frame(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Shuffled synthetic transactions with repeated timestamps."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n) + 3_000_000,
            "TransactionDT": rng.integers(86_400, 200_000, size=n),
            "isFraud": rng.binomial(1, 0.035, size=n),
        }
    )


def test_split_is_strictly_chronological() -> None:
    df = make_frame()
    masks = split_masks(df["TransactionDT"])
    dt = df["TransactionDT"]
    assert dt[masks["train"]].max() < dt[masks["validation"]].min()
    assert dt[masks["validation"]].max() < dt[masks["test"]].min()


def test_split_has_no_id_overlap_and_covers_all_rows() -> None:
    df = make_frame()
    masks = split_masks(df["TransactionDT"])
    stacked = np.vstack([masks[name] for name in SPLIT_NAMES])
    assert (stacked.sum(axis=0) == 1).all()
    ids = [set(df.loc[masks[name], "TransactionID"]) for name in SPLIT_NAMES]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_split_sizes_close_to_70_15_15() -> None:
    df = make_frame(n=10_000)
    masks = split_masks(df["TransactionDT"])
    shares = [masks[name].mean() for name in SPLIT_NAMES]
    assert np.allclose(shares, [0.70, 0.15, 0.15], atol=0.01)


def test_tied_timestamps_stay_in_one_split() -> None:
    df = pd.DataFrame({"TransactionDT": [1, 2, 3, 4, 5, 5, 5, 5, 6, 7]})
    masks = split_masks(df["TransactionDT"])
    tied = df["TransactionDT"] == 5
    assert any(masks[name][tied].all() for name in SPLIT_NAMES)


def test_split_info_reports_counts() -> None:
    df = make_frame()
    masks = split_masks(df["TransactionDT"])
    info = split_info(df, masks)
    assert sum(info[name]["rows"] for name in SPLIT_NAMES) == len(df)
    assert sum(info[name]["frauds"] for name in SPLIT_NAMES) == df["isFraud"].sum()
    assert info["train"]["dt_max"] < info["validation"]["dt_min"]
