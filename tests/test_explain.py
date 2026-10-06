"""Tests for the time buckets and grouped error rates used in the error analysis."""

import numpy as np
import pandas as pd
import pytest

from fraud.explain import error_rates, median_example, time_buckets

DAY = 86_400


def test_time_buckets_are_seven_day_windows() -> None:
    dt = pd.Series([0, 6 * DAY, 7 * DAY, 13 * DAY, 14 * DAY])
    labels = pd.Series([1, 1, 1, 1, 1])
    assert time_buckets(dt, labels, start=0, min_frauds=1).tolist() == [0, 0, 1, 1, 2]


def test_small_final_bucket_merges_into_previous() -> None:
    dt = pd.Series([0, 1 * DAY, 7 * DAY, 8 * DAY, 14 * DAY])
    labels = pd.Series([1, 1, 1, 1, 0])  # final bucket has 0 frauds
    assert time_buckets(dt, labels, start=0, min_frauds=2).tolist() == [0, 0, 1, 1, 1]


def test_merge_repeats_until_final_bucket_is_large_enough() -> None:
    dt = pd.Series([0, 1 * DAY, 7 * DAY, 14 * DAY])
    labels = pd.Series([1, 1, 1, 1])  # buckets 1 and 2 hold one fraud each
    assert time_buckets(dt, labels, start=0, min_frauds=2).tolist() == [0, 0, 1, 1]


def test_only_the_final_bucket_is_merged() -> None:
    dt = pd.Series([0, 7 * DAY, 14 * DAY, 15 * DAY])
    labels = pd.Series([1, 0, 1, 1])  # middle bucket is small but not final
    assert time_buckets(dt, labels, start=0, min_frauds=2).tolist() == [0, 1, 2, 2]


def test_error_rates_hand_checked() -> None:
    df = pd.DataFrame(
        {
            "g": ["a", "a", "a", "a", "b", "b", "b"],
            "label": [1, 1, 0, 0, 1, 0, 0],
            "flagged": [True, False, True, False, False, False, False],
        }
    )
    out = error_rates(df, "dim", "g").set_index("group")
    # Group a: 2 frauds, 1 missed; 2 legitimate, 1 flagged.
    assert out.loc["a", "fnr"] == pytest.approx(0.5)
    assert out.loc["a", "fpr"] == pytest.approx(0.5)
    assert (out.loc["a", "frauds"], out.loc["a", "legit"]) == (2, 2)
    # Group b: 1 fraud, missed; 2 legitimate, none flagged.
    assert out.loc["b", "fnr"] == pytest.approx(1.0)
    assert out.loc["b", "fpr"] == 0.0
    assert (out["dimension"] == "dim").all()


def test_error_rates_nan_without_frauds_and_missing_group() -> None:
    df = pd.DataFrame({"g": ["a", None], "label": [0, 0], "flagged": [True, False]})
    out = error_rates(df, "dim", "g").set_index("group")
    assert np.isnan(out.loc["a", "fnr"])
    assert "missing" in out.index


def test_median_example_is_lower_median_by_score() -> None:
    df = pd.DataFrame({"TransactionID": [4, 3, 2, 1], "score": [0.4, 0.1, 0.3, 0.2]})
    assert median_example(df)["TransactionID"] == 1
