"""Tests for the EDA helpers."""

import pandas as pd

from fraud.eda import chronological_windows, column_groups, fraud_rate_by, has_identity


def test_column_groups_assigns_each_prefix() -> None:
    cols = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "card1",
            "C1", "D15", "M4", "V339", "id_01", "DeviceType", "DeviceInfo"]
    groups = column_groups(cols)
    assert groups["transaction"] == ["TransactionAmt", "card1"]
    assert groups["C"] == ["C1"]
    assert groups["D"] == ["D15"]
    assert groups["M"] == ["M4"]
    assert groups["V"] == ["V339"]
    assert groups["identity"] == ["id_01", "DeviceType", "DeviceInfo"]


def test_fraud_rate_by_labels_missing() -> None:
    df = pd.DataFrame({"cat": ["a", "a", None, None], "isFraud": [1, 0, 1, 1]})
    out = fraud_rate_by(df, "cat")
    assert out.loc["a", "fraud_rate"] == 0.5
    assert out.loc["missing", "rows"] == 2


def test_has_identity_any_column() -> None:
    df = pd.DataFrame({"id_01": [1.0, None, None], "DeviceType": [None, "mobile", None]})
    assert has_identity(df, ["id_01", "DeviceType"]).tolist() == [True, True, False]


def test_chronological_windows_are_ordered_and_complete() -> None:
    df = pd.DataFrame({"TransactionDT": range(100, 0, -1), "isFraud": [0] * 100})
    windows = chronological_windows(df)
    assert windows["rows"].tolist() == [70, 15, 15]
    assert windows["dt_max"].iloc[0] < windows["dt_min"].iloc[1]
    assert windows["dt_max"].iloc[1] < windows["dt_min"].iloc[2]
