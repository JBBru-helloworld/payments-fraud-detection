"""Tests for the FeatureBuilder."""

import joblib
import numpy as np
import pandas as pd
import pytest

from fraud.features import FORBIDDEN_COLS, FeatureBuilder, amount_cents, check_schema


def make_frame(n: int = 400, start: int = 0, seed: int = 42) -> pd.DataFrame:
    """Synthetic transactions covering every column family the builder touches."""
    rng = np.random.default_rng(seed)
    v1 = rng.normal(size=n)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(start, start + n),
            "isFraud": rng.binomial(1, 0.1, size=n),
            "TransactionDT": np.arange(start, start + n) * 3_600 + 86_400,
            "TransactionAmt": rng.choice([49.99, 10.0, 25.5, 100.0], size=n).astype(np.float32),
            "card1": rng.choice([1000, 2000, 3000], size=n).astype(np.int16),
            "card2": rng.choice([111.0, 222.0, np.nan], size=n).astype(np.float32),
            "addr1": rng.choice([300.0, 400.0], size=n).astype(np.float32),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com", None], size=n),
            "R_emaildomain": rng.choice(["gmail.com", None], size=n),
            "ProductCD": rng.choice(["W", "C"], size=n),
            "DeviceInfo": rng.choice(["Windows", "iOS Device", None], size=n),
            "D1": rng.choice([0.0, 5.0, np.nan], size=n).astype(np.float32),
            "M1": rng.choice(["T", "F", None], size=n),
            "V1": v1.astype(np.float32),
            "V2": (2 * v1 + 1).astype(np.float32),
            "V3": rng.normal(size=n).astype(np.float32),
            "V4": np.ones(n, dtype=np.float32),
            "id_01": np.where(rng.random(n) < 0.95, np.nan, -5.0).astype(np.float32),
        }
    )


@pytest.fixture(scope="module")
def fitted() -> tuple[FeatureBuilder, pd.DataFrame]:
    train = make_frame()
    return FeatureBuilder(corr_sample_rows=200).fit(train), train


def test_cents_removes_float32_noise() -> None:
    stored = pd.Series([49.99], dtype=np.float32)
    assert float(stored.iloc[0]) != 49.99
    assert amount_cents(stored).iloc[0] == 0.99


def test_cents_and_whole_flag_in_output(fitted: tuple[FeatureBuilder, pd.DataFrame]) -> None:
    builder, train = fitted
    out = builder.transform(train)
    is_4999 = train["TransactionAmt"].to_numpy() == np.float32(49.99)
    assert np.allclose(out.loc[is_4999, "amt_cents"], 0.99)
    assert (out.loc[train["TransactionAmt"] == 10.0, "amt_is_whole"] == 1).all()


def test_drops_constant_missing_and_correlated(fitted: tuple[FeatureBuilder, pd.DataFrame]) -> None:
    builder, _ = fitted
    assert builder.dropped_["constant"] == ["V4"]
    assert builder.dropped_["over_90_missing"] == ["id_01"]
    assert builder.dropped_["correlated_v"] == ["V2"]
    assert "V1" in builder.output_columns_ and "V3" in builder.output_columns_


def test_unseen_values_do_not_raise(fitted: tuple[FeatureBuilder, pd.DataFrame]) -> None:
    builder, _ = fitted
    new = make_frame(n=5, start=10_000, seed=7)
    new["ProductCD"] = "Z"
    new["card1"] = np.int16(9999)
    new["P_emaildomain"] = "unseen.org"
    out = builder.transform(new)
    assert out["ProductCD"].isna().all()
    assert (out["freq_card1"] == 0).all()
    assert (out["freq_P_emaildomain"] == 0).all()
    assert out["card1_amt_mean"].isna().all()
    assert (out["P_email_provider"] == "other").all()


def test_missing_input_columns_are_tolerated(fitted: tuple[FeatureBuilder, pd.DataFrame]) -> None:
    builder, train = fitted
    out = builder.transform(train.drop(columns=["DeviceInfo", "V3"]).head(3))
    assert list(out.columns) == builder.output_columns_
    assert out["V3"].isna().all()


def test_fit_uses_train_rows_only(fitted: tuple[FeatureBuilder, pd.DataFrame]) -> None:
    builder, train = fitted
    expected = train["card1"].astype(np.float64).value_counts(normalize=True).to_dict()
    assert builder.freq_maps_["card1"] == expected
    means = train.groupby("card1")["TransactionAmt"].apply(lambda s: s.astype(np.float64).mean())
    assert np.allclose(builder.card_stats_["mean"].to_numpy(), means.to_numpy())

    valid = make_frame(start=400, seed=1)
    valid["card1"] = np.int16(1000)
    refit = FeatureBuilder(corr_sample_rows=200).fit(pd.concat([train, valid]))
    assert refit.freq_maps_["card1"] != builder.freq_maps_["card1"]


def test_transform_does_not_change_state(fitted: tuple[FeatureBuilder, pd.DataFrame]) -> None:
    builder, _ = fitted
    before = joblib.hash(builder)
    builder.transform(make_frame(start=400, seed=1))
    assert joblib.hash(builder) == before


def test_identical_schema_and_no_forbidden_columns(
    fitted: tuple[FeatureBuilder, pd.DataFrame],
) -> None:
    builder, train = fitted
    frames = {
        "train": builder.transform(train),
        "validation": builder.transform(make_frame(start=400, seed=1)),
        "test": builder.transform(make_frame(start=800, seed=2)),
    }
    check_schema(frames)
    for frame in frames.values():
        assert not set(FORBIDDEN_COLS) & set(frame.columns)
        numeric = frame.select_dtypes(exclude="category")
        assert (numeric.dtypes == np.float32).all()
