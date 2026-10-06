"""Tests for the data loading helpers."""

import numpy as np
import pandas as pd

from fraud.data import downcast


def test_downcast_preserves_values_and_shrinks_types() -> None:
    df = pd.DataFrame(
        {
            "small_int": np.array([0, 1, 1], dtype=np.int64),
            "large_int": np.array([0, 3_000_000, 5], dtype=np.int64),
            "flt": np.array([1.5, np.nan, 2.25], dtype=np.float64),
            "cat": ["a", "b", None],
        }
    )
    out = downcast(df)

    assert out["small_int"].dtype == np.int8
    assert out["large_int"].dtype == np.int32
    assert out["flt"].dtype == np.float32
    assert out["cat"].dtype == df["cat"].dtype
    pd.testing.assert_frame_equal(out.astype(df.dtypes.to_dict()), df)


def test_downcast_does_not_mutate_input() -> None:
    df = pd.DataFrame({"x": np.array([1, 2], dtype=np.int64)})
    downcast(df)
    assert df["x"].dtype == np.int64
