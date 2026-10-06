"""Leakage-safe feature engineering: every statistic is fitted on the train split only.

TransactionDT has an unknown reference date, so hour of day and day of week are
relative values. TransactionDT itself, any day index, TransactionID and isFraud
never reach the output, because the fraud rate drifts over time.
"""

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from fraud.config import INTERIM_DIR, MERGED_PARQUET, MODELS_DIR, RANDOM_SEED, REPORTS_DIR
from fraud.data import ID_COL, TARGET, TIME_COL, column_groups
from fraud.split import split_masks

FEATURE_BUILDER_PATH: Path = MODELS_DIR / "feature_builder.joblib"
FEATURE_PATHS: dict[str, Path] = {
    "train": INTERIM_DIR / "features_train.parquet",
    "validation": INTERIM_DIR / "features_valid.parquet",
    "test": INTERIM_DIR / "features_test.parquet",
}
FEATURE_REPORT_MD: Path = REPORTS_DIR / "feature_report.md"

FORBIDDEN_COLS: tuple[str, ...] = (ID_COL, TIME_COL, TARGET)
MISSING_GROUPS: tuple[str, ...] = ("D", "M", "V", "identity")
FREQ_COLS: tuple[str, ...] = ("card1", "card2", "addr1", "P_emaildomain", "DeviceInfo")
EMAIL_COLS: tuple[str, str] = ("P_emaildomain", "R_emaildomain")

# Fixed domain-to-provider map keyed on the first label of the domain. No fitted statistics.
EMAIL_PROVIDERS: dict[str, str] = {
    "gmail": "google",
    "yahoo": "yahoo",
    "ymail": "yahoo",
    "rocketmail": "yahoo",
    "hotmail": "microsoft",
    "outlook": "microsoft",
    "live": "microsoft",
    "msn": "microsoft",
    "icloud": "apple",
    "me": "apple",
    "mac": "apple",
    "aol": "aol",
    "anonymous": "anonymous",
}

# Provider levels are fixed by the map above, so unseen domains still land in "other".
PROVIDER_LEVELS: list[str] = sorted({*EMAIL_PROVIDERS.values(), "other"})

ENGINEERED_NUMERIC: list[str] = [
    "log_amt",
    "amt_cents",
    "amt_is_whole",
    "hour_of_day",
    "day_of_week",
    "n_missing",
    *[f"n_missing_{g}" for g in MISSING_GROUPS],
    "email_match",
    "email_both_missing",
    *[f"freq_{c}" for c in FREQ_COLS],
    "card1_amt_mean",
    "card1_amt_std",
    "card1_amt_ratio",
]
ENGINEERED_CATEGORICAL: list[str] = ["P_email_provider", "R_email_provider"]


def amount_cents(amount: pd.Series) -> pd.Series:
    """Decimal part of the amount after rounding to 2 dp in float64, removing float32 noise."""
    amt = amount.astype(np.float64).round(2)
    return (amt - np.floor(amt)).round(2)


def email_provider(domain: object) -> object:
    """Provider name for a known domain, 'other' for unknown domains and NaN when missing."""
    if not isinstance(domain, str):
        return np.nan
    return EMAIL_PROVIDERS.get(domain.split(".")[0], "other")


def is_text(series: pd.Series) -> bool:
    """True for object or string columns."""
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def v_number(col: str) -> int:
    """Numeric suffix of a V column, used to order columns within a group."""
    return int(re.sub(r"\D", "", col))


class FeatureBuilder:
    """Fit encodings, aggregates, category levels and drop lists on train; transform any split."""

    def __init__(
        self,
        missing_threshold: float = 0.90,
        corr_threshold: float = 0.98,
        corr_sample_rows: int = 100_000,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.missing_threshold = missing_threshold
        self.corr_threshold = corr_threshold
        self.corr_sample_rows = corr_sample_rows
        self.random_state = random_state

    # Fitting

    def fit(self, train_df: pd.DataFrame) -> "FeatureBuilder":
        """Learn every statistic from the training rows only."""
        df = train_df.drop(columns=[c for c in (ID_COL, TARGET) if c in train_df])
        self.input_columns_: list[str] = list(df.columns)
        groups = column_groups(self.input_columns_)
        self.missing_groups_: dict[str, list[str]] = {g: groups[g] for g in MISSING_GROUPS}
        self.counted_columns_: list[str] = [c for c in self.input_columns_ if c != TIME_COL]

        self.dropped_: dict[str, list[str]] = self._fit_drops(df, groups["V"])
        dropped = {c for cols in self.dropped_.values() for c in cols}
        self.kept_columns_: list[str] = [
            c for c in self.counted_columns_ if c not in dropped
        ]

        self.freq_maps_: dict[str, dict] = {
            col: self._key(df[col]).value_counts(normalize=True).to_dict() for col in FREQ_COLS
        }
        amt = df["TransactionAmt"].astype(np.float64)
        self.card_stats_: pd.DataFrame = amt.groupby(self._key(df["card1"])).agg(["mean", "std"])

        self.categories_: dict[str, list[str]] = {
            col: sorted(df[col].dropna().unique().tolist())
            for col in self.kept_columns_
            if is_text(df[col])
        }
        for col in ENGINEERED_CATEGORICAL:
            self.categories_[col] = PROVIDER_LEVELS

        self.output_columns_: list[str] = [
            *self.kept_columns_, *ENGINEERED_NUMERIC, *ENGINEERED_CATEGORICAL
        ]
        return self

    def _fit_drops(self, df: pd.DataFrame, v_cols: list[str]) -> dict[str, list[str]]:
        """Constant, mostly missing and correlated V columns, decided on train."""
        cols = [c for c in df.columns if c != TIME_COL]
        constant = [c for c in cols if df[c].nunique(dropna=False) <= 1]
        missing = df[cols].isna().mean()
        mostly_missing = [
            c for c in cols if missing[c] > self.missing_threshold and c not in constant
        ]
        removed = set(constant) | set(mostly_missing)
        candidates = [c for c in v_cols if c not in removed]
        return {
            "constant": constant,
            "over_90_missing": mostly_missing,
            "correlated_v": self._correlated_v(df, candidates),
        }

    def _correlated_v(self, df: pd.DataFrame, v_cols: list[str]) -> list[str]:
        """V columns with |corr| above the threshold against an earlier kept column in their group.

        Groups share an identical missing count on train. Correlations use a row sample.
        """
        if not v_cols:
            return []
        n = min(self.corr_sample_rows, len(df))
        sample = df[v_cols].sample(n=n, random_state=self.random_state).astype(np.float64)
        missing_counts = df[v_cols].isna().sum()
        dropped: list[str] = []
        for _, members in missing_counts.groupby(missing_counts, sort=False):
            group = sorted(members.index, key=v_number)
            corr = sample[group].corr().abs()
            kept: list[str] = []
            for col in group:
                if any(corr.loc[col, k] > self.corr_threshold for k in kept):
                    dropped.append(col)
                else:
                    kept.append(col)
        return sorted(dropped, key=v_number)

    # Transforming

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the feature matrix. Reads fitted state only and never changes it."""
        x = df.reindex(columns=self.input_columns_)
        amt = x["TransactionAmt"].astype(np.float64)
        dt = x[TIME_COL].astype(np.float64)

        eng: dict[str, pd.Series] = {
            "log_amt": np.log1p(amt),
            "amt_cents": amount_cents(amt),
            "hour_of_day": np.floor(dt / 3600) % 24,
            "day_of_week": np.floor(dt / 86_400) % 7,
            "n_missing": x[self.counted_columns_].isna().sum(axis=1),
        }
        eng["amt_is_whole"] = (eng["amt_cents"] == 0).where(amt.notna())
        for group, cols in self.missing_groups_.items():
            eng[f"n_missing_{group}"] = x[cols].isna().sum(axis=1)

        p_mail, r_mail = (x[c] for c in EMAIL_COLS)
        eng["email_match"] = (p_mail.notna() & p_mail.eq(r_mail).fillna(False)).astype(int)
        eng["email_both_missing"] = (p_mail.isna() & r_mail.isna()).astype(int)

        for col in FREQ_COLS:
            key = self._key(x[col])
            encoded = key.map(self.freq_maps_[col])
            eng[f"freq_{col}"] = encoded.mask(key.notna() & encoded.isna(), 0.0)

        card = self._key(x["card1"])
        eng["card1_amt_mean"] = card.map(self.card_stats_["mean"])
        eng["card1_amt_std"] = card.map(self.card_stats_["std"])
        eng["card1_amt_ratio"] = amt / eng["card1_amt_mean"]

        numeric = pd.DataFrame(eng, index=x.index)[ENGINEERED_NUMERIC]
        providers = pd.DataFrame(
            {new: x[old].map(email_provider) for new, old in zip(ENGINEERED_CATEGORICAL, EMAIL_COLS)},
            index=x.index,
        )
        out = pd.concat([x[self.kept_columns_], numeric, providers], axis=1)
        return self._cast(out)[self.output_columns_]

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit on the training rows and return their feature matrix."""
        return self.fit(train_df).transform(train_df)

    def _cast(self, out: pd.DataFrame) -> pd.DataFrame:
        """Categories with train levels (unseen levels become NaN); everything else float32."""
        cast: dict[str, pd.Series] = {}
        for col in out.columns:
            if col in self.categories_:
                levels = self.categories_[col]
                known = out[col].where(out[col].isin(levels))
                dtype = pd.CategoricalDtype(levels)
                cast[col] = pd.Series(pd.Categorical(known, dtype=dtype), index=out.index)
            else:
                cast[col] = out[col].astype(np.float32)
        return pd.DataFrame(cast, index=out.index)

    @staticmethod
    def _key(series: pd.Series) -> pd.Series:
        """Lookup key for encodings: float64 for numeric columns, unchanged for text."""
        return series if is_text(series) else series.astype(np.float64)

    # Reporting

    def engineered_features(self) -> list[str]:
        """Names of the features created here, numeric then categorical."""
        return [*ENGINEERED_NUMERIC, *ENGINEERED_CATEGORICAL]


def load_unlabelled(path: Path = MERGED_PARQUET) -> pd.DataFrame:
    """Load every column except the label, so no split's labels are read here."""
    columns = [c for c in pq.read_schema(path).names if c != TARGET]
    return pd.read_parquet(path, columns=columns)


def check_schema(frames: dict[str, pd.DataFrame]) -> None:
    """Assert identical column names, order and dtypes, and no forbidden columns."""
    reference = frames["train"]
    for name, frame in frames.items():
        assert list(frame.columns) == list(reference.columns), f"{name} columns differ"
        assert frame.dtypes.equals(reference.dtypes), f"{name} dtypes differ"
        assert not set(FORBIDDEN_COLS) & set(frame.columns), f"{name} has forbidden columns"


def check_train_only(builder: FeatureBuilder, train: pd.DataFrame) -> None:
    """Assert the card1 frequency map and card aggregates match a recomputation on train rows."""
    card = train["card1"].astype(np.float64)
    assert builder.freq_maps_["card1"] == card.value_counts(normalize=True).to_dict()
    expected = train["TransactionAmt"].astype(np.float64).groupby(card).agg(["mean", "std"])
    pd.testing.assert_frame_equal(builder.card_stats_, expected)


def write_report(builder: FeatureBuilder, n_rows: dict[str, int], path: Path = FEATURE_REPORT_MD) -> Path:
    """Write the generated feature report."""
    n_inputs = len(builder.input_columns_)
    n_cat = sum(1 for c in builder.output_columns_ if c in builder.categories_)
    lines = [
        "# Feature report",
        "",
        "Generated by `python -m fraud.features`. All statistics and drop decisions are fitted on the train split only.",
        "",
        f"- Input columns (excluding {ID_COL} and {TARGET}): {n_inputs}",
        f"- Output columns: {len(builder.output_columns_)} ({n_cat} categorical, "
        f"{len(builder.output_columns_) - n_cat} numeric as float32)",
        f"- Raw columns kept: {len(builder.kept_columns_)}",
        f"- Engineered features: {len(builder.engineered_features())}",
        "- Rows: " + ", ".join(f"{k} {v:,}" for k, v in n_rows.items()),
        f"- {TIME_COL} is used to derive hour of day and day of week, then removed.",
        "",
        "## Dropped columns",
        "",
        "| reason | count | columns |",
        "| --- | --- | --- |",
    ]
    for reason, cols in builder.dropped_.items():
        lines.append(f"| {reason} | {len(cols)} | {', '.join(cols) or 'none'} |")
    lines += ["", "## Engineered features", ""]
    lines += [f"- {name}" for name in builder.engineered_features()]
    lines += ["", "## Categorical columns (levels learned on train; email providers use a fixed set)", "", "| column | levels |", "| --- | --- |"]
    lines += [f"| {c} | {len(v)} |" for c, v in builder.categories_.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def run() -> None:
    """Fit on train, transform every split, verify, and write the artefacts and report."""
    df = load_unlabelled()
    masks = split_masks(df[TIME_COL])
    splits = {name: df.loc[mask] for name, mask in masks.items()}

    builder = FeatureBuilder()
    frames = {"train": builder.fit_transform(splits["train"])}
    state = joblib.hash(builder)
    for name in ("validation", "test"):
        frames[name] = builder.transform(splits[name])
        assert joblib.hash(builder) == state, f"transform({name}) changed fitted state"

    check_schema(frames)
    check_train_only(builder, splits["train"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(builder, FEATURE_BUILDER_PATH)
    for name, frame in frames.items():
        frame.to_parquet(FEATURE_PATHS[name], index=False)
    write_report(builder, {k: len(v) for k, v in frames.items()})

    print(f"Input columns: {len(builder.input_columns_)}, output columns: {frames['train'].shape[1]}")
    for reason, cols in builder.dropped_.items():
        print(f"Dropped ({reason}): {len(cols)}")
    print("Checks passed: schema identical, no forbidden columns, state unchanged, train-only stats")
    print(f"Wrote {FEATURE_BUILDER_PATH}, {FEATURE_REPORT_MD} and {len(frames)} feature parquets")


if __name__ == "__main__":
    run()
