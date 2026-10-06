"""The saved final model reloads and reproduces its recorded validation scores."""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from fraud.config import RANDOM_SEED
from fraud.features import FEATURE_PATHS
from fraud.train import FINAL_MODEL_PATH, FINAL_PARAMS_JSON, VALID_SCORES_PATH, to_category_codes

ARTEFACTS = (FINAL_MODEL_PATH, FINAL_PARAMS_JSON, VALID_SCORES_PATH, FEATURE_PATHS["validation"])


@pytest.mark.skipif(
    not all(p.exists() for p in ARTEFACTS), reason="run `make features` and `make models` first"
)
def test_saved_model_reproduces_validation_scores() -> None:
    info = json.loads(FINAL_PARAMS_JSON.read_text())
    features = pd.read_parquet(FEATURE_PATHS["validation"])
    scores = pd.read_parquet(VALID_SCORES_PATH)
    assert len(features) == len(scores)

    rows = np.random.default_rng(RANDOM_SEED).choice(len(features), size=1_000, replace=False)
    sample = to_category_codes(features.iloc[rows], info["categorical_as_codes"])
    assert list(sample.columns) == info["feature_names"]

    booster = lgb.Booster(model_file=FINAL_MODEL_PATH)
    predicted = np.asarray(booster.predict(sample), dtype=np.float64)
    np.testing.assert_allclose(predicted, scores["score"].to_numpy()[rows], rtol=1e-9)
