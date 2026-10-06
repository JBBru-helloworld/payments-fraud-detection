"""FastAPI service for the fraud model: GET /health and POST /predict.

The FeatureBuilder, the LightGBM model and the decision threshold are loaded once at
start-up. Request bodies are never logged.
"""

import hashlib
import json
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from fraud.config import MODELS_DIR
from fraud.features import FEATURE_BUILDER_PATH, load_feature_builder
from fraud.payload import CORE_FIELDS
from fraud.threshold import THRESHOLD_JSON

FINAL_MODEL_PATH: Path = MODELS_DIR / "final_model.txt"
ARTEFACTS: dict[Path, str] = {
    FEATURE_BUILDER_PATH: "make features",
    FINAL_MODEL_PATH: "make models",
    THRESHOLD_JSON: "make threshold",
}
SCORE_NOTE = (
    "Model output score on the same scale as the threshold. It is not a calibrated probability: "
    "it ranks transactions by risk but does not equal the chance of fraud."
)
DESCRIPTION = (
    "Scores a single card-not-present transaction with the LightGBM model trained on the IEEE-CIS data.\n\n"
    "**Accuracy depends on how many raw columns are supplied.** The model relies heavily on anonymised "
    "columns (the C, D, M, V and id groups), which go under `extra`. Missing fields are treated "
    "as missing values, so predictions from the core fields alone will be less reliable than the "
    "validation metrics suggest.\n\n" + SCORE_NOTE
)



def optional_number() -> Any:
    """Optional numeric field that rejects strings and booleans."""
    return Field(default=None, strict=True)


class Transaction(BaseModel):
    """One raw transaction. Core fields at the top level; any other raw column in `extra`."""

    model_config = ConfigDict(extra="forbid")

    TransactionAmt: float = Field(..., ge=0, strict=True, description="Transaction amount, at least 0.")
    TransactionDT: float | None = Field(default=None, ge=0, strict=True,
                                        description="Seconds from the dataset's unknown reference time.")
    ProductCD: str | None = None
    card1: float | None = optional_number()
    card2: float | None = optional_number()
    card3: float | None = optional_number()
    card4: str | None = None
    card5: float | None = optional_number()
    card6: str | None = None
    addr1: float | None = optional_number()
    addr2: float | None = optional_number()
    dist1: float | None = optional_number()
    P_emaildomain: str | None = None
    R_emaildomain: str | None = None
    DeviceType: str | None = None
    DeviceInfo: str | None = None
    extra: dict[str, float | str | None] = Field(
        default_factory=dict,
        description="Other raw columns, for example C1, D4, M6, V258, id_31. Unknown keys are ignored.",
    )


class Prediction(BaseModel):
    """Score, threshold and decision for one transaction."""

    fraud_score: float = Field(description=SCORE_NOTE)
    threshold: float = Field(description="Cost-based threshold chosen on validation, on the same scale.")
    decision: Literal["flag", "allow"] = Field(description="'flag' when fraud_score >= threshold.")
    ignored_fields: list[str] = Field(description="Keys in `extra` that are not raw dataset columns.")


class Health(BaseModel):
    """Service status."""

    status: str
    model_version: str = Field(description="First 12 characters of the SHA-256 of final_model.txt.")
    threshold: float


def check_artefacts() -> None:
    """Fail with the missing file and the make target that creates it."""
    for path, target in ARTEFACTS.items():
        if not path.exists():
            raise RuntimeError(f"Missing artefact {path}. Run `{target}` to create it.")


def model_version(path: Path = FINAL_MODEL_PATH) -> str:
    """Short content hash of the model file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the FeatureBuilder, model and threshold once."""
    check_artefacts()
    builder = load_feature_builder(FEATURE_BUILDER_PATH)
    booster = lgb.Booster(model_file=FINAL_MODEL_PATH)
    if booster.feature_name() != builder.output_columns_:
        raise RuntimeError("Model features do not match the FeatureBuilder output columns.")
    app.state.builder = builder
    app.state.booster = booster
    app.state.threshold = float(json.loads(THRESHOLD_JSON.read_text())["chosen"]["threshold"])
    app.state.version = model_version()
    app.state.raw_columns = set(builder.input_columns_)
    app.state.text_columns = set(builder.categories_)
    yield


app = FastAPI(title="Payments fraud detection", version="1.0.0", description=DESCRIPTION, lifespan=lifespan)


def build_row(txn: Transaction, raw_columns: set[str], text_columns: set[str]) -> tuple[dict[str, Any], list[str]]:
    """Merge core fields and `extra` into one raw row; return it and the ignored keys."""
    row: dict[str, Any] = {f: getattr(txn, f) for f in CORE_FIELDS}
    ignored = sorted(k for k in txn.extra if k not in raw_columns or k in CORE_FIELDS)
    wrong_type = sorted(
        k for k, v in txn.extra.items()
        if k not in ignored and k not in text_columns and isinstance(v, str)
    )
    if wrong_type:
        raise HTTPException(status_code=422, detail=f"Numeric columns given as strings: {wrong_type}")
    row.update({k: v for k, v in txn.extra.items() if k not in ignored})
    return {k: (np.nan if v is None else v) for k, v in row.items()}, ignored


@app.get("/health", response_model=Health)
def health(request: Request) -> Health:
    """Liveness check with the model version and threshold."""
    state = request.app.state
    return Health(status="ok", model_version=state.version, threshold=state.threshold)


@app.post("/predict", response_model=Prediction)
def predict(txn: Transaction, request: Request) -> Prediction:
    """Score one transaction.

    fraud_score is the model's output score on the same scale as the threshold. It is not a
    calibrated probability. Missing raw columns are treated as missing values.
    """
    state = request.app.state
    row, ignored = build_row(txn, state.raw_columns, state.text_columns)
    features = state.builder.transform(pd.DataFrame([row]))
    score = float(state.booster.predict(features)[0])
    if not math.isfinite(score):
        raise HTTPException(status_code=500, detail="Model returned a non-finite score.")
    decision: Literal["flag", "allow"] = "flag" if score >= state.threshold else "allow"
    return Prediction(fraud_score=score, threshold=state.threshold, decision=decision, ignored_fields=ignored)
