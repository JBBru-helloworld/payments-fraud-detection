"""API tests with FastAPI's TestClient. Skipped when the trained artefacts are absent."""

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import ARTEFACTS, app
from fraud.config import MERGED_PARQUET, RANDOM_SEED
from fraud.data import ID_COL
from fraud.threshold import VALID_SCORES_PATH

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in ARTEFACTS),
    reason="trained artefacts missing: run `make features`, `make models` and `make threshold`",
)

# Invented values, not rows from the dataset.
SYNTHETIC = {
    "TransactionAmt": 59.95,
    "TransactionDT": 1_000_000,
    "ProductCD": "W",
    "card1": 12345,
    "card2": 321.0,
    "card4": "visa",
    "card6": "debit",
    "addr1": 299.0,
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com",
    "DeviceType": "desktop",
    "extra": {"C1": 2.0, "C13": 3.0, "D1": 14.0, "M4": "M0", "V258": 1.0, "id_12": "NotFound"},
}


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert len(body["model_version"]) == 12
    assert body["threshold"] > 0


def test_predict_with_amount_only(client: TestClient) -> None:
    response = client.post("/predict", json={"TransactionAmt": 25.0})
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["fraud_score"], float)
    assert body["decision"] in ("flag", "allow")
    assert body["decision"] == ("flag" if body["fraud_score"] >= body["threshold"] else "allow")


def test_predict_synthetic_payload(client: TestClient) -> None:
    response = client.post("/predict", json=SYNTHETIC)
    assert response.status_code == 200
    assert response.json()["ignored_fields"] == []


def test_unknown_extra_keys_are_ignored_and_listed(client: TestClient) -> None:
    payload = {**SYNTHETIC, "extra": {**SYNTHETIC["extra"], "not_a_column": 1.0, "isFraud": 1}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["ignored_fields"] == ["isFraud", "not_a_column"]


@pytest.mark.parametrize(
    "payload",
    [
        {"TransactionAmt": -1.0},
        {"TransactionAmt": "12.50"},
        {"ProductCD": "W"},
        {"TransactionAmt": 10.0, "card1": "1234"},
        {"TransactionAmt": 10.0, "unexpected_top_level": 1},
        {"TransactionAmt": 10.0, "extra": {"C1": "two"}},
    ],
    ids=["negative", "string_amount", "missing_amount", "string_card1", "unknown_top_level", "string_numeric_extra"],
)
def test_invalid_input_returns_422(client: TestClient, payload: dict) -> None:
    assert client.post("/predict", json=payload).status_code == 422


def test_unseen_category_is_not_a_server_error(client: TestClient) -> None:
    response = client.post("/predict", json={**SYNTHETIC, "ProductCD": "ZZ_unseen"})
    assert response.status_code == 200


@pytest.mark.skipif(
    not (MERGED_PARQUET.exists() and VALID_SCORES_PATH.exists()),
    reason="data/interim files missing: run `make data`, `make features` and `make models`",
)
def test_served_scores_match_batch_scores(client: TestClient) -> None:
    from fraud.payload import read_validation_rows, row_to_payload

    rows = read_validation_rows()
    sample = rows.iloc[np.random.default_rng(RANDOM_SEED).choice(len(rows), size=50, replace=False)]
    expected = pd.read_parquet(VALID_SCORES_PATH).set_index(ID_COL)["score"]
    for _, row in sample.iterrows():
        response = client.post("/predict", json=row_to_payload(row))
        assert response.status_code == 200
        assert response.json()["ignored_fields"] == []
        assert response.json()["fraud_score"] == pytest.approx(expected[int(row[ID_COL])], abs=1e-6)
