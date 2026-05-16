import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Real rows taken directly from creditcard.csv for testing
# ---------------------------------------------------------------------------
FRAUD_TRANSACTION = {
    "Time": 406, "Amount": 0.0,
    "V1": -2.3122, "V2": 1.9519, "V3": -1.6097, "V4": 3.9979,
    "V5": -0.5222, "V6": -1.4265, "V7": -2.5374, "V8": 1.3914,
    "V9": -2.7700, "V10": -2.7722, "V11": 3.2020, "V12": -2.8992,
    "V13": -0.5950, "V14": -4.2895, "V15": 0.3898, "V16": -1.1407,
    "V17": -2.8300, "V18": -0.0168, "V19": 0.4165, "V20": 0.3269,
    "V21": 0.1474, "V22": -0.1703, "V23": 0.0359, "V24": -0.4118,
    "V25": 0.0714, "V26": 0.0719, "V27": 0.2127, "V28": 0.0952,
}

NORMAL_TRANSACTION = {
    "Time": 0, "Amount": 149.62,
    "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782,
    "V5": -0.3383, "V6": 0.4624, "V7": 0.2396, "V8": 0.0987,
    "V9": 0.3638, "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
    "V13": -0.9914, "V14": -0.3112, "V15": 1.4681, "V16": -0.4704,
    "V17": 0.2080, "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
    "V21": -0.0183, "V22": 0.2778, "V23": -0.1105, "V24": 0.0669,
    "V25": 0.1285, "V26": -0.1892, "V27": 0.1336, "V28": -0.0211,
}

ALL_ZEROS = {
    "Time": 0, "Amount": 0,
    **{f"V{i}": 0.0 for i in range(1, 29)},
}

LARGE_AMOUNT = {
    "Time": 100000, "Amount": 99999.99,
    **{f"V{i}": 0.0 for i in range(1, 29)},
}


# ---------------------------------------------------------------------------
# Health & root
# ---------------------------------------------------------------------------
def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_model_loaded():
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_root_returns_200():
    response = client.get("/")
    assert response.status_code == 200


def test_root_contains_endpoints():
    data = client.get("/").json()
    assert "endpoints" in data


# ---------------------------------------------------------------------------
# /predict — correct responses
# ---------------------------------------------------------------------------
def test_predict_fraud_transaction():
    """Known fraud row from dataset must return is_fraud = true."""
    response = client.post("/predict", json=FRAUD_TRANSACTION)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] is True
    assert data["fraud_probability"] > 0.5


def test_predict_normal_transaction():
    """Known normal row from dataset must return is_fraud = false."""
    response = client.post("/predict", json=NORMAL_TRANSACTION)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] is False
    assert data["fraud_probability"] < 0.5


# ---------------------------------------------------------------------------
# /predict — response schema
# ---------------------------------------------------------------------------
def test_predict_response_has_required_fields():
    response = client.post("/predict", json=NORMAL_TRANSACTION)
    data = response.json()
    assert "is_fraud"          in data
    assert "fraud_probability" in data
    assert "inference_ms"      in data


def test_predict_probability_in_range():
    response = client.post("/predict", json=NORMAL_TRANSACTION)
    prob = response.json()["fraud_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_inference_ms_is_positive():
    response = client.post("/predict", json=NORMAL_TRANSACTION)
    assert response.json()["inference_ms"] > 0


# ---------------------------------------------------------------------------
# /predict — edge cases
# ---------------------------------------------------------------------------
def test_predict_all_zeros():
    """All-zero input must not crash — returns a valid response."""
    response = client.post("/predict", json=ALL_ZEROS)
    assert response.status_code == 200
    assert "is_fraud" in response.json()


def test_predict_large_amount():
    """Very large transaction amount must not crash."""
    response = client.post("/predict", json=LARGE_AMOUNT)
    assert response.status_code == 200
    assert "is_fraud" in response.json()


# ---------------------------------------------------------------------------
# /predict — bad input
# ---------------------------------------------------------------------------
def test_predict_missing_field_returns_422():
    """Sending incomplete data must return HTTP 422 Unprocessable Entity."""
    incomplete = {"Time": 0, "Amount": 100}   # missing all V features
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_negative_amount_returns_422():
    """Amount must be >= 0. Negative value must be rejected."""
    bad = {**NORMAL_TRANSACTION, "Amount": -50}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_wrong_type_returns_422():
    """String value where float expected must be rejected."""
    bad = {**NORMAL_TRANSACTION, "V1": "not_a_number"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422
