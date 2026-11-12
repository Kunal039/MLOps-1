"""
Automated tests for the Flask API (uses Flask's test client - no server needed).
Run:  python -m pytest test_api.py -v
"""

import pytest

from app import app

SETOSA = {"SepalLengthCm": 5.1, "SepalWidthCm": 3.5, "PetalLengthCm": 1.4, "PetalWidthCm": 0.2}
VIRGINICA = {"SepalLengthCm": 6.7, "SepalWidthCm": 3.0, "PetalLengthCm": 5.2, "PetalWidthCm": 2.3}


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "healthy"


def test_single_prediction(client):
    response = client.post("/predict", json=SETOSA)
    assert response.status_code == 200
    assert response.json["predictions"][0]["prediction"] == "Iris-setosa"


def test_batch_prediction(client):
    response = client.post("/predict", json={"instances": [SETOSA, VIRGINICA]})
    labels = [p["prediction"] for p in response.json["predictions"]]
    assert labels == ["Iris-setosa", "Iris-virginica"]


def test_missing_feature_returns_400(client):
    response = client.post("/predict", json={"SepalLengthCm": 5.1})
    assert response.status_code == 400
    assert "Missing features" in response.json["error"]


def test_metrics_endpoint_exposes_prometheus_metrics(client):
    client.post("/predict", json=SETOSA)
    text = client.get("/metrics").get_data(as_text=True)
    assert "model_predictions_total" in text
    assert "api_request_latency_seconds" in text
