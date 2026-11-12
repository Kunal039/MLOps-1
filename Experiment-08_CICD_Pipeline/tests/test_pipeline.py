"""
Tests executed automatically by the CI pipeline on every push / pull request.
    * data tests   - is the input data valid?
    * model tests  - does training produce a good enough model?
    * API tests    - does the deployed service respond correctly?
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

from data import load_data, split_data, validate_data  # noqa: E402
from train import build_model  # noqa: E402

PARAMS = json.loads((PROJECT_DIR / "params.json").read_text())


# ---------------- Data tests ----------------
def test_data_is_valid():
    X, y = load_data()
    assert validate_data(X, y)


def test_validation_catches_missing_values():
    X, y = load_data()
    X.iloc[0, 0] = None
    with pytest.raises(AssertionError):
        validate_data(X, y)


def test_split_is_stratified():
    X, y = load_data()
    _, _, y_train, y_test = split_data(X, y)
    assert abs(y_train.mean() - y_test.mean()) < 0.1


# ---------------- Model tests ----------------
def test_model_beats_quality_gate():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = build_model(PARAMS).fit(X_train, y_train)
    assert model.score(X_test, y_test) >= PARAMS["quality_gate"]["min_accuracy"]


def test_model_is_deterministic():
    X, y = load_data()
    X_train, X_test, y_train, _ = split_data(X, y)
    p1 = build_model(PARAMS).fit(X_train, y_train).predict(X_test)
    p2 = build_model(PARAMS).fit(X_train, y_train).predict(X_test)
    assert (p1 == p2).all()


# ---------------- API tests ----------------
@pytest.fixture(scope="module")
def client():
    if not (PROJECT_DIR / "model" / "model.joblib").exists():
        subprocess.run([sys.executable, str(PROJECT_DIR / "src" / "train.py")], check=True)
    from app import app
    return app.test_client()


def test_health_endpoint(client):
    assert client.get("/health").status_code == 200


def test_predict_endpoint(client):
    X, _ = load_data()
    response = client.post("/predict", json={"features": X.iloc[0].tolist()})
    assert response.status_code == 200
    assert response.json["prediction"] == "class_0"


def test_predict_rejects_bad_input(client):
    assert client.post("/predict", json={"features": [1, 2]}).status_code == 400
