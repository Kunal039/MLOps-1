"""
Tests for the deployed API. Run:  python -m pytest test_api.py -v

The most important test checks there is no training-serving skew: features built by the
API from raw history must equal the features Phase-2 produced for the same hour.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import FEATURES, app
from features import build_features

PHASE2 = Path(__file__).resolve().parent.parent / "Experiment-12_MiniProject_Phase2_Preprocessing_Features" / "outputs"
TEST = pd.read_csv(PHASE2 / "test.csv", index_col=0, parse_dates=True)
ROW = 200                                   # predict this hour of the test set using the 168 hours before it


def make_payload(row=ROW):
    target_row = TEST.iloc[row]
    return {
        "timestamp": str(TEST.index[row]),
        "recent_consumption": TEST["Global_active_power"].iloc[row - 168: row].tolist(),
        "voltage_last_hour": target_row["Voltage_lag_1"],
        "reactive_power_last_hour": target_row["Global_reactive_power_lag_1"],
        "sub_metering_3_last_hour": target_row["Sub_metering_3_lag_1"],
    }


@pytest.fixture
def client():
    return app.test_client()


def test_features_match_training_pipeline():
    api_features = build_features(make_payload(), FEATURES).iloc[0].to_numpy(dtype=float)
    training_features = TEST.iloc[ROW][FEATURES].to_numpy(dtype=float)
    assert np.allclose(api_features, training_features, atol=1e-6)


def test_health(client):
    assert client.get("/health").json["status"] == "ok"


def test_model_info(client):
    info = client.get("/model-info").json
    assert info["version"].startswith("v") and len(info["features"]) == len(FEATURES)


def test_prediction_is_close_to_actual(client):
    response = client.post("/predict", json=make_payload())
    assert response.status_code == 200
    actual = TEST["Global_active_power"].iloc[ROW]
    assert abs(response.json["predicted_kw"] - actual) < 0.5


def test_forecast_24_hours(client):
    response = client.post("/forecast", json={**make_payload(), "hours": 24})
    assert response.status_code == 200
    assert len(response.json["forecast"]) == 24


def test_short_history_rejected(client):
    response = client.post("/predict", json={"recent_consumption": [1.0] * 10})
    assert response.status_code == 400


if __name__ == "__main__":
    print(json.dumps(make_payload(), indent=2)[:400], "...")
