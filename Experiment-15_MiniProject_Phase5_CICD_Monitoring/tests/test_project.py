"""CI tests for the mini project: data quality, model quality and the monitored API."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
PHASE2 = ROOT / "Experiment-12_MiniProject_Phase2_Preprocessing_Features" / "outputs"
PHASE3 = ROOT / "Experiment-13_MiniProject_Phase3_Model_Building" / "outputs" / "models"
sys.path.insert(0, str(HERE))


# ---------- data quality ----------
@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_processed_data_has_no_missing_values(split):
    df = pd.read_csv(PHASE2 / f"{split}.csv", index_col=0)
    assert not df.isna().any().any()


def test_time_split_has_no_overlap():
    train = pd.read_csv(PHASE2 / "train.csv", index_col=0, parse_dates=True)
    test = pd.read_csv(PHASE2 / "test.csv", index_col=0, parse_dates=True)
    assert train.index.max() < test.index.min()


def test_consumption_is_physically_plausible():
    df = pd.read_csv(PHASE2 / "train.csv", index_col=0)
    assert df["Global_active_power"].between(0, 11).all()


# ---------- model quality ----------
def test_model_beats_baseline():
    meta = json.loads((PHASE3 / "model_metadata.json").read_text())
    assert meta["test_metrics"]["RMSE"] < meta["baseline_test_rmse"]
    assert meta["test_metrics"]["R2"] > 0.8


# ---------- monitored API ----------
@pytest.fixture(scope="module")
def client():
    from monitored_app import app
    return app.test_client()


def test_metrics_endpoint(client):
    text = client.get("/metrics").get_data(as_text=True)
    assert "energy_live_mae_kw" in text and "energy_model_info" in text


def test_prediction_and_feedback_loop(client):
    test = pd.read_csv(PHASE2 / "test.csv", index_col=0, parse_dates=True)
    i = 300
    payload = {"timestamp": str(test.index[i]),
               "recent_consumption": test["Global_active_power"].iloc[i - 168:i].tolist()}
    assert client.post("/predict", json=payload).status_code == 200
    fb = client.post("/actual", json={"timestamp": str(test.index[i]),
                                      "actual_kw": float(test["Global_active_power"].iloc[i])})
    assert fb.status_code == 200 and fb.json["abs_error"] >= 0
