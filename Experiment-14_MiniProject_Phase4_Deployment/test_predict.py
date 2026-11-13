"""
Tests for the deployed prediction service. Run:  python -m pytest test_predict.py -v

The most important test checks there is no training-serving skew: features built at
prediction time from raw history must equal the features Phase-2 produced for the same hour.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features import build_features
from predict import FEATURES, forecast, metadata, predict_next_hour

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


def test_features_match_training_pipeline():
    serving_features = build_features(make_payload(), FEATURES).iloc[0].to_numpy(dtype=float)
    training_features = TEST.iloc[ROW][FEATURES].to_numpy(dtype=float)
    assert np.allclose(serving_features, training_features, atol=1e-6)


def test_registry_metadata():
    assert metadata["version"].startswith("v") and len(metadata["features"]) == len(FEATURES)


def test_prediction_is_close_to_actual():
    actual = TEST["Global_active_power"].iloc[ROW]
    assert abs(predict_next_hour(make_payload()) - actual) < 0.5


def test_forecast_24_hours():
    assert len(forecast(make_payload(), hours=24)) == 24


def test_short_history_rejected():
    with pytest.raises(ValueError):
        predict_next_hour({"recent_consumption": [1.0] * 10})
