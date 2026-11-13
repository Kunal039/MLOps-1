"""
Experiment 14: Mini Project Phase-4 - Deployment (batch prediction service + Docker)
====================================================================================
Loads the latest registered model version from registry/ and predicts household consumption.

    predict_next_hour(payload)      -> predicted kW for the requested hour
    forecast(payload, hours=24)     -> recursive forecast for the next N hours

Run locally :  python model_registry.py  &&  python predict.py
Run Docker  :  docker build -t household-energy-model:1.0 .  &&  docker run --rm household-energy-model:1.0

payload = {"timestamp": "2009-06-20 18:00:00",
           "recent_consumption": [... at least 168 hourly kW values, oldest first ...],
           "voltage_last_hour": 239.4, "reactive_power_last_hour": 0.15, "sub_metering_3_last_hour": 7.2}
"""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from features import HISTORY_HOURS, build_features

BASE_DIR = Path(__file__).resolve().parent
REGISTRY = BASE_DIR / "registry"
PHASE2 = BASE_DIR.parent / "Experiment-12_MiniProject_Phase2_Preprocessing_Features" / "outputs"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("energy-model")


def load_latest_model():
    version = json.loads((REGISTRY / "latest.json").read_text())["version"]
    model = joblib.load(REGISTRY / version / "model.joblib")
    metadata = json.loads((REGISTRY / version / "metadata.json").read_text())
    logger.info("Loaded model %s (%s)", version, metadata["model_name"])
    return model, metadata


model, metadata = load_latest_model()
FEATURES = metadata["features"]


def predict_next_hour(payload):
    """Predicted consumption (kW) for payload['timestamp']. Raises ValueError for invalid input."""
    X = build_features(payload, FEATURES)
    return max(float(model.predict(X)[0]), 0.0)


def forecast(payload, hours=24):
    """Predict the next N hours, feeding each prediction back as the newest history value."""
    if not 1 <= hours <= 168:
        raise ValueError("'hours' must be between 1 and 168")
    history = list(payload.get("recent_consumption") or [])
    if len(history) < HISTORY_HOURS:
        raise ValueError(f"'recent_consumption' must contain at least {HISTORY_HOURS} values")
    start = pd.Timestamp(payload.get("timestamp", pd.Timestamp.now().floor("h")))
    results = []
    for h in range(hours):
        ts = start + pd.Timedelta(hours=h)
        value = predict_next_hour({**payload, "timestamp": str(ts), "recent_consumption": history})
        results.append({"timestamp": str(ts), "predicted_kw": round(value, 4)})
        history = history[1:] + [value]
    return results


if __name__ == "__main__":
    print(f"Model {metadata['version']}: {metadata['model_name']}, test RMSE {metadata['test_metrics']['RMSE']} kW\n")
    if (PHASE2 / "test.csv").exists():
        # a real week of history from the Phase-2 test set
        test = pd.read_csv(PHASE2 / "test.csv", index_col=0, parse_dates=True)
        i = 200
        payload = {
            "timestamp": str(test.index[i]),
            "recent_consumption": test["Global_active_power"].iloc[i - 168:i].tolist(),
            "voltage_last_hour": float(test["Voltage_lag_1"].iloc[i]),
            "reactive_power_last_hour": float(test["Global_reactive_power_lag_1"].iloc[i]),
            "sub_metering_3_last_hour": float(test["Sub_metering_3_lag_1"].iloc[i]),
        }
        actual = f", actual {test['Global_active_power'].iloc[i]:.3f} kW"
    else:
        # inside the Docker image there is no Phase-2 data: use a typical week (low at night, evening peak)
        day = [0.45] * 6 + [0.9, 1.2, 1.0] + [0.5] * 9 + [1.2, 1.8, 2.1, 1.9, 1.4, 0.8]
        payload = {"timestamp": "2009-06-20 18:00:00", "recent_consumption": day * 7}
        actual = ""
    predicted = predict_next_hour(payload)
    print(f"Next hour {payload['timestamp']}: predicted {predicted:.3f} kW{actual}")

    day = forecast(payload, hours=24)
    print(f"\n24-hour forecast from {day[0]['timestamp']}: total {sum(r['predicted_kw'] for r in day):.2f} kWh")
    for r in day[::4]:
        print(f"  {r['timestamp']}  {r['predicted_kw']:.3f} kW")
