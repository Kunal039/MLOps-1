"""
Experiment 14: Mini Project Phase-4 - Deployment (Flask REST API + Docker)
==========================================================================
Serves the latest registered model version from registry/.

    GET  /health        -> service status
    GET  /model-info    -> version, algorithm, metrics, features
    POST /predict       -> predicted consumption (kW) for the requested hour
    POST /forecast      -> recursive forecast for the next N hours (default 24)

Run locally :  python model_registry.py  &&  python app.py
Run Docker  :  docker build -t household-energy-api:1.0 .  &&  docker run -p 5000:5000 household-energy-api:1.0
"""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from features import HISTORY_HOURS, build_features

BASE_DIR = Path(__file__).resolve().parent
REGISTRY = BASE_DIR / "registry"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("energy-api")


def load_latest_model():
    version = json.loads((REGISTRY / "latest.json").read_text())["version"]
    model = joblib.load(REGISTRY / version / "model.joblib")
    metadata = json.loads((REGISTRY / version / "metadata.json").read_text())
    logger.info("Loaded model %s (%s)", version, metadata["model_name"])
    return model, metadata


model, metadata = load_latest_model()
FEATURES = metadata["features"]
app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(status="ok", model_version=metadata["version"])


@app.get("/model-info")
def model_info():
    return jsonify({k: metadata[k] for k in
                    ["version", "model_name", "params", "test_metrics", "baseline_test_rmse",
                     "features", "registered_at"]})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    try:
        X = build_features(payload, FEATURES)
    except (ValueError, TypeError) as err:
        logger.warning("Invalid request: %s", err)
        return jsonify(error=str(err)), 400
    prediction = max(float(model.predict(X)[0]), 0.0)
    logger.info("Prediction %.3f kW for %s", prediction, payload.get("timestamp"))
    return jsonify(timestamp=payload.get("timestamp"), predicted_kw=round(prediction, 4),
                   predicted_kwh_for_hour=round(prediction, 4), model_version=metadata["version"])


@app.post("/forecast")
def forecast():
    """Predict the next N hours, feeding each prediction back as the newest history value."""
    payload = request.get_json(silent=True) or {}
    hours = int(payload.get("hours", 24))
    if not 1 <= hours <= 168:
        return jsonify(error="'hours' must be between 1 and 168"), 400
    history = list(payload.get("recent_consumption", []))
    if len(history) < HISTORY_HOURS:
        return jsonify(error=f"'recent_consumption' must contain at least {HISTORY_HOURS} values"), 400

    start = pd.Timestamp(payload.get("timestamp", pd.Timestamp.now().floor("h")))
    results = []
    for h in range(hours):
        ts = start + pd.Timedelta(hours=h)
        step = {**payload, "timestamp": str(ts), "recent_consumption": history}
        value = max(float(model.predict(build_features(step, FEATURES))[0]), 0.0)
        results.append({"timestamp": str(ts), "predicted_kw": round(value, 4)})
        history = history[1:] + [value]
    total = sum(r["predicted_kw"] for r in results)
    return jsonify(model_version=metadata["version"], forecast=results, total_kwh=round(total, 3))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
