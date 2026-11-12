"""
Experiment 10: Set up monitoring and logging for deployed ML models
-------------------------------------------------------------------
Aim:
    * Use Prometheus and Grafana to monitor the performance of the deployed ML model.
    * Implement logging to track predictions and model performance in real time.

Monitoring (Prometheus metrics at GET /metrics, visualised in Grafana):
    ml_requests_total              - requests by endpoint and HTTP status
    ml_request_latency_seconds     - latency histogram (p50 / p95 / p99 in Grafana)
    ml_requests_in_progress        - concurrent requests
    ml_predictions_total           - predictions per class (class distribution)
    ml_prediction_confidence       - confidence histogram (low confidence = model is unsure)
    ml_prediction_errors_total     - invalid inputs / failures
    ml_feature_value               - latest value of each input feature
    ml_feature_drift_zscore        - drift of recent inputs vs the training data
    ml_live_accuracy               - accuracy on requests where the true label is sent back (feedback)

Logging:
    logs/app.log              - human-readable application log (rotating file)
    logs/predictions.jsonl    - one JSON line per prediction (input, output, confidence, latency)

Endpoints:  GET /health   POST /predict   POST /feedback   GET /metrics
"""

import json
import logging
import time
import uuid
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("ml-service")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
file_handler = RotatingFileHandler(LOG_DIR / "app.log", maxBytes=1_000_000, backupCount=3)
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

prediction_logger = logging.getLogger("predictions")
prediction_logger.setLevel(logging.INFO)
prediction_logger.addHandler(logging.FileHandler(LOG_DIR / "predictions.jsonl"))
prediction_logger.propagate = False

# ---------------------------------------------------------------------------
# Model + baseline
# ---------------------------------------------------------------------------
model = joblib.load(BASE_DIR / "model" / "model.joblib")
baseline = json.loads((BASE_DIR / "model" / "baseline.json").read_text())
FEATURES = baseline["features"]
logger.info("Model v%s loaded (training accuracy %.4f)", baseline["version"], baseline["accuracy"])

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUESTS = Counter("ml_requests_total", "HTTP requests", ["endpoint", "status"])
LATENCY = Histogram("ml_request_latency_seconds", "Request latency", ["endpoint"],
                    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1])
IN_PROGRESS = Gauge("ml_requests_in_progress", "Requests being processed")
PREDICTIONS = Counter("ml_predictions_total", "Predictions by class", ["predicted_class"])
CONFIDENCE = Histogram("ml_prediction_confidence", "Prediction confidence",
                       buckets=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
ERRORS = Counter("ml_prediction_errors_total", "Prediction errors", ["reason"])
FEATURE_VALUE = Gauge("ml_feature_value", "Latest input feature value", ["feature"])
DRIFT = Gauge("ml_feature_drift_zscore", "|mean(recent) - mean(train)| / std(train)", ["feature"])
LIVE_ACCURACY = Gauge("ml_live_accuracy", "Accuracy on feedback received from users")
MODEL_VERSION = Gauge("ml_model_info", "Deployed model", ["version"])
MODEL_VERSION.labels(baseline["version"]).set(1)

DRIFT_THRESHOLD = 1.0   # alert when the recent mean moves more than 1 std from training
recent_inputs = deque(maxlen=100)        # sliding window for drift detection
predictions_by_id = {}                   # request_id -> predicted label (for feedback)
feedback = deque(maxlen=200)             # 1 = correct, 0 = wrong

app = Flask(__name__)


def update_drift():
    window = pd.DataFrame(list(recent_inputs))
    for f in FEATURES:
        z = abs(window[f].mean() - baseline["mean"][f]) / baseline["std"][f]
        DRIFT.labels(f).set(round(z, 3))
        if z > DRIFT_THRESHOLD and len(window) >= 20:
            logger.warning("DATA DRIFT detected on %s (z=%.2f) - consider retraining", f, z)


@app.before_request
def before():
    request.start = time.perf_counter()
    IN_PROGRESS.inc()


@app.after_request
def after(response):
    IN_PROGRESS.dec()
    if request.path != "/metrics":
        LATENCY.labels(request.path).observe(time.perf_counter() - request.start)
        REQUESTS.labels(request.path, response.status_code).inc()
    return response


@app.get("/health")
def health():
    return jsonify(status="ok", model_version=baseline["version"])


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    missing = [f for f in FEATURES if f not in data]
    if missing:
        ERRORS.labels("missing_feature").inc()
        logger.error("Bad request - missing features %s", missing)
        return jsonify(error=f"Missing features: {missing}"), 400

    try:
        row = pd.DataFrame([{f: float(data[f]) for f in FEATURES}])
    except (TypeError, ValueError):
        ERRORS.labels("invalid_value").inc()
        logger.error("Bad request - non-numeric input %s", data)
        return jsonify(error="All features must be numbers"), 400

    start = time.perf_counter()
    probs = model.predict_proba(row)[0]
    label = model.classes_[int(np.argmax(probs))]
    confidence = float(probs.max())
    latency_ms = (time.perf_counter() - start) * 1000

    # --- monitoring ---
    PREDICTIONS.labels(label).inc()
    CONFIDENCE.observe(confidence)
    for f in FEATURES:
        FEATURE_VALUE.labels(f).set(row[f].iloc[0])
    recent_inputs.append(row.iloc[0].to_dict())
    update_drift()

    # --- logging ---
    request_id = str(uuid.uuid4())[:8]
    predictions_by_id[request_id] = label
    prediction_logger.info(json.dumps({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "request_id": request_id,
        "model_version": baseline["version"],
        "input": row.iloc[0].to_dict(),
        "prediction": label,
        "confidence": round(confidence, 4),
        "latency_ms": round(latency_ms, 3),
    }))
    if confidence < 0.6:
        logger.warning("Low-confidence prediction %s (%.2f) for request %s", label, confidence, request_id)

    return jsonify(request_id=request_id, prediction=label, confidence=round(confidence, 4))


@app.post("/feedback")
def receive_feedback():
    """Users send the true label later -> real-time accuracy of the deployed model."""
    data = request.get_json(silent=True) or {}
    predicted = predictions_by_id.get(data.get("request_id"))
    if predicted is None:
        return jsonify(error="Unknown request_id"), 404
    feedback.append(int(predicted == data.get("actual")))
    accuracy = sum(feedback) / len(feedback)
    LIVE_ACCURACY.set(round(accuracy, 4))
    if len(feedback) >= 20 and accuracy < 0.8:
        logger.warning("Live accuracy dropped to %.2f - model performance degraded", accuracy)
    return jsonify(live_accuracy=round(accuracy, 4), feedback_count=len(feedback))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
