"""
Experiment 10: Set up monitoring and logging for deployed ML models
-------------------------------------------------------------------
Aim:
    * Use Prometheus and Grafana to monitor the performance of the deployed ML model.
    * Implement logging to track predictions and model performance in real time.

`predict()` and `feedback()` wrap the model with monitoring. Prometheus metrics
(visualised in Grafana) are exposed with `start_metrics_server()` on http://localhost:8000/metrics:
    ml_predictions_total           - predictions per class (class distribution)
    ml_prediction_confidence       - confidence histogram (low confidence = model is unsure)
    ml_prediction_latency_seconds  - latency histogram (p50 / p95 / p99 in Grafana)
    ml_prediction_errors_total     - invalid inputs
    ml_feature_value               - latest value of each input feature
    ml_feature_drift_zscore        - drift of recent inputs vs the training data
    ml_live_accuracy               - accuracy on predictions whose true label came back (feedback)

Logging:
    logs/app.log              - human-readable log (rotating file): drift, low confidence, errors
    logs/predictions.jsonl    - one JSON line per prediction (input, output, confidence, latency)
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
from prometheus_client import Counter, Gauge, Histogram, start_http_server

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("ml-monitor")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
file_handler = RotatingFileHandler(LOG_DIR / "app.log", maxBytes=1_000_000, backupCount=3)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

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
LATENCY = Histogram("ml_prediction_latency_seconds", "Prediction latency",
                    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1])
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
feedback_window = deque(maxlen=200)      # 1 = correct, 0 = wrong


def start_metrics_server(port=8000):
    """Expose the metrics at http://localhost:<port>/metrics for Prometheus to scrape."""
    start_http_server(port)


def update_drift():
    window = pd.DataFrame(list(recent_inputs))
    for f in FEATURES:
        z = abs(window[f].mean() - baseline["mean"][f]) / baseline["std"][f]
        DRIFT.labels(f).set(round(z, 3))
        if z > DRIFT_THRESHOLD and len(window) >= 20:
            logger.warning("DATA DRIFT detected on %s (z=%.2f) - consider retraining", f, z)


def predict(record):
    """Predict one flower. Raises ValueError for invalid input. Returns request id, label, confidence."""
    missing = [f for f in FEATURES if f not in record]
    if missing:
        ERRORS.labels("missing_feature").inc()
        logger.error("Bad input - missing features %s", missing)
        raise ValueError(f"Missing features: {missing}")
    try:
        row = pd.DataFrame([{f: float(record[f]) for f in FEATURES}])
    except (TypeError, ValueError):
        ERRORS.labels("invalid_value").inc()
        logger.error("Bad input - non-numeric values %s", record)
        raise ValueError("All features must be numbers") from None

    start = time.perf_counter()
    probs = model.predict_proba(row)[0]
    label = model.classes_[int(np.argmax(probs))]
    confidence = float(probs.max())
    latency = time.perf_counter() - start

    # --- monitoring ---
    LATENCY.observe(latency)
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
        "latency_ms": round(latency * 1000, 3),
    }))
    if confidence < 0.6:
        logger.warning("Low-confidence prediction %s (%.2f) for request %s", label, confidence, request_id)
    return {"request_id": request_id, "prediction": label, "confidence": round(confidence, 4)}


def feedback(request_id, actual):
    """The true label arrives later -> real-time accuracy of the deployed model."""
    predicted = predictions_by_id.pop(request_id, None)
    if predicted is None:
        raise KeyError(f"Unknown request_id {request_id}")
    feedback_window.append(int(predicted == actual))
    accuracy = sum(feedback_window) / len(feedback_window)
    LIVE_ACCURACY.set(round(accuracy, 4))
    if len(feedback_window) >= 20 and accuracy < 0.8:
        logger.warning("Live accuracy dropped to %.2f - model performance degraded", accuracy)
    return round(accuracy, 4)
