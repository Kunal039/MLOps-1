"""
Experiment 7: Create a REST API to serve an ML model using Flask and
              set up Prometheus to collect metrics from the deployed model.
------------------------------------------------------------------------------
Endpoints
    GET  /                -> API information
    GET  /health          -> health check (used by load balancers / Docker / Kubernetes)
    GET  /model-info      -> model metadata (name, version, accuracy)
    POST /predict         -> receive input data, make prediction, return result
                             single : {"SepalLengthCm": 5.1, "SepalWidthCm": 3.5, "PetalLengthCm": 1.4, "PetalWidthCm": 0.2}
                             batch  : {"instances": [ {...}, {...} ]}
    GET  /metrics         -> Prometheus metrics (scraped by Prometheus every few seconds)

Run:  python train_model.py   then   python app.py
"""

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, Response, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

app = Flask(__name__)
model = joblib.load(MODEL_DIR / "model.joblib")
metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
FEATURES = metadata["features"]

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter("api_requests_total", "Total HTTP requests", ["endpoint", "method", "status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency in seconds", ["endpoint"])
PREDICTION_COUNT = Counter("model_predictions_total", "Predictions made per class", ["predicted_class"])
PREDICTION_CONFIDENCE = Histogram(
    "model_prediction_confidence", "Confidence of predictions",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)
MODEL_INFO = Gauge("model_info", "Model metadata", ["name", "version"])
MODEL_INFO.labels(metadata["name"], metadata["version"]).set(1)
MODEL_ACCURACY = Gauge("model_test_accuracy", "Accuracy of the deployed model on the test set")
MODEL_ACCURACY.set(metadata["test_accuracy"])


@app.before_request
def start_timer():
    request.start_time = time.perf_counter()


@app.after_request
def record_metrics(response):
    if request.path != "/metrics":
        REQUEST_LATENCY.labels(request.path).observe(time.perf_counter() - request.start_time)
        REQUEST_COUNT.labels(request.path, request.method, response.status_code).inc()
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return jsonify(
        message="Iris classification API",
        endpoints={"GET /health": "health check", "GET /model-info": "model metadata",
                   "POST /predict": "make predictions", "GET /metrics": "Prometheus metrics"},
        example_input={f: 1.0 for f in FEATURES},
    )


@app.get("/health")
def health():
    return jsonify(status="healthy", model_loaded=model is not None)


@app.get("/model-info")
def model_info():
    return jsonify(metadata)


def validate(record):
    """Return an error message if the input record is invalid, else None."""
    missing = [f for f in FEATURES if f not in record]
    if missing:
        return f"Missing features: {missing}"
    for f in FEATURES:
        if not isinstance(record[f], (int, float)) or record[f] < 0:
            return f"Feature '{f}' must be a non-negative number"
    return None


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify(error="Request body must be JSON"), 400

    records = payload.get("instances", [payload]) if isinstance(payload, dict) else payload
    for i, record in enumerate(records):
        error = validate(record)
        if error:
            return jsonify(error=f"Instance {i}: {error}"), 400

    df = pd.DataFrame(records)[FEATURES]
    labels = model.predict(df)
    probabilities = model.predict_proba(df)

    results = []
    for label, probs in zip(labels, probabilities):
        confidence = float(probs.max())
        PREDICTION_COUNT.labels(label).inc()
        PREDICTION_CONFIDENCE.observe(confidence)
        results.append({
            "prediction": label,
            "confidence": round(confidence, 4),
            "probabilities": {c: round(float(p), 4) for c, p in zip(model.classes_, probs)},
        })
    return jsonify(model_version=metadata["version"], predictions=results)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
