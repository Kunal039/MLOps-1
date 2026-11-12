"""
Experiment 15: Mini Project Phase-5 - Monitoring and logging of the deployed model
==================================================================================
Wraps the Phase-4 API with Prometheus metrics, prediction logging and a feedback
endpoint that receives the actual meter reading once the hour is over:

    POST /predict, /forecast   (from Phase-4)
    POST /actual   {"timestamp": "...", "actual_kw": 0.83}  -> live error tracking
    GET  /metrics  Prometheus metrics

Metrics: energy_predictions_total, energy_predicted_kw, energy_request_latency_seconds,
         energy_live_mae_kw (rolling 24h), energy_abs_error_kw, energy_requests_total
Logs   : logs/service.log, logs/predictions.jsonl
"""

import json
import logging
import sys
import time
from collections import deque
from pathlib import Path

from flask import Response, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

BASE_DIR = Path(__file__).resolve().parent
PHASE4_DIR = BASE_DIR.parent / "Experiment-14_MiniProject_Phase4_Deployment"
sys.path.insert(0, str(PHASE4_DIR))
from app import app, metadata  # noqa: E402  (Phase-4 Flask app and model metadata)

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("energy-monitor")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(LOG_DIR / "service.log")
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(handler)
prediction_log = (LOG_DIR / "predictions.jsonl").open("a", buffering=1)

REQUESTS = Counter("energy_requests_total", "Requests", ["endpoint", "status"])
LATENCY = Histogram("energy_request_latency_seconds", "Latency", ["endpoint"])
PREDICTIONS = Counter("energy_predictions_total", "Predictions served")
PREDICTED_KW = Gauge("energy_predicted_kw", "Last predicted consumption (kW)")
ABS_ERROR = Histogram("energy_abs_error_kw", "Absolute error once actual is known",
                      buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 1, 2])
LIVE_MAE = Gauge("energy_live_mae_kw", "Rolling 24-hour MAE of the deployed model")
MODEL_INFO = Gauge("energy_model_info", "Deployed model version", ["version", "algorithm"])
MODEL_INFO.labels(metadata["version"], metadata["model_name"]).set(1)

MAE_ALERT_THRESHOLD = 1.5 * metadata["test_metrics"]["MAE"]   # 50% worse than offline test MAE
pending = {}                    # timestamp -> predicted kW, waiting for the actual value
errors = deque(maxlen=24)       # last 24 hourly absolute errors


@app.before_request
def _start():
    request.t0 = time.perf_counter()


@app.after_request
def _record(response):
    if request.path != "/metrics":
        LATENCY.labels(request.path).observe(time.perf_counter() - request.t0)
        REQUESTS.labels(request.path, response.status_code).inc()
    if request.path == "/predict" and response.status_code == 200:
        body = response.get_json()
        PREDICTIONS.inc()
        PREDICTED_KW.set(body["predicted_kw"])
        pending[body["timestamp"]] = body["predicted_kw"]
        prediction_log.write(json.dumps({"logged_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **body}) + "\n")
    return response


@app.post("/actual")
def actual():
    data = request.get_json(silent=True) or {}
    predicted = pending.pop(data.get("timestamp"), None)
    if predicted is None or "actual_kw" not in data:
        return jsonify(error="Unknown timestamp or missing actual_kw"), 404
    error = abs(predicted - float(data["actual_kw"]))
    errors.append(error)
    ABS_ERROR.observe(error)
    mae = sum(errors) / len(errors)
    LIVE_MAE.set(round(mae, 4))
    prediction_log.write(json.dumps({"timestamp": data["timestamp"], "predicted_kw": predicted,
                                     "actual_kw": data["actual_kw"], "abs_error": round(error, 4)}) + "\n")
    degraded = len(errors) == errors.maxlen and mae > MAE_ALERT_THRESHOLD
    if degraded:
        logger.warning("Model degraded: 24h MAE %.3f kW > threshold %.3f kW - retraining required",
                       mae, MAE_ALERT_THRESHOLD)
    return jsonify(abs_error=round(error, 4), live_mae_24h=round(mae, 4), retrain_required=degraded)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
