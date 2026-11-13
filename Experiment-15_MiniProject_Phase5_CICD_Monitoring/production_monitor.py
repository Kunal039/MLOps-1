"""
Experiment 15: Mini Project Phase-5 - Monitoring and logging of the deployed model
==================================================================================
Wraps the Phase-4 predictor (predict.py) with Prometheus metrics, a prediction log and
live error tracking once the actual meter reading for an hour is known:

    predict(payload)                 -> predicted kW (logged + counted)
    record_actual(timestamp, kw)     -> absolute error, rolling 24h MAE, retrain flag
    start_metrics_server(port)       -> expose metrics for Prometheus at http://localhost:<port>/metrics

Metrics: energy_predictions_total, energy_predicted_kw, energy_prediction_latency_seconds,
         energy_abs_error_kw, energy_live_mae_kw (rolling 24h), energy_model_info
Logs   : logs/service.log, logs/predictions.jsonl
"""

import json
import logging
import sys
import time
from collections import deque
from pathlib import Path

from prometheus_client import Counter, Gauge, Histogram, start_http_server

BASE_DIR = Path(__file__).resolve().parent
PHASE4_DIR = BASE_DIR.parent / "Experiment-14_MiniProject_Phase4_Deployment"
sys.path.insert(0, str(PHASE4_DIR))
import predict as phase4  # noqa: E402  (Phase-4 predictor and model metadata)

metadata = phase4.metadata
logging.getLogger("energy-model").setLevel(logging.WARNING)   # keep the console readable

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("energy-monitor")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(LOG_DIR / "service.log")
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(handler)
prediction_log = (LOG_DIR / "predictions.jsonl").open("a", buffering=1)

PREDICTIONS = Counter("energy_predictions_total", "Predictions served")
PREDICTED_KW = Gauge("energy_predicted_kw", "Last predicted consumption (kW)")
LATENCY = Histogram("energy_prediction_latency_seconds", "Prediction latency")
ABS_ERROR = Histogram("energy_abs_error_kw", "Absolute error once actual is known",
                      buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 1, 2])
LIVE_MAE = Gauge("energy_live_mae_kw", "Rolling 24-hour MAE of the deployed model")
MODEL_INFO = Gauge("energy_model_info", "Deployed model version", ["version", "algorithm"])
MODEL_INFO.labels(metadata["version"], metadata["model_name"]).set(1)

MAE_ALERT_THRESHOLD = 1.5 * metadata["test_metrics"]["MAE"]   # 50% worse than offline test MAE
pending = {}                    # timestamp -> predicted kW, waiting for the actual value
errors = deque(maxlen=24)       # last 24 hourly absolute errors


def start_metrics_server(port=8000):
    start_http_server(port)


def predict(payload):
    start = time.perf_counter()
    predicted = round(phase4.predict_next_hour(payload), 4)
    LATENCY.observe(time.perf_counter() - start)
    PREDICTIONS.inc()
    PREDICTED_KW.set(predicted)
    pending[payload["timestamp"]] = predicted
    prediction_log.write(json.dumps({"logged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                     "timestamp": payload["timestamp"], "predicted_kw": predicted,
                                     "model_version": metadata["version"]}) + "\n")
    return predicted


def record_actual(timestamp, actual_kw):
    predicted = pending.pop(timestamp, None)
    if predicted is None:
        raise KeyError(f"No prediction waiting for {timestamp}")
    error = abs(predicted - float(actual_kw))
    errors.append(error)
    ABS_ERROR.observe(error)
    mae = sum(errors) / len(errors)
    LIVE_MAE.set(round(mae, 4))
    prediction_log.write(json.dumps({"timestamp": timestamp, "predicted_kw": predicted,
                                     "actual_kw": actual_kw, "abs_error": round(error, 4)}) + "\n")
    degraded = len(errors) == errors.maxlen and mae > MAE_ALERT_THRESHOLD
    if degraded:
        logger.warning("Model degraded: 24h MAE %.3f kW > threshold %.3f kW - retraining required",
                       mae, MAE_ALERT_THRESHOLD)
    return {"abs_error": round(error, 4), "live_mae_24h": round(mae, 4), "retrain_required": degraded}
