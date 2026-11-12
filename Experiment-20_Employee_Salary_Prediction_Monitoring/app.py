"""
Experiment 20 - Step 2: Salary prediction service with monitoring and logging
    GET  /health
    POST /predict       {"experience_years": 5, "skills_count": 8, "certifications": 1, "performance_rating": 4,
                         "job_role": "Data Scientist", "city": "Bengaluru", "company_size": "Enterprise",
                         "education": "Master"}
    GET  /metrics       Prometheus metrics (requests, latency, predicted salary distribution, input values)
Every prediction is appended to logs/predictions.jsonl - monitor_drift.py uses it to detect drift.
"""

import json
import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, Response, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from generate_data import CITY, EDUCATION, ROLES, SIZE

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
model = joblib.load(BASE_DIR / "model" / "salary_model.joblib")
info = json.loads((BASE_DIR / "model" / "model_info.json").read_text())
FEATURES = info["features"]
ALLOWED = {"job_role": set(ROLES), "city": set(CITY), "company_size": set(SIZE), "education": set(EDUCATION)}
RANGES = {"experience_years": (0, 45), "skills_count": (0, 50), "certifications": (0, 30), "performance_rating": (1, 5)}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.FileHandler(LOG_DIR / "service.log"), logging.StreamHandler()])
log = logging.getLogger("salary-api")
prediction_log = (LOG_DIR / "predictions.jsonl").open("a", buffering=1)

REQUESTS = Counter("salary_requests_total", "Requests", ["endpoint", "status"])
LATENCY = Histogram("salary_request_latency_seconds", "Latency", ["endpoint"])
PREDICTED = Histogram("salary_predicted_lpa", "Predicted salary (LPA)", buckets=[3, 5, 8, 12, 16, 20, 30, 45, 70])
ROLE_COUNT = Counter("salary_requests_by_role_total", "Requests by job role", ["job_role"])
EXPERIENCE = Histogram("salary_input_experience_years", "Experience in requests", buckets=[1, 2, 4, 6, 8, 12, 20, 30])
VALIDATION_ERRORS = Counter("salary_validation_errors_total", "Rejected requests", ["field"])
MODEL_INFO = Gauge("salary_model_info", "Model in production", ["algorithm", "registry_version"])
MODEL_INFO.labels(info["algorithm"], str(info["registry_version"])).set(1)

app = Flask(__name__)


def validate(data):
    for field in FEATURES:
        if field not in data:
            return field, f"Missing field '{field}'"
    for field, allowed in ALLOWED.items():
        if data[field] not in allowed:
            return field, f"'{field}' must be one of {sorted(allowed)}"
    for field, (lo, hi) in RANGES.items():
        try:
            value = float(data[field])
        except (TypeError, ValueError):
            return field, f"'{field}' must be a number"
        if not lo <= value <= hi:
            return field, f"'{field}' must be between {lo} and {hi}"
    return None, None


@app.before_request
def _start():
    request.t0 = time.perf_counter()


@app.after_request
def _record(response):
    if request.path != "/metrics":
        LATENCY.labels(request.path).observe(time.perf_counter() - request.t0)
        REQUESTS.labels(request.path, response.status_code).inc()
    return response


@app.get("/health")
def health():
    return jsonify(status="ok", model=info)


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    field, error = validate(data)
    if error:
        VALIDATION_ERRORS.labels(field).inc()
        log.warning("Rejected request: %s", error)
        return jsonify(error=error), 400

    row = pd.DataFrame([{f: data[f] for f in FEATURES}])
    salary = float(model.predict(row)[0])
    PREDICTED.observe(salary)
    ROLE_COUNT.labels(data["job_role"]).inc()
    EXPERIENCE.observe(float(data["experience_years"]))
    prediction_log.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                     **{f: data[f] for f in FEATURES}, "predicted_salary_lpa": round(salary, 2),
                                     "model_version": info["registry_version"]}) + "\n")
    rmse = info["metrics"]["test_rmse"]
    return jsonify(predicted_salary_lpa=round(salary, 2),
                   expected_range_lpa=[round(max(salary - 1.96 * rmse, 0), 2), round(salary + 1.96 * rmse, 2)],
                   model=info["algorithm"], model_version=info["registry_version"])


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
