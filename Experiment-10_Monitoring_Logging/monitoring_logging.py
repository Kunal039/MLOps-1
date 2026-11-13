"""
Experiment 10: Set up monitoring and logging for deployed ML models
-------------------------------------------------------------------
Aim:
    * Use Prometheus and Grafana to monitor the performance of the deployed ML model.
    * Implement logging to track predictions and model performance in real time.

Steps (commands):
    python monitoring_logging.py                 -> train + simulate traffic + analyze logs + write the
                                                    Prometheus/Grafana stack (whole experiment)
    python monitoring_logging.py train           -> train the model and save training-data statistics (drift baseline)
    python monitoring_logging.py simulate        -> send normal, invalid and drifted traffic through the monitored model
    python monitoring_logging.py simulate --serve [--delay 0.2]
                                                 -> also expose metrics on :8000 and keep simulating
                                                    (used by docker-compose so Prometheus can scrape it)
    python monitoring_logging.py analyze         -> summary + plots of logs/predictions.jsonl
    python monitoring_logging.py stack           -> write outputs/monitoring-stack/ (model service + Prometheus +
                                                    Grafana, start with `docker compose up -d --build`)

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

import argparse
import json
import logging
import random
import shutil
import time
import uuid
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
LOG_DIR = BASE_DIR / "logs"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR.mkdir(exist_ok=True)
FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

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

DRIFT_THRESHOLD = 1.0   # alert when the recent mean moves more than 1 std from training
recent_inputs = deque(maxlen=100)        # sliding window for drift detection
predictions_by_id = {}                   # request_id -> predicted label (for feedback)
feedback_window = deque(maxlen=200)      # 1 = correct, 0 = wrong
model, baseline = None, None


# ---------------------------------------------------------------------------
# Step 1: train the model + save the baseline for drift detection
# ---------------------------------------------------------------------------
def train():
    df = pd.read_csv(BASE_DIR / "data" / "Iris.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["Species"], test_size=0.2, random_state=42, stratify=df["Species"]
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    accuracy = accuracy_score(y_test, rf.predict(X_test))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(rf, MODEL_DIR / "model.joblib")
    stats = {
        "version": "1.0.0",
        "accuracy": round(accuracy, 4),
        "features": FEATURES,
        "mean": X_train.mean().round(4).to_dict(),
        "std": X_train.std().round(4).to_dict(),
    }
    (MODEL_DIR / "baseline.json").write_text(json.dumps(stats, indent=2))
    print(f"Model trained (accuracy {accuracy:.4f}); baseline statistics saved for drift detection.")


# ---------------------------------------------------------------------------
# Step 2: the monitored model
# ---------------------------------------------------------------------------
def load_model():
    global model, baseline
    model = joblib.load(MODEL_DIR / "model.joblib")
    baseline = json.loads((MODEL_DIR / "baseline.json").read_text())
    MODEL_VERSION.labels(baseline["version"]).set(1)
    logger.info("Model v%s loaded (training accuracy %.4f)", baseline["version"], baseline["accuracy"])


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


# ---------------------------------------------------------------------------
# Step 3: simulate production traffic
#   Phase 1 (normal)  : inputs look like the training data        -> high accuracy, no drift
#   Phase 2 (drift)   : sensor fault adds +2.5 cm to petal length -> drift alert, accuracy drops
#   Some invalid inputs are also sent to exercise the error metrics.
# ---------------------------------------------------------------------------
def run_phase(data, name, petal_shift, n, delay):
    correct = total = 0
    live_accuracy = None
    for _ in range(n):
        row = data.sample(1).iloc[0]
        record = {f: round(float(row[f]) + random.gauss(0, 0.1), 2) for f in FEATURES}
        record["PetalLengthCm"] = round(record["PetalLengthCm"] + petal_shift, 2)
        result = predict(record)
        live_accuracy = feedback(result["request_id"], row["Species"])   # true label arrives later
        correct += result["prediction"] == row["Species"]
        total += 1
        time.sleep(delay)
    print(f"{name:25s} predictions={total}  accuracy={correct / total:.2%}  live_accuracy_metric={live_accuracy}")


def simulate(n=150, delay=0.0, serve=False):
    load_model()
    data = pd.read_csv(BASE_DIR / "data" / "Iris.csv")

    def run_once():
        run_phase(data, "Phase 1 - normal traffic", 0.0, n, delay)
        for bad in [{"SepalLengthCm": 5.0}, {f: "abc" for f in FEATURES}]:
            try:
                predict(bad)
            except ValueError as err:
                print("Invalid input rejected ->", err)
        run_phase(data, "Phase 2 - drifted data", 2.5, n, delay)

    if serve:
        start_metrics_server(8000)
        print("Prometheus metrics at http://localhost:8000/metrics (Ctrl+C to stop)")
        while True:
            run_once()
    run_once()
    print("\nDone. Check logs/app.log for drift warnings, logs/predictions.jsonl for the prediction log.")
    print("Run 'simulate --serve' (or docker compose up) to watch the metrics in Prometheus/Grafana.")


# ---------------------------------------------------------------------------
# Step 4: offline analysis of the prediction log
# ---------------------------------------------------------------------------
def analyze():
    OUTPUT_DIR.mkdir(exist_ok=True)
    records = [json.loads(line) for line in (LOG_DIR / "predictions.jsonl").read_text().splitlines() if line.strip()]
    df = pd.json_normalize(records)
    print(f"Predictions logged: {len(df)}")
    print("\nClass distribution:\n", df["prediction"].value_counts())
    print("\nConfidence: mean={:.3f}, min={:.3f}".format(df["confidence"].mean(), df["confidence"].min()))
    print("Low-confidence predictions (<0.6):", (df["confidence"] < 0.6).sum())
    print("Latency (ms): p50={:.2f}, p95={:.2f}".format(df["latency_ms"].median(), df["latency_ms"].quantile(0.95)))

    stats = json.loads((MODEL_DIR / "baseline.json").read_text())
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(df.index, df["confidence"].rolling(20).mean())
    axes[0].set_ylabel("Confidence (rolling 20)")
    axes[0].set_title("Model confidence over time")

    col = "input.PetalLengthCm"
    axes[1].plot(df.index, df[col].rolling(20).mean(), label="Recent mean")
    axes[1].axhline(stats["mean"]["PetalLengthCm"], color="red", linestyle="--", label="Training mean")
    axes[1].set_ylabel("Petal length")
    axes[1].set_title("Input drift on PetalLengthCm")
    axes[1].legend()

    for cls in df["prediction"].unique():
        axes[2].plot(df.index, (df["prediction"] == cls).rolling(30).mean(), label=cls)
    axes[2].set_ylabel("Share of predictions")
    axes[2].set_xlabel("Request number")
    axes[2].set_title("Prediction class distribution (rolling 30)")
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "monitoring_report.png")
    print("\nPlot saved to outputs/monitoring_report.png")


# ---------------------------------------------------------------------------
# Step 5: Prometheus + Grafana stack (docker compose)
# ---------------------------------------------------------------------------
STACK_FILES = {
    "Dockerfile": """\
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY data/ data/
COPY monitoring_logging.py ./
RUN python monitoring_logging.py train
EXPOSE 8000
# keep simulating traffic and expose Prometheus metrics on port 8000
CMD ["python", "monitoring_logging.py", "simulate", "--serve", "--delay", "0.2"]
""",
    "requirements.txt": """\
pandas>=2.0
numpy>=1.26
scikit-learn>=1.4
joblib>=1.3
prometheus-client>=0.20
matplotlib>=3.8
""",
    "docker-compose.yml": """\
# Experiment 10 - ML service + Prometheus + Grafana
#   docker compose up -d --build
#   ML metrics : http://localhost:8000/metrics
#   Prometheus : http://localhost:9090   (Status -> Targets, Alerts)
#   Grafana    : http://localhost:3000   (admin / admin) -> dashboard "ML Model Monitoring"
services:
  ml-service:
    build: .
    ports: ["8000:8000"]
    volumes:
      - ./logs:/app/logs          # prediction logs are kept on the host

  prometheus:
    image: prom/prometheus:v2.53.0
    ports: ["9090:9090"]
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/alert_rules.yml:/etc/prometheus/alert_rules.yml
    depends_on: [ml-service]

  grafana:
    image: grafana/grafana:11.1.0
    ports: ["3000:3000"]
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
    depends_on: [prometheus]
""",
    "monitoring/prometheus.yml": """\
global:
  scrape_interval: 5s
  evaluation_interval: 5s

rule_files:
  - /etc/prometheus/alert_rules.yml

scrape_configs:
  - job_name: ml-service
    static_configs:
      - targets: ["ml-service:8000"]   # use "host.docker.internal:8000" if "monitoring_logging.py simulate --serve" runs outside Docker
""",
    "monitoring/alert_rules.yml": """\
groups:
  - name: ml-model-alerts
    rules:
      - alert: DataDriftDetected
        expr: max(ml_feature_drift_zscore) > 1
        for: 30s
        labels: {severity: warning}
        annotations:
          summary: "Input data drift detected (z-score > 1) - consider retraining"

      - alert: LiveAccuracyLow
        expr: ml_live_accuracy < 0.8
        for: 1m
        labels: {severity: critical}
        annotations:
          summary: "Live model accuracy below 80%"

      - alert: HighLatency
        expr: histogram_quantile(0.95, sum(rate(ml_prediction_latency_seconds_bucket[1m])) by (le)) > 0.5
        for: 1m
        labels: {severity: warning}
        annotations:
          summary: "95th percentile latency above 500 ms"

      - alert: HighErrorRate
        expr: sum(rate(ml_prediction_errors_total[1m])) > 0.1
        for: 1m
        labels: {severity: warning}
        annotations:
          summary: "More than 0.1 invalid inputs per second"
""",
    "monitoring/grafana/provisioning/datasources/datasource.yml": """\
apiVersion: 1
datasources:
  - name: Prometheus
    uid: prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
""",
    "monitoring/grafana/provisioning/dashboards/dashboard.yml": """\
apiVersion: 1
providers:
  - name: ml-dashboards
    folder: MLOps
    type: file
    options:
      path: /var/lib/grafana/dashboards
""",
}

DASHBOARD_PANELS = [   # (title, panel type, PromQL query)
    ("Predictions per second", "timeseries", "sum(rate(ml_predictions_total[1m])) by (predicted_class)"),
    ("Latency p95 (s)", "timeseries",
     "histogram_quantile(0.95, sum(rate(ml_prediction_latency_seconds_bucket[1m])) by (le))"),
    ("Live accuracy", "gauge", "ml_live_accuracy"),
    ("Predictions by class", "piechart", "sum(ml_predictions_total) by (predicted_class)"),
    ("Feature drift (z-score)", "timeseries", "ml_feature_drift_zscore"),
    ("Average confidence", "timeseries",
     "rate(ml_prediction_confidence_sum[1m]) / rate(ml_prediction_confidence_count[1m])"),
    ("Prediction errors", "timeseries", "sum(rate(ml_prediction_errors_total[1m])) by (reason)"),
]


def grafana_dashboard():
    datasource = {"type": "prometheus", "uid": "prometheus"}
    panels = [{"id": i + 1, "title": t, "type": kind, "datasource": datasource,
               "gridPos": {"h": 8, "w": 12, "x": (i % 2) * 12, "y": (i // 2) * 8},
               "targets": [{"refId": "A", "expr": expr, "datasource": datasource}]}
              for i, (t, kind, expr) in enumerate(DASHBOARD_PANELS)]
    return {"title": "ML Model Monitoring", "uid": "ml-model-monitoring", "schemaVersion": 39,
            "refresh": "5s", "time": {"from": "now-15m", "to": "now"}, "panels": panels}


def write_monitoring_stack():
    stack = OUTPUT_DIR / "monitoring-stack"
    if stack.exists():
        shutil.rmtree(stack)
    for name, content in STACK_FILES.items():
        path = stack / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    dashboards = stack / "monitoring" / "grafana" / "dashboards"
    dashboards.mkdir(parents=True)
    (dashboards / "ml-model-dashboard.json").write_text(json.dumps(grafana_dashboard(), indent=2))
    (stack / "data").mkdir()
    shutil.copy2(BASE_DIR / "data" / "Iris.csv", stack / "data" / "Iris.csv")
    shutil.copy2(Path(__file__), stack / "monitoring_logging.py")
    print(f"Prometheus + Grafana stack written to {stack}")
    print("Start it (Docker Desktop must be running):")
    print(f'  cd "{stack}"')
    print("  docker compose up -d --build")
    print("  Metrics http://localhost:8000/metrics | Prometheus http://localhost:9090 | "
          "Grafana http://localhost:3000 (admin/admin)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 10 - monitoring and logging")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("train", help="train the model and save the drift baseline")
    s = sub.add_parser("simulate", help="simulate normal, invalid and drifted traffic")
    s.add_argument("--n", type=int, default=150, help="predictions per phase")
    s.add_argument("--delay", type=float, default=0.0, help="seconds between predictions")
    s.add_argument("--serve", action="store_true", help="expose Prometheus metrics on port 8000 and loop forever")
    sub.add_parser("analyze", help="summary and plots of logs/predictions.jsonl")
    sub.add_parser("stack", help="write the Prometheus + Grafana docker compose stack to outputs/")
    args = parser.parse_args()

    if args.command == "train":
        train()
    elif args.command == "simulate":
        simulate(args.n, args.delay, args.serve)
    elif args.command == "analyze":
        analyze()
    elif args.command == "stack":
        write_monitoring_stack()
    else:
        train()
        print()
        simulate()
        print()
        analyze()
        print()
        write_monitoring_stack()
