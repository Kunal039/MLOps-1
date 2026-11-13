"""
Experiment 7: Serve an ML model and set up Prometheus to collect metrics from it
--------------------------------------------------------------------------------
Step 1 - `train`  : train and save the model that will be served.
Step 2 - `predict`: serve predictions from the trained model and record Prometheus metrics.
Step 3 - `serve`  : keep predicting and expose the metrics on http://localhost:8000/metrics
                    for Prometheus to scrape (configuration written to outputs/prometheus.yml).

    python model_serving_prometheus.py                       -> train, predict a batch of samples, print the metrics,
                                                                write outputs/prometheus.yml
    python model_serving_prometheus.py train                 -> train and save the model
    python model_serving_prometheus.py predict               -> predict a batch of sample flowers, print the metrics
    python model_serving_prometheus.py predict 6.7 3.0 5.2 2.3   -> predict one flower
    python model_serving_prometheus.py serve                 -> expose metrics on :8000 (Ctrl+C to stop)

Metrics: model_predictions_total{predicted_class}, model_prediction_confidence,
         model_prediction_latency_seconds, model_invalid_inputs_total, model_test_accuracy, model_info
"""

import argparse
import json
import random
import time
from pathlib import Path

import joblib
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram, generate_latest, start_http_server
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Iris.csv"
MODEL_DIR = BASE_DIR / "model"
FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

PREDICTIONS = Counter("model_predictions_total", "Predictions made per class", ["predicted_class"])
CONFIDENCE = Histogram("model_prediction_confidence", "Confidence of predictions",
                       buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
LATENCY = Histogram("model_prediction_latency_seconds", "Time taken to make one prediction")
INVALID = Counter("model_invalid_inputs_total", "Rejected inputs")
ACCURACY = Gauge("model_test_accuracy", "Accuracy of the deployed model on the test set")
MODEL_INFO = Gauge("model_info", "Model metadata", ["name", "version"])

SAMPLES = [
    {"SepalLengthCm": 5.1, "SepalWidthCm": 3.5, "PetalLengthCm": 1.4, "PetalWidthCm": 0.2},
    {"SepalLengthCm": 6.0, "SepalWidthCm": 2.7, "PetalLengthCm": 4.1, "PetalWidthCm": 1.0},
    {"SepalLengthCm": 6.7, "SepalWidthCm": 3.0, "PetalLengthCm": 5.2, "PetalWidthCm": 2.3},
]
model, metadata = None, None


# ---------------------------------------------------------------------------
# Step 1: train
# ---------------------------------------------------------------------------
def train():
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["Species"], test_size=0.2, random_state=42, stratify=df["Species"]
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    accuracy = accuracy_score(y_test, rf.predict(X_test))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(rf, MODEL_DIR / "model.joblib")
    meta = {"name": "iris-random-forest", "version": "1.0.0", "features": FEATURES,
            "classes": list(rf.classes_), "test_accuracy": round(accuracy, 4)}
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Model trained - test accuracy {accuracy:.4f}. Saved to {MODEL_DIR}")


# ---------------------------------------------------------------------------
# Step 2: serve predictions + metrics
# ---------------------------------------------------------------------------
def load_model():
    global model, metadata
    model = joblib.load(MODEL_DIR / "model.joblib")
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
    ACCURACY.set(metadata["test_accuracy"])
    MODEL_INFO.labels(metadata["name"], metadata["version"]).set(1)


def predict(record):
    """Validate one input record and return the predicted class with its probabilities."""
    missing = [f for f in FEATURES if f not in record]
    if missing or any(not isinstance(record[f], (int, float)) or record[f] < 0 for f in FEATURES):
        INVALID.inc()
        raise ValueError(f"Invalid input - need non-negative numbers for {FEATURES}")

    start = time.perf_counter()
    probs = model.predict_proba(pd.DataFrame([record])[FEATURES])[0]
    LATENCY.observe(time.perf_counter() - start)

    label = model.classes_[probs.argmax()]
    PREDICTIONS.labels(label).inc()
    CONFIDENCE.observe(float(probs.max()))
    return {"prediction": label, "confidence": round(float(probs.max()), 4),
            "probabilities": {c: round(float(p), 4) for c, p in zip(model.classes_, probs)}}


def random_sample():
    return {k: round(max(v + random.uniform(-0.3, 0.3), 0.1), 2) for k, v in random.choice(SAMPLES).items()}


def print_metrics():
    print("Prometheus metrics:")
    for line in generate_latest().decode().splitlines():
        if line.startswith(("model_predictions_total", "model_invalid_inputs_total", "model_test_accuracy",
                            "model_prediction_confidence_count")):
            print(" ", line)


def run_predict(values=None):
    load_model()
    if values:
        record = dict(zip(FEATURES, values))
        print(f"{list(record.values())} -> {predict(record)}")
    else:
        print(f"Model: {metadata['name']} v{metadata['version']} (test accuracy {metadata['test_accuracy']})\n")
        for s in SAMPLES:
            r = predict(s)
            print(f"{list(s.values())} -> {r['prediction']} (confidence {r['confidence']})")
        try:
            predict({"SepalLengthCm": 5.0})
        except ValueError as err:
            print("\nInvalid input rejected:", err)
        for _ in range(50):
            predict(random_sample())
        print("Made 50 more predictions.\n")
    print_metrics()


PROMETHEUS_CONFIG = """\
# Prometheus configuration - scrapes the model's metrics endpoint every 5 seconds.
# Start the metrics endpoint first:  python model_serving_prometheus.py serve
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: "ml-model"
    metrics_path: /metrics
    static_configs:
      # host.docker.internal = your PC, as seen from inside the Prometheus container.
      # If Prometheus is installed directly on your PC, use "localhost:8000" instead.
      - targets: ["host.docker.internal:8000"]
"""


def write_prometheus_config():
    path = BASE_DIR / "outputs" / "prometheus.yml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(PROMETHEUS_CONFIG)
    print(f"\nPrometheus configuration written to {path}")
    print("Start Prometheus (Docker) next to `python model_serving_prometheus.py serve`:")
    print(f'  docker run -p 9090:9090 -v "{path}:/etc/prometheus/prometheus.yml" prom/prometheus')
    print("then open http://localhost:9090 and query model_predictions_total")


def serve(port=8000):
    load_model()
    write_prometheus_config()
    start_http_server(port)
    print(f"Metrics at http://localhost:{port}/metrics - predicting one sample per second (Ctrl+C to stop)")
    while True:
        print(predict(random_sample())["prediction"])
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 7 - model serving with Prometheus metrics")
    parser.add_argument("--serve", action="store_true", help="same as the 'serve' command")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("train", help="train and save the model")
    p = sub.add_parser("predict", help="predict sample flowers or one flower, then print the metrics")
    p.add_argument("values", nargs="*", type=float, help="4 measurements: sepal length/width, petal length/width")
    sub.add_parser("serve", help="expose Prometheus metrics on port 8000 and keep predicting")
    args = parser.parse_args()

    if args.serve or args.command == "serve":
        serve()
    elif args.command == "train":
        train()
    elif args.command == "predict":
        if args.values and len(args.values) != 4:
            parser.error("predict needs exactly 4 numbers")
        run_predict(args.values or None)
    else:
        train()
        print()
        run_predict()
        write_prometheus_config()
