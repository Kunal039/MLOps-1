"""
Experiment 7 - Step 2: serve predictions from the trained model and expose Prometheus metrics.

    python predict.py                      -> predict a batch of sample flowers, print the metrics
    python predict.py 6.7 3.0 5.2 2.3      -> predict one flower
    python predict.py --serve              -> keep predicting and expose metrics on http://localhost:8000/metrics
                                              for Prometheus to scrape (see prometheus.yml)

Metrics: model_predictions_total{predicted_class}, model_prediction_confidence,
         model_prediction_latency_seconds, model_invalid_inputs_total, model_test_accuracy, model_info
"""

import json
import random
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram, generate_latest, start_http_server

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
model = joblib.load(MODEL_DIR / "model.joblib")
metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
FEATURES = metadata["features"]

PREDICTIONS = Counter("model_predictions_total", "Predictions made per class", ["predicted_class"])
CONFIDENCE = Histogram("model_prediction_confidence", "Confidence of predictions",
                       buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
LATENCY = Histogram("model_prediction_latency_seconds", "Time taken to make one prediction")
INVALID = Counter("model_invalid_inputs_total", "Rejected inputs")
Gauge("model_test_accuracy", "Accuracy of the deployed model on the test set").set(metadata["test_accuracy"])
Gauge("model_info", "Model metadata", ["name", "version"]).labels(metadata["name"], metadata["version"]).set(1)

SAMPLES = [
    {"SepalLengthCm": 5.1, "SepalWidthCm": 3.5, "PetalLengthCm": 1.4, "PetalWidthCm": 0.2},
    {"SepalLengthCm": 6.0, "SepalWidthCm": 2.7, "PetalLengthCm": 4.1, "PetalWidthCm": 1.0},
    {"SepalLengthCm": 6.7, "SepalWidthCm": 3.0, "PetalLengthCm": 5.2, "PetalWidthCm": 2.3},
]


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


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--serve"]:
        start_http_server(8000)
        print("Metrics at http://localhost:8000/metrics - predicting one sample per second (Ctrl+C to stop)")
        while True:
            print(predict(random_sample())["prediction"])
            time.sleep(1)

    if len(args) == 4:
        record = dict(zip(FEATURES, map(float, args)))
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

    print("Prometheus metrics:")
    for line in generate_latest().decode().splitlines():
        if line.startswith(("model_predictions_total", "model_invalid_inputs_total", "model_test_accuracy",
                            "model_prediction_confidence_count")):
            print(" ", line)
