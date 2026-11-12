"""Flask service deployed by the CD stage. POST /predict {"features": [13 numbers]}"""

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.datasets import load_wine

PROJECT_DIR = Path(__file__).resolve().parent
FEATURES = load_wine().feature_names
CLASSES = ["class_0", "class_1", "class_2"]

app = Flask(__name__)
model = joblib.load(PROJECT_DIR / "model" / "model.joblib")


@app.get("/health")
def health():
    metrics_file = PROJECT_DIR / "reports" / "metrics.json"
    metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}
    return jsonify(status="ok", model_metrics=metrics)


@app.post("/predict")
def predict():
    features = (request.get_json(silent=True) or {}).get("features")
    if not isinstance(features, list) or len(features) != len(FEATURES):
        return jsonify(error=f"'features' must be a list of {len(FEATURES)} numbers"), 400
    row = pd.DataFrame([features], columns=FEATURES)
    label = int(model.predict(row)[0])
    return jsonify(prediction=CLASSES[label], confidence=round(float(model.predict_proba(row).max()), 4))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
