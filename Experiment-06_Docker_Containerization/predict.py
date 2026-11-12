"""
Batch / command-line prediction using the model packaged in the container.

Usage:
    python predict.py                      -> predicts a few built-in samples
    python predict.py 5.1 3.5 1.4 0.2      -> predicts one flower
Inside Docker:
    docker run --rm iris-ml-model:1.0 python predict.py 6.7 3.0 5.2 2.3
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "iris_model.joblib"
FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]


def load_model():
    if not MODEL_PATH.exists():
        sys.exit("Model not found - run `python train.py` first.")
    return joblib.load(MODEL_PATH)


def predict(model, rows):
    df = pd.DataFrame(rows, columns=FEATURES)
    labels = model.predict(df)
    confidence = model.predict_proba(df).max(axis=1)
    return list(zip(labels, confidence))


if __name__ == "__main__":
    model = load_model()
    if len(sys.argv) == 5:
        samples = [[float(v) for v in sys.argv[1:]]]
    else:
        samples = [[5.1, 3.5, 1.4, 0.2], [6.0, 2.7, 4.1, 1.0], [6.7, 3.0, 5.2, 2.3]]

    for features, (label, conf) in zip(samples, predict(model, samples)):
        print(f"{features} -> {label} (confidence {conf:.2f})")
