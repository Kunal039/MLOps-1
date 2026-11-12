"""
Experiment 7 - Step 1: train and save the model that the Flask API will serve.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Iris.csv"
MODEL_DIR = BASE_DIR / "model"
FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]


def main():
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["Species"], test_size=0.2, random_state=42, stratify=df["Species"]
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    meta = {"name": "iris-random-forest", "version": "1.0.0", "features": FEATURES,
            "classes": list(model.classes_), "test_accuracy": round(accuracy, 4)}
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Model trained - test accuracy {accuracy:.4f}. Saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
