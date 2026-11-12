"""
Experiment 6: Containerize a machine learning model using Docker
----------------------------------------------------------------
Step 1 - train the model that will be packaged inside the Docker image.
This script runs during `docker build` (see the RUN python train.py line in the Dockerfile).
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Iris.csv"
MODEL_DIR = BASE_DIR / "model"
FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]


def main():
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df["Species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling + Logistic Regression wrapped in one pipeline, so the container only needs one file
    model = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=1000))])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print(classification_report(y_test, pred))
    print(f"Test accuracy: {accuracy:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "iris_model.joblib")
    (MODEL_DIR / "model_info.json").write_text(
        json.dumps({"model": "LogisticRegression + StandardScaler", "features": FEATURES,
                    "accuracy": round(accuracy, 4), "version": "1.0"}, indent=2)
    )
    print("Model saved to", MODEL_DIR / "iris_model.joblib")


if __name__ == "__main__":
    main()
