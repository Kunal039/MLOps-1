"""
Experiment 10 - Step 1: train the model and save the training-data statistics
that the monitoring service uses as a baseline for data-drift detection.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

df = pd.read_csv(BASE_DIR / "data" / "Iris.csv")
X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], df["Species"], test_size=0.2, random_state=42, stratify=df["Species"]
)
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(model, MODEL_DIR / "model.joblib")
baseline = {
    "version": "1.0.0",
    "accuracy": round(accuracy, 4),
    "features": FEATURES,
    "mean": X_train.mean().round(4).to_dict(),
    "std": X_train.std().round(4).to_dict(),
}
(MODEL_DIR / "baseline.json").write_text(json.dumps(baseline, indent=2))
print(f"Model trained (accuracy {accuracy:.4f}); baseline statistics saved for drift detection.")
