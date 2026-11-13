"""Load the trained model and predict wine classes (the artifact deployed by the CD stage).

    python src/predict.py           -> predict the first 3 wines of the dataset
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
from data import load_data  # noqa: E402

CLASSES = ["class_0", "class_1", "class_2"]
model = joblib.load(PROJECT_DIR / "model" / "model.joblib")
FEATURES = list(model.feature_names_in_)


def predict(features):
    if not isinstance(features, list) or len(features) != len(FEATURES):
        raise ValueError(f"features must be a list of {len(FEATURES)} numbers")
    row = pd.DataFrame([features], columns=FEATURES)
    return CLASSES[int(model.predict(row)[0])], round(float(model.predict_proba(row).max()), 4)


if __name__ == "__main__":
    metrics = json.loads((PROJECT_DIR / "reports" / "metrics.json").read_text())
    print("Deployed model metrics:", metrics)
    X, y = load_data()
    for i in [0, 70, 150]:
        label, confidence = predict(X.iloc[i].tolist())
        print(f"Wine #{i}: predicted {label} (confidence {confidence}), actual class_{y.iloc[i]}")
