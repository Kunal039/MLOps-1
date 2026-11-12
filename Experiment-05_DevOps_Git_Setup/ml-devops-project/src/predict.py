"""Load the saved model and make a prediction (the 'serve' step of the ML pipeline)."""

import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
CLASSES = ["setosa", "versicolor", "virginica"]
COLUMNS = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]


def predict(features):
    model = joblib.load(PROJECT_DIR / "models" / "model.joblib")
    row = pd.DataFrame([features], columns=COLUMNS)
    return CLASSES[int(model.predict(row)[0])]


if __name__ == "__main__":
    values = [float(v) for v in sys.argv[1:5]] if len(sys.argv) == 5 else [5.1, 3.5, 1.4, 0.2]
    print(f"Input: {values} -> Predicted species: {predict(values)}")
