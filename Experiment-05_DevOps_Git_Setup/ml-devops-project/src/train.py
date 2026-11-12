"""Train an Iris classifier and save the model + metrics (the 'build' step of the ML pipeline)."""

import json
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models"


def load_data():
    iris = load_iris(as_frame=True)
    return train_test_split(iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)


def train(n_estimators=100, max_depth=None):
    X_train, X_test, y_train, y_test = load_data()
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "f1_macro": round(f1_score(y_test, pred, average="macro"), 4),
    }
    return model, metrics


def main():
    model, metrics = train()
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Model trained. Metrics:", metrics)


if __name__ == "__main__":
    main()
