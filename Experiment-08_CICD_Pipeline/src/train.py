"""Train the model with the parameters in params.json and save model + metrics (CI 'train' stage)."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
from data import load_data, split_data, validate_data  # noqa: E402

MODEL_DIR = PROJECT_DIR / "model"
REPORT_DIR = PROJECT_DIR / "reports"


def build_model(params):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=params["n_estimators"],
                                       max_depth=params["max_depth"],
                                       random_state=params["random_state"])),
    ])


def main():
    params = json.loads((PROJECT_DIR / "params.json").read_text())
    X, y = load_data()
    validate_data(X, y)
    X_train, X_test, y_train, y_test = split_data(X, y, params["test_size"], params["random_state"])

    model = build_model(params)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "cv_accuracy_mean": round(cv_scores.mean(), 4),
        "test_accuracy": round(accuracy_score(y_test, pred), 4),
        "test_f1_macro": round(f1_score(y_test, pred, average="macro"), 4),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Training complete:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
