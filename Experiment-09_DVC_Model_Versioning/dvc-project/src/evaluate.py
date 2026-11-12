"""DVC stage 3 - evaluate: write metrics.json (tracked by `dvc metrics`) and confusion-matrix data for `dvc plots`."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

PROJECT_DIR = Path(__file__).resolve().parents[1]

test = pd.read_csv(PROJECT_DIR / "data" / "processed" / "test.csv")
X, y = test.drop(columns=["target"]), test["target"]
model = joblib.load(PROJECT_DIR / "models" / "model.joblib")
pred = model.predict(X)

metrics = {
    "accuracy": round(accuracy_score(y, pred), 4),
    "precision": round(precision_score(y, pred), 4),
    "recall": round(recall_score(y, pred), 4),
    "f1": round(f1_score(y, pred), 4),
    "roc_auc": round(roc_auc_score(y, model.predict_proba(X)[:, 1]), 4),
    "test_rows": len(test),
}
(PROJECT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

plots = PROJECT_DIR / "plots"
plots.mkdir(exist_ok=True)
pd.DataFrame({"actual": y, "predicted": pred}).to_csv(plots / "confusion_matrix.csv", index=False)
print("evaluate:", metrics)
