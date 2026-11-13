"""
Experiment 20 - Step 3: Model monitoring - data drift and performance drift
============================================================================
Compares a NEW batch of employees (data/employees_new_batch.csv, with actual salaries)
against the training reference data (model/reference_data.csv):

  * Data drift      - Population Stability Index (PSI) for every feature
                      PSI < 0.1 stable | 0.1-0.25 moderate drift | > 0.25 significant drift
  * Prediction drift - distribution of predicted salaries (PSI)
  * Performance drift - RMSE / MAE / R2 on the new labelled data vs. the training-time metrics
  * Decision         - if significant drift or RMSE degrades by > 20% -> retrain
                       (python train_mlflow.py registers a new version in MLflow)
The new batch is also scored through predict.py so the predictions are logged.
Output: outputs/drift_report.csv, outputs/drift_report.png, outputs/monitoring_decision.json
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from generate_data import DATA_DIR, generate

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

info = json.loads((BASE_DIR / "model" / "model_info.json").read_text())
model = joblib.load(BASE_DIR / "model" / "salary_model.joblib")
FEATURES = info["features"]
reference = pd.read_csv(BASE_DIR / "model" / "reference_data.csv")
new_file = DATA_DIR / "employees_new_batch.csv"
if not new_file.exists():
    generate(800, seed=1, drift=True).to_csv(new_file, index=False)
current = pd.read_csv(new_file)


def psi(expected, actual, bins=10):
    """Population Stability Index between two samples (numeric or categorical)."""
    if not pd.api.types.is_numeric_dtype(expected):
        cats = sorted(set(expected) | set(actual))
        e = expected.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        a = actual.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    else:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def level(value):
    return "significant" if value > 0.25 else "moderate" if value > 0.1 else "stable"


# 1. Data drift -------------------------------------------------------------------
rows = [{"feature": f, "psi": round(psi(reference[f], current[f]), 4)} for f in FEATURES]
ref_pred, cur_pred = model.predict(reference[FEATURES]), model.predict(current[FEATURES])
rows.append({"feature": "PREDICTED salary", "psi": round(psi(pd.Series(ref_pred), pd.Series(cur_pred)), 4)})
rows.append({"feature": "ACTUAL salary (target)",
             "psi": round(psi(reference["salary_lpa"], current["salary_lpa"]), 4)})
report = pd.DataFrame(rows)
report["status"] = report["psi"].apply(level)
print("=== DATA DRIFT (PSI) ===")
print(report.to_string(index=False))
report.to_csv(OUTPUT_DIR / "drift_report.csv", index=False)

# 2. Performance drift -------------------------------------------------------------
rmse = float(np.sqrt(mean_squared_error(current["salary_lpa"], cur_pred)))
mae = float(mean_absolute_error(current["salary_lpa"], cur_pred))
r2 = float(r2_score(current["salary_lpa"], cur_pred))
bias = float((current["salary_lpa"] - cur_pred).mean())
base_rmse = info["metrics"]["test_rmse"]
print("\n=== PERFORMANCE ON NEW DATA ===")
print(f"RMSE {rmse:.3f} (training-time {base_rmse:.3f}, change {rmse / base_rmse - 1:+.0%})")
print(f"MAE  {mae:.3f} | R2 {r2:.3f} | mean under-prediction {bias:+.2f} LPA")

# 3. Score the new batch through the prediction service so that it is logged ---------------
from predict import predict_salary  # noqa: E402

scored = [predict_salary(r) for r in current[FEATURES].head(200).to_dict(orient="records")]
print(f"\nScored {len(scored)} new employees through predict.py -> logged to logs/predictions.jsonl")

# 4. Decision ------------------------------------------------------------------------
drifted = report.loc[report["status"] == "significant", "feature"].tolist()
retrain = bool(drifted) or rmse > 1.2 * base_rmse
decision = {"significant_drift_features": drifted, "new_data_rmse": round(rmse, 3),
            "training_rmse": base_rmse, "retrain_required": retrain,
            "action": "Retrain with recent data: python train_mlflow.py (new MLflow model version)" if retrain
            else "No action needed"}
(OUTPUT_DIR / "monitoring_decision.json").write_text(json.dumps(decision, indent=2))
print("\n=== DECISION ===\n" + json.dumps(decision, indent=2))

# 5. Plots ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
axes[0].hist(reference["experience_years"], bins=30, alpha=0.6, density=True, label="Training")
axes[0].hist(current["experience_years"], bins=30, alpha=0.6, density=True, label="New batch")
axes[0].set_title("Experience distribution")
axes[0].legend()
axes[1].scatter(current["salary_lpa"], cur_pred, s=6, alpha=0.5)
lim = [0, current["salary_lpa"].max()]
axes[1].plot(lim, lim, "r--")
axes[1].set_xlabel("Actual (LPA)")
axes[1].set_ylabel("Predicted (LPA)")
axes[1].set_title(f"New batch: RMSE {rmse:.2f} vs {base_rmse:.2f} at training")
colors = report["status"].map({"stable": "tab:green", "moderate": "tab:orange", "significant": "tab:red"})
axes[2].barh(report["feature"], report["psi"], color=colors)
axes[2].axvline(0.1, color="orange", linestyle="--")
axes[2].axvline(0.25, color="red", linestyle="--")
axes[2].set_title("PSI by feature")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "drift_report.png")
print("Plots saved to outputs/drift_report.png")
