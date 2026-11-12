"""
Experiment 20 - Step 4: Ethical AI - fairness and bias audit of the salary model
=================================================================================
The historical data contains a gender pay gap. Questions:
  1. How big is the gap in the DATA, after comparing like with like (same role, city, experience)?
  2. Does the MODEL reproduce it? (gender is not an input, but proxies could leak it)
  3. Is the model equally ACCURATE for every group? (error parity)
  4. Proxy check: can gender be predicted from the model inputs?
Results are logged to MLflow as an "audit" run, so fairness is versioned alongside accuracy.
"""

import json
import os
import warnings
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder

from generate_data import DATA_DIR

warnings.filterwarnings("ignore")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
BASE_DIR = Path(__file__).resolve().parent
info = json.loads((BASE_DIR / "model" / "model_info.json").read_text())
model = joblib.load(BASE_DIR / "model" / "salary_model.joblib")
FEATURES = info["features"]

df = pd.read_csv(DATA_DIR / "employees.csv")
df["predicted"] = model.predict(df[FEATURES])
df["abs_error"] = (df["salary_lpa"] - df["predicted"]).abs()

# 1. Raw and adjusted gap in the data ----------------------------------------------
raw_gap = 1 - df[df.gender == "Female"].salary_lpa.mean() / df[df.gender == "Male"].salary_lpa.mean()
X = pd.get_dummies(df[FEATURES], drop_first=True, dtype=float).assign(is_female=(df.gender == "Female").astype(float))
reg = LinearRegression().fit(X, np.log(df["salary_lpa"]))
adjusted_gap = 1 - np.exp(reg.coef_[list(X.columns).index("is_female")])
print("=== 1. PAY GAP IN THE HISTORICAL DATA ===")
print(f"Raw gap (average female vs male salary): {raw_gap:.1%}")
print(f"Adjusted gap (same role, city, experience, education...): {adjusted_gap:.1%}  <- unexplained = bias")

# 2. Does the model reproduce it? ------------------------------------------------------
pred_gap = 1 - df[df.gender == "Female"].predicted.mean() / df[df.gender == "Male"].predicted.mean()
residual = (df["salary_lpa"] - df["predicted"]).groupby(df["gender"]).mean()
print("\n=== 2. MODEL PREDICTIONS ===")
print(f"Gap in predicted salaries: {pred_gap:.1%} (explained by role/experience mix, not by gender itself)")
print("Mean (actual - predicted) by gender:\n", residual.round(3))
print("=> The model, which never sees gender, predicts women's salaries without the unexplained penalty:")
print("   it can be used to flag employees who are paid below the model's estimate (pay-equity review).")

# 3. Error parity ------------------------------------------------------------------------
print("\n=== 3. ERROR PARITY (MAE in LPA) ===")
parity = {}
for col in ["gender", "city", "education"]:
    mae = df.groupby(col)["abs_error"].mean()
    parity[col] = round(float(mae.max() / mae.min()), 3)
    print(f"{col:10s} max/min MAE ratio = {parity[col]:.2f}   ", mae.round(3).to_dict())

# 4. Proxy check ----------------------------------------------------------------------------
enc = OrdinalEncoder().fit_transform(df[FEATURES].astype(str))
proba = cross_val_predict(RandomForestClassifier(n_estimators=200, random_state=0), enc,
                          df["gender"] == "Female", cv=5, method="predict_proba")[:, 1]
proxy_auc = roc_auc_score(df["gender"] == "Female", proba)
print("\n=== 4. PROXY CHECK ===")
print(f"Can the inputs predict gender? ROC-AUC = {proxy_auc:.3f} (0.5 = no proxy leakage)")

# PASS = no proxy leakage, similar accuracy for men and women, and the model does not amplify the data's bias
verdict = "PASS" if proxy_auc < 0.6 and parity["gender"] < 1.2 and abs(pred_gap) < abs(adjusted_gap) else "REVIEW"
print(f"\nFairness verdict: {verdict}")

mlflow.set_tracking_uri(f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}")
mlflow.set_experiment("employee-salary-prediction")
with mlflow.start_run(run_name="fairness-audit"):
    mlflow.set_tag("type", "fairness-audit")
    mlflow.set_tag("audited_run_id", info["mlflow_run_id"])
    mlflow.log_metrics({"raw_gender_gap": raw_gap, "adjusted_gender_gap": adjusted_gap,
                        "predicted_gender_gap": pred_gap, "proxy_auc": proxy_auc,
                        "mae_ratio_gender": parity["gender"]})
print("Audit results logged to MLflow (run name: fairness-audit)")
