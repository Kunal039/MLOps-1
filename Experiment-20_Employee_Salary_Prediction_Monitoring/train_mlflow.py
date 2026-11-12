"""
Experiment 20: Sample project - Employee Salary Prediction and Monitoring System using MLOps tools
==================================================================================================
Step 1 - Experiment tracking and model registry with MLflow
    * every model run logs its parameters, metrics (RMSE, MAE, R2, CV) and the model artifact
    * the best run is registered in the MLflow Model Registry as "employee-salary-model"
      and given the alias "production"
    * the production model is also exported to model/ for the Flask API

View the experiments:   mlflow ui --backend-store-uri sqlite:///mlflow.db   ->  http://127.0.0.1:5000
"""

import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from generate_data import DATA_DIR, generate

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# Gender is deliberately NOT a model input (fairness by design). Age is excluded because it
# duplicates experience and is a protected attribute in many anti-discrimination laws.
NUMERIC = ["experience_years", "skills_count", "certifications", "performance_rating"]
NOMINAL = ["job_role", "city", "company_size"]
ORDINAL = ["education"]
FEATURES = NUMERIC + NOMINAL + ORDINAL
TARGET = "salary_lpa"
MODEL_NAME = "employee-salary-model"

if not (DATA_DIR / "employees.csv").exists():
    DATA_DIR.mkdir(exist_ok=True)
    generate().to_csv(DATA_DIR / "employees.csv", index=False)
df = pd.read_csv(DATA_DIR / "employees.csv")
X_train, X_test, y_train, y_test = train_test_split(df[FEATURES], df[TARGET], test_size=0.2, random_state=42)

preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("nom", OneHotEncoder(handle_unknown="ignore"), NOMINAL),
    ("ord", OrdinalEncoder(categories=[["Diploma", "Bachelor", "Master", "PhD"]]), ORDINAL),
])

candidates = {
    "LinearRegression": (LinearRegression(), {}),
    "Ridge": (Ridge(alpha=1.0), {"alpha": 1.0}),
    "DecisionTree": (DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=42),
                     {"max_depth": 8, "min_samples_leaf": 10}),
    "RandomForest": (RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1),
                     {"n_estimators": 300, "min_samples_leaf": 3}),
    "GradientBoosting": (GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=3,
                                                   random_state=42),
                         {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 3}),
}

def log_model(pipe, example):
    """Log the model artifact. New MLflow versions save with skops and need tree types to be
    explicitly trusted (we trained the model ourselves, so it is safe); older versions don't."""
    try:
        mlflow.sklearn.log_model(pipe, name="model", input_example=example,
                                 skops_trusted_types=["sklearn.tree._tree.Tree"])
    except TypeError:
        mlflow.sklearn.log_model(pipe, artifact_path="model", input_example=example)


mlflow.set_tracking_uri(f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}")
mlflow.set_experiment("employee-salary-prediction")

results = []
for name, (estimator, params) in candidates.items():
    pipe = Pipeline([("prep", preprocess), ("model", estimator)])
    with mlflow.start_run(run_name=name) as run:
        cv_rmse = -cross_val_score(pipe, X_train, y_train, cv=KFold(5, shuffle=True, random_state=42),
                                   scoring="neg_root_mean_squared_error").mean()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        metrics = {"cv_rmse": cv_rmse, "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                   "test_mae": float(mean_absolute_error(y_test, pred)), "test_r2": float(r2_score(y_test, pred))}
        mlflow.log_param("algorithm", name)
        mlflow.log_params(params)
        mlflow.log_param("features", ",".join(FEATURES))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_metrics(metrics)
        mlflow.set_tag("dataset", "employees.csv")
        log_model(pipe, X_test.head(2))
        results.append({"run_id": run.info.run_id, "model": name, **metrics, "pipeline": pipe})
        print(f"{name:17s} CV RMSE {cv_rmse:.3f} | test RMSE {metrics['test_rmse']:.3f} | "
              f"MAE {metrics['test_mae']:.3f} | R2 {metrics['test_r2']:.4f}")

best = min(results, key=lambda r: r["test_rmse"])
print(f"\nBest model: {best['model']} (run {best['run_id'][:8]})")

# --- Model registry: register + promote to production -----------------------------
version = mlflow.register_model(f"runs:/{best['run_id']}/model", MODEL_NAME)
client = MlflowClient()
client.set_registered_model_alias(MODEL_NAME, "production", version.version)
client.update_model_version(MODEL_NAME, version.version,
                            description=f"{best['model']} - test RMSE {best['test_rmse']:.3f} LPA")
print(f"Registered '{MODEL_NAME}' version {version.version} with alias 'production'")

# Export for the API (avoids needing the MLflow server inside the container)
joblib.dump(best["pipeline"], MODEL_DIR / "salary_model.joblib")
reference = X_train.assign(salary_lpa=y_train)
reference.to_csv(MODEL_DIR / "reference_data.csv", index=False)       # baseline for drift monitoring
(MODEL_DIR / "model_info.json").write_text(json.dumps({
    "name": MODEL_NAME, "registry_version": int(version.version), "algorithm": best["model"],
    "features": FEATURES, "metrics": {k: round(best[k], 4) for k in ["cv_rmse", "test_rmse", "test_mae", "test_r2"]},
    "mlflow_run_id": best["run_id"]}, indent=2))
print("Exported production model to model/ for the API.")
