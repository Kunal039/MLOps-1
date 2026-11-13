"""
Experiment 20: Sample project - Employee Salary Prediction and Monitoring System using MLOps tools
==================================================================================================
Step 1 - generate    : employee dataset (annual salary in lakh INR)
                       data/employees.csv           historical training data (contains a small
                                                    unexplained gender pay gap for the fairness audit)
                       data/employees_new_batch.csv newer data with drift: more senior hires,
                                                    12% salary inflation, more AI roles
Step 2 - train       : experiment tracking and model registry with MLflow
                       * every model run logs its parameters, metrics (RMSE, MAE, R2, CV) and the model artifact
                       * the best run is registered in the MLflow Model Registry as "employee-salary-model"
                         and given the alias "production"
                       * the production model is also exported to model/ for prediction
Step 3 - predict     : salary prediction with input validation and prediction logging (logs/predictions.jsonl)
Step 4 - monitor     : data drift (PSI), prediction drift and performance drift on the new batch -> retrain decision
Step 5 - audit       : fairness and bias audit, logged to MLflow

Usage:
    python salary_prediction_monitoring.py              -> run all steps in order
    python salary_prediction_monitoring.py generate     (or train / predict / monitor / audit) -> one step

View the experiments:   mlflow ui --backend-store-uri sqlite:///mlflow.db   ->  http://127.0.0.1:5000
"""

import argparse
import json
import logging
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = BASE_DIR / "logs"
TRACKING_URI = f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}"

ROLES = {"Data Analyst": 6, "Software Engineer": 9, "Data Scientist": 12, "ML Engineer": 14,
         "DevOps Engineer": 10, "Product Manager": 16, "HR Executive": 5, "Sales Executive": 5.5}
EDUCATION = {"Diploma": 0.85, "Bachelor": 1.0, "Master": 1.15, "PhD": 1.3}
CITY = {"Bengaluru": 1.15, "Mumbai": 1.12, "Delhi NCR": 1.08, "Hyderabad": 1.05, "Pune": 1.0, "Chennai": 0.98,
        "Jaipur": 0.85, "Chandigarh": 0.88}
SIZE = {"Startup": 0.95, "Mid-size": 1.0, "Enterprise": 1.1}

# Gender is deliberately NOT a model input (fairness by design). Age is excluded because it
# duplicates experience and is a protected attribute in many anti-discrimination laws.
NUMERIC = ["experience_years", "skills_count", "certifications", "performance_rating"]
NOMINAL = ["job_role", "city", "company_size"]
ORDINAL = ["education"]
FEATURES = NUMERIC + NOMINAL + ORDINAL
TARGET = "salary_lpa"
MODEL_NAME = "employee-salary-model"


# ===========================================================================
# Step 1 - Data generation
# ===========================================================================
def generate(n=3000, seed=0, drift=False):
    rng = np.random.default_rng(seed)
    role_p = np.array([.15, .25, .12, .08, .1, .08, .11, .11])
    if drift:                                      # new hiring trend: more ML roles
        role_p = np.array([.1, .2, .15, .2, .1, .08, .08, .09])
    df = pd.DataFrame({
        "employee_id": [f"E{seed}{i:05d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n, p=[.62, .38]),
        "job_role": rng.choice(list(ROLES), n, p=role_p / role_p.sum()),
        "education": rng.choice(list(EDUCATION), n, p=[.1, .55, .3, .05]),
        "city": rng.choice(list(CITY), n),
        "company_size": rng.choice(list(SIZE), n, p=[.3, .4, .3]),
        "experience_years": np.clip(rng.gamma(2.2 if not drift else 3.2, 2.3, n), 0, 30).round(1),
        "skills_count": rng.integers(2, 16, n),
        "certifications": rng.poisson(1.2, n),
        "performance_rating": rng.choice([1, 2, 3, 4, 5], n, p=[.05, .15, .4, .3, .1]),
    })
    df["age"] = (22 + df["experience_years"] + rng.normal(1.5, 1.5, n)).round().astype(int)
    salary = (df["job_role"].map(ROLES)
              * df["education"].map(EDUCATION)
              * df["city"].map(CITY)
              * df["company_size"].map(SIZE)
              * (1 + 0.085 * df["experience_years"] - 0.0012 * df["experience_years"] ** 2)
              * (1 + 0.012 * df["skills_count"] + 0.02 * df["certifications"])
              * (0.9 + 0.05 * df["performance_rating"])
              * np.where(df["gender"] == "Female", 0.94, 1.0)          # historical bias (6% gap)
              * rng.lognormal(0, 0.08, n))
    if drift:
        salary *= 1.12                                                  # inflation / market correction
    df["salary_lpa"] = salary.round(2)
    return df


def generate_data():
    DATA_DIR.mkdir(exist_ok=True)
    for out, df in [(DATA_DIR / "employees.csv", generate()),
                    (DATA_DIR / "employees_new_batch.csv", generate(800, seed=1, drift=True))]:
        df.to_csv(out, index=False)
        print(f"Saved {len(df)} employees to {out.name}; mean salary {df.salary_lpa.mean():.2f} LPA")


# ===========================================================================
# Step 2 - Experiment tracking and model registry with MLflow
# ===========================================================================
def log_model(pipe, example):
    """Log the model artifact. New MLflow versions save with skops and need tree types to be
    explicitly trusted (we trained the model ourselves, so it is safe); older versions don't."""
    try:
        mlflow.sklearn.log_model(pipe, name="model", input_example=example,
                                 skops_trusted_types=["sklearn.tree._tree.Tree"])
    except TypeError:
        mlflow.sklearn.log_model(pipe, artifact_path="model", input_example=example)


def train():
    MODEL_DIR.mkdir(exist_ok=True)
    if not (DATA_DIR / "employees.csv").exists():
        generate_data()
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

    mlflow.set_tracking_uri(TRACKING_URI)
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
                       "test_mae": float(mean_absolute_error(y_test, pred)),
                       "test_r2": float(r2_score(y_test, pred))}
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

    # Export for prediction (avoids needing the MLflow server at prediction time)
    joblib.dump(best["pipeline"], MODEL_DIR / "salary_model.joblib")
    reference = X_train.assign(salary_lpa=y_train)
    reference.to_csv(MODEL_DIR / "reference_data.csv", index=False)       # baseline for drift monitoring
    (MODEL_DIR / "model_info.json").write_text(json.dumps({
        "name": MODEL_NAME, "registry_version": int(version.version), "algorithm": best["model"],
        "features": FEATURES,
        "metrics": {k: round(best[k], 4) for k in ["cv_rmse", "test_rmse", "test_mae", "test_r2"]},
        "mlflow_run_id": best["run_id"]}, indent=2))
    print("Exported production model to model/.")


def load_production_model():
    info = json.loads((MODEL_DIR / "model_info.json").read_text())
    return joblib.load(MODEL_DIR / "salary_model.joblib"), info


# ===========================================================================
# Step 3 - Prediction with input validation and logging
# ===========================================================================
ALLOWED = {"job_role": set(ROLES), "city": set(CITY), "company_size": set(SIZE), "education": set(EDUCATION)}
RANGES = {"experience_years": (0, 45), "skills_count": (0, 50), "certifications": (0, 30), "performance_rating": (1, 5)}
log = logging.getLogger("salary-model")


def _prediction_log():
    """Open the prediction log (and the service log handler) the first time a prediction is made."""
    if not log.handlers:
        LOG_DIR.mkdir(exist_ok=True)
        log.setLevel(logging.INFO)
        handler = logging.FileHandler(LOG_DIR / "service.log")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        log.addHandler(handler)
    return (LOG_DIR / "predictions.jsonl").open("a")


def validate(employee):
    """Return an error message for invalid input, else None."""
    for field in FEATURES:
        if field not in employee:
            return f"Missing field '{field}'"
    for field, allowed in ALLOWED.items():
        if employee[field] not in allowed:
            return f"'{field}' must be one of {sorted(allowed)}"
    for field, (lo, hi) in RANGES.items():
        try:
            value = float(employee[field])
        except (TypeError, ValueError):
            return f"'{field}' must be a number"
        if not lo <= value <= hi:
            return f"'{field}' must be between {lo} and {hi}"
    return None


def predict_salary(model, info, employee):
    """Predicted salary (lakh INR per year) with a 95% range. Raises ValueError for invalid input."""
    with _prediction_log() as prediction_log:
        error = validate(employee)
        if error:
            log.warning("Rejected input: %s", error)
            raise ValueError(error)

        salary = float(model.predict(pd.DataFrame([{f: employee[f] for f in FEATURES}]))[0])
        prediction_log.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                         **{f: employee[f] for f in FEATURES},
                                         "predicted_salary_lpa": round(salary, 2),
                                         "model_version": info["registry_version"]}) + "\n")
    rmse = info["metrics"]["test_rmse"]
    return {"predicted_salary_lpa": round(salary, 2),
            "expected_range_lpa": [round(max(salary - 1.96 * rmse, 0), 2), round(salary + 1.96 * rmse, 2)]}


def predict_examples():
    model, info = load_production_model()
    print(f"Model: {info['algorithm']} (MLflow registry version {info['registry_version']}, "
          f"test RMSE {info['metrics']['test_rmse']} LPA)\n")
    examples = {
        "Fresher data analyst, Jaipur": dict(experience_years=0.5, skills_count=4, certifications=0,
                                             performance_rating=3, job_role="Data Analyst", city="Jaipur",
                                             company_size="Startup", education="Bachelor"),
        "Data scientist, 5 yrs, Bengaluru": dict(experience_years=5, skills_count=8, certifications=1,
                                                 performance_rating=4, job_role="Data Scientist", city="Bengaluru",
                                                 company_size="Enterprise", education="Master"),
        "Product manager, 12 yrs, Mumbai": dict(experience_years=12, skills_count=12, certifications=2,
                                                performance_rating=5, job_role="Product Manager", city="Mumbai",
                                                company_size="Enterprise", education="Master"),
    }
    for name, employee in examples.items():
        r = predict_salary(model, info, employee)
        low, high = r["expected_range_lpa"]
        print(f"{name:34s} -> {r['predicted_salary_lpa']:6.2f} LPA  (95% range {low}-{high})")
    try:
        predict_salary(model, info, {"experience_years": 5})
    except ValueError as err:
        print("\nInvalid input rejected:", err)


# ===========================================================================
# Step 4 - Model monitoring: data drift and performance drift
#   * Data drift        - Population Stability Index (PSI) for every feature
#                         PSI < 0.1 stable | 0.1-0.25 moderate drift | > 0.25 significant drift
#   * Prediction drift  - distribution of predicted salaries (PSI)
#   * Performance drift - RMSE / MAE / R2 on the new labelled data vs. the training-time metrics
#   * Decision          - if significant drift or RMSE degrades by > 20% -> retrain
# Output: outputs/drift_report.csv, outputs/drift_report.png, outputs/monitoring_decision.json
# ===========================================================================
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


def monitor_drift():
    OUTPUT_DIR.mkdir(exist_ok=True)
    model, info = load_production_model()
    reference = pd.read_csv(MODEL_DIR / "reference_data.csv")
    new_file = DATA_DIR / "employees_new_batch.csv"
    if not new_file.exists():
        generate(800, seed=1, drift=True).to_csv(new_file, index=False)
    current = pd.read_csv(new_file)

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

    # 3. Score the new batch through the prediction step so that it is logged ---------------
    scored = [predict_salary(model, info, r) for r in current[FEATURES].head(200).to_dict(orient="records")]
    print(f"\nScored {len(scored)} new employees -> logged to logs/predictions.jsonl")

    # 4. Decision ------------------------------------------------------------------------
    drifted = report.loc[report["status"] == "significant", "feature"].tolist()
    retrain = bool(drifted) or rmse > 1.2 * base_rmse
    decision = {"significant_drift_features": drifted, "new_data_rmse": round(rmse, 3),
                "training_rmse": base_rmse, "retrain_required": retrain,
                "action": "Retrain with recent data: python salary_prediction_monitoring.py train "
                          "(new MLflow model version)" if retrain else "No action needed"}
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
    plt.close()
    print("Plots saved to outputs/drift_report.png")


# ===========================================================================
# Step 5 - Ethical AI: fairness and bias audit of the salary model
# The historical data contains a gender pay gap. Questions:
#   1. How big is the gap in the DATA, after comparing like with like (same role, city, experience)?
#   2. Does the MODEL reproduce it? (gender is not an input, but proxies could leak it)
#   3. Is the model equally ACCURATE for every group? (error parity)
#   4. Proxy check: can gender be predicted from the model inputs?
# Results are logged to MLflow as an "audit" run, so fairness is versioned alongside accuracy.
# ===========================================================================
def fairness_audit():
    model, info = load_production_model()
    df = pd.read_csv(DATA_DIR / "employees.csv")
    df["predicted"] = model.predict(df[FEATURES])
    df["abs_error"] = (df["salary_lpa"] - df["predicted"]).abs()

    # 1. Raw and adjusted gap in the data ----------------------------------------------
    raw_gap = 1 - df[df.gender == "Female"].salary_lpa.mean() / df[df.gender == "Male"].salary_lpa.mean()
    X = pd.get_dummies(df[FEATURES], drop_first=True, dtype=float).assign(
        is_female=(df.gender == "Female").astype(float))
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
    verdict = ("PASS" if proxy_auc < 0.6 and parity["gender"] < 1.2 and abs(pred_gap) < abs(adjusted_gap)
               else "REVIEW")
    print(f"\nFairness verdict: {verdict}")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("employee-salary-prediction")
    with mlflow.start_run(run_name="fairness-audit"):
        mlflow.set_tag("type", "fairness-audit")
        mlflow.set_tag("audited_run_id", info["mlflow_run_id"])
        mlflow.log_metrics({"raw_gender_gap": raw_gap, "adjusted_gender_gap": adjusted_gap,
                            "predicted_gender_gap": pred_gap, "proxy_auc": proxy_auc,
                            "mae_ratio_gender": parity["gender"]})
    print("Audit results logged to MLflow (run name: fairness-audit)")


# ===========================================================================
# Command line
# ===========================================================================
STEPS = {"generate": generate_data, "train": train, "predict": predict_examples,
         "monitor": monitor_drift, "audit": fairness_audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", nargs="?", choices=list(STEPS), help="run one step (default: all steps in order)")
    args = parser.parse_args()
    for name, step in STEPS.items():
        if args.step in (None, name):
            print("\n" + "=" * 70 + f"\nSTEP: {name}\n" + "=" * 70)
            step()


if __name__ == "__main__":
    main()
