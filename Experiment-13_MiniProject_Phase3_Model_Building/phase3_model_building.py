"""
Experiment 13: Mini Project Phase-3 - Model building, tuning, evaluation and selection
======================================================================================
Project : Household Electricity Consumption Prediction System
Input   : ../Experiment-12_.../outputs/{train,validation,test}.csv   (run Phase-2 first)

Steps
    1. Baseline: "same hour yesterday" (a model must beat this to be useful)
    2. Train Linear Regression, Ridge, Decision Tree, Random Forest, SVM (SVR) and KNN
    3. Hyper-parameter tuning with GridSearchCV + TimeSeriesSplit cross-validation
    4. Compare on the validation set with MSE, RMSE, MAE and R2; check over/under-fitting
    5. Retrain the best model on train + validation, final evaluation on the test set
Output  : outputs/model_comparison.csv, plots, models/best_model.joblib + model_metadata.json
"""

import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = Path(__file__).resolve().parent
PHASE2_DIR = BASE_DIR.parent / "Experiment-12_MiniProject_Phase2_Preprocessing_Features" / "outputs"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

if not (PHASE2_DIR / "train.csv").exists():
    raise SystemExit("Processed data not found - run Experiment-12 (Phase-2) first.")

config = json.loads((PHASE2_DIR / "feature_config.json").read_text())
FEATURES, TARGET = config["features"], config["target"]
load = lambda name: pd.read_csv(PHASE2_DIR / f"{name}.csv", index_col=0, parse_dates=True)  # noqa: E731
train, val, test = load("train"), load("validation"), load("test")
X_train, y_train = train[FEATURES], train[TARGET]
X_val, y_val = val[FEATURES], val[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]


def scores(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MSE": float(mse), "RMSE": float(np.sqrt(mse)),
            "MAE": float(mean_absolute_error(y_true, y_pred)), "R2": float(r2_score(y_true, y_pred))}


# 1. Baseline -----------------------------------------------------------------------
baseline = scores(y_val, X_val["lag_24"])
print("Baseline (same hour yesterday) on validation:", {k: round(v, 4) for k, v in baseline.items()})

# 2 + 3. Models and hyper-parameter grids -----------------------------------------
# Every model is a Pipeline(scaler -> model) so scaling is learnt inside each CV fold.
candidates = {
    "Linear Regression": (LinearRegression(), {}),
    "Ridge Regression": (Ridge(), {"model__alpha": [0.1, 1, 10, 100]}),
    "Decision Tree": (DecisionTreeRegressor(random_state=42),
                      {"model__max_depth": [4, 6, 8, 12], "model__min_samples_leaf": [1, 5, 20]}),
    "Random Forest": (RandomForestRegressor(random_state=42, n_jobs=-1),
                      {"model__n_estimators": [100, 200], "model__max_depth": [8, 12, None],
                       "model__min_samples_leaf": [1, 5]}),
    "SVM (SVR)": (SVR(), {"model__C": [1, 10], "model__epsilon": [0.05, 0.1], "model__kernel": ["rbf"]}),
    "KNN": (KNeighborsRegressor(), {"model__n_neighbors": [5, 10, 20], "model__weights": ["uniform", "distance"]}),
}

tscv = TimeSeriesSplit(n_splits=5)       # always validates on data that comes AFTER the training folds
rows, fitted = [], {}
for name, (estimator, grid) in candidates.items():
    start = time.perf_counter()
    pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    search = GridSearchCV(pipe, grid, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    fitted[name] = best

    train_s = scores(y_train, best.predict(X_train))
    val_s = scores(y_val, best.predict(X_val))
    rows.append({"Model": name, "CV_RMSE": -search.best_score_, "Train_RMSE": train_s["RMSE"],
                 "Val_MSE": val_s["MSE"], "Val_RMSE": val_s["RMSE"], "Val_MAE": val_s["MAE"],
                 "Val_R2": val_s["R2"], "Best_params": {k.replace("model__", ""): v
                                                        for k, v in search.best_params_.items()},
                 "Time_s": time.perf_counter() - start})
    print(f"{name:18s} CV RMSE={-search.best_score_:.4f}  Val RMSE={val_s['RMSE']:.4f}  "
          f"params={rows[-1]['Best_params']}")

# 4. Comparison --------------------------------------------------------------------
comparison = pd.DataFrame(rows).sort_values("Val_RMSE").reset_index(drop=True)
comparison["Overfit_gap"] = comparison["Val_RMSE"] - comparison["Train_RMSE"]
comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
print("\n=== MODEL COMPARISON (validation set) ===")
print(comparison.drop(columns=["Best_params"]).round(4).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].barh(comparison["Model"], comparison["Val_RMSE"], color="tab:blue", label="Validation")
axes[0].barh(comparison["Model"], comparison["Train_RMSE"], color="tab:orange", alpha=0.6, label="Train")
axes[0].axvline(baseline["RMSE"], color="red", linestyle="--", label="Baseline (lag 24)")
axes[0].set_xlabel("RMSE (kW)")
axes[0].set_title("Model comparison - train vs validation RMSE")
axes[0].legend()

best_name = comparison.loc[0, "Model"]
print(f"\nBest model: {best_name}")

# 5. Retrain best on train+validation, evaluate on test ---------------------------
final_model = fitted[best_name]
final_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
test_pred = final_model.predict(X_test)
test_scores = scores(y_test, test_pred)
baseline_test = scores(y_test, X_test["lag_24"])
print("Test set:", {k: round(v, 4) for k, v in test_scores.items()})
print("Baseline on test:", {k: round(v, 4) for k, v in baseline_test.items()})
print(f"Improvement over baseline RMSE: {(1 - test_scores['RMSE'] / baseline_test['RMSE']):.1%}")

week = slice(0, 24 * 7)
axes[1].plot(y_test.index[week], y_test.values[week], label="Actual")
axes[1].plot(y_test.index[week], test_pred[week], label=f"Predicted ({best_name})")
axes[1].set_title("Test set - first week")
axes[1].set_ylabel("kW")
axes[1].legend()
axes[1].tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison.png")
plt.close()

if hasattr(final_model.named_steps["model"], "feature_importances_"):
    imp = pd.Series(final_model.named_steps["model"].feature_importances_, index=FEATURES).sort_values()
    imp.plot(kind="barh", figsize=(8, 6), title=f"Feature importance - {best_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png")
    plt.close()

joblib.dump(final_model, MODEL_DIR / "best_model.joblib")
metadata = {
    "project": "Household Electricity Consumption Prediction",
    "model_name": best_name,
    "params": comparison.loc[0, "Best_params"],
    "features": FEATURES,
    "target": TARGET,
    "test_metrics": {k: round(v, 4) for k, v in test_scores.items()},
    "baseline_test_rmse": round(baseline_test["RMSE"], 4),
    "trained_on_rows": len(X_train) + len(X_val),
    "trained_at": datetime.now().isoformat(timespec="seconds"),
}
(MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
print("\nSaved best model and metadata to", MODEL_DIR)
print("Experiment 13 (Mini Project Phase-3) completed successfully.")
