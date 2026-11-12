"""
Experiment 16: Sample project - House Price Prediction System using Linear Regression
======================================================================================
Dataset : California Housing (20,640 districts, 8 features, target = median house value in $100,000s)
Steps   : EDA -> cleaning (cap outliers) -> feature engineering -> Linear Regression
          (+ Ridge / Lasso / polynomial to study regularisation & under/over-fitting)
          -> 5-fold cross-validation -> evaluation (MSE, RMSE, MAE, R2) -> save model for predict.py
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from features import FeatureEngineer

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "california_housing.csv"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

INPUT_FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
TARGET = "MedHouseVal"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


if __name__ == "__main__":
    # 1. Load + EDA --------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    print("Shape:", df.shape, "| missing values:", int(df.isna().sum().sum()), "| duplicates:", df.duplicated().sum())
    print(df.describe().T[["mean", "min", "max"]].round(2))

    # 2. Cleaning: target is capped at 5.0 in the census data; extreme ratios are data errors
    capped = (df[TARGET] >= 5.0).sum()
    df = df[df[TARGET] < 5.0]
    for col in ["AveRooms", "AveBedrms", "AveOccup", "Population"]:
        df = df[df[col] <= df[col].quantile(0.995)]
    print(f"\nRemoved {capped} capped-target rows and extreme outliers -> {len(df)} rows")

    X, y = df[INPUT_FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Models --------------------------------------------------------------------
    def make(reg, degree=1):
        steps = [("features", FeatureEngineer()), ("scaler", StandardScaler())]
        if degree > 1:
            steps.append(("poly", PolynomialFeatures(degree, include_bias=False)))
        return Pipeline(steps + [("reg", reg)])

    models = {
        "Linear Regression": make(LinearRegression()),
        "Ridge (L2, alpha=10)": make(Ridge(alpha=10)),
        "Lasso (L1, alpha=0.01)": make(Lasso(alpha=0.01, max_iter=20000)),
        "Polynomial deg 2 + Ridge": make(Ridge(alpha=10), degree=2),
    }

    cv = KFold(5, shuffle=True, random_state=42)
    rows = []
    for name, model in models.items():
        cv_rmse = -cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({"Model": name, "CV_RMSE": cv_rmse.mean(),
                     "Train_R2": r2_score(y_train, model.predict(X_train)),
                     "Test_MSE": mean_squared_error(y_test, pred),
                     "Test_RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                     "Test_MAE": mean_absolute_error(y_test, pred),
                     "Test_R2": r2_score(y_test, pred)})
    results = pd.DataFrame(rows).set_index("Model")
    print("\n=== Model comparison ===")
    print(results.round(4))
    results.to_csv(OUTPUT_DIR / "model_comparison.csv")

    # 4. Final model: plain Linear Regression (the project requirement) ----------
    final = models["Linear Regression"]
    pred = final.predict(X_test)
    coef = pd.Series(final.named_steps["reg"].coef_,
                     index=final.named_steps["features"].transform(X_test.head(1)).columns)
    print("\nLinear Regression coefficients (standardised features):\n", coef.sort_values().round(3))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].scatter(y_test, pred, s=4, alpha=0.3)
    axes[0].plot([0, 5], [0, 5], "r--")
    axes[0].set_xlabel("Actual ($100k)")
    axes[0].set_ylabel("Predicted ($100k)")
    axes[0].set_title(f"Actual vs Predicted (R2={r2_score(y_test, pred):.3f})")
    axes[1].scatter(pred, y_test - pred, s=4, alpha=0.3)
    axes[1].axhline(0, color="red")
    axes[1].set_title("Residuals vs Predicted")
    coef.sort_values().plot(kind="barh", ax=axes[2], title="Coefficients")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "evaluation.png")
    plt.close()

    joblib.dump(final, MODEL_DIR / "house_price_model.joblib")
    lr = results.loc["Linear Regression"]
    info = {"model": "Linear Regression with feature engineering", "features": INPUT_FEATURES,
            "target": "Median house value (USD 100,000s)",
            "metrics": {k: round(float(lr[k]), 4) for k in ["Test_MSE", "Test_RMSE", "Test_MAE", "Test_R2"]},
            "input_ranges": {c: [round(float(X[c].min()), 2), round(float(X[c].max()), 2)] for c in INPUT_FEATURES}}
    (MODEL_DIR / "model_info.json").write_text(json.dumps(info, indent=2))
    print("\nModel saved to model/house_price_model.joblib -> predict with: python predict.py")
