"""
Experiment 16: Sample project - House Price Prediction System using Linear Regression
======================================================================================
Dataset : California Housing (20,640 districts, 8 features, target = median house value in $100,000s)
Steps   : EDA -> cleaning (cap outliers) -> feature engineering -> Linear Regression
          (+ Ridge / Lasso / polynomial to study regularisation & under/over-fitting)
          -> 5-fold cross-validation -> evaluation (MSE, RMSE, MAE, R2) -> save model -> predict

Usage:
    python house_price_prediction.py                 -> train, then predict a few example houses
    python house_price_prediction.py train           -> train and save the model only
    python house_price_prediction.py predict         -> predict the example houses with the saved model
    python house_price_prediction.py predict --medinc 8.3 --houseage 41 --averooms 6.9 --avebedrms 1.0
                                     --population 322 --aveoccup 2.6 --latitude 37.88 --longitude -122.23
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "california_housing.csv"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = MODEL_DIR / "house_price_model.joblib"

INPUT_FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
TARGET = "MedHouseVal"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


# ===========================================================================
# Feature engineering (part of the saved pipeline)
# ===========================================================================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds engineered features. Part of the pipeline, so predictions get them automatically."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["RoomsPerPerson"] = X["AveRooms"] / X["AveOccup"]
        X["BedroomRatio"] = X["AveBedrms"] / X["AveRooms"]
        X["LogPopulation"] = np.log1p(X["Population"])
        X["LogIncome"] = np.log(X["MedInc"])
        # distance to the two big cities (Los Angeles, San Francisco) - location drives price
        X["DistLA"] = np.hypot(X["Latitude"] - 34.05, X["Longitude"] + 118.24)
        X["DistSF"] = np.hypot(X["Latitude"] - 37.77, X["Longitude"] + 122.42)
        return X


# ===========================================================================
# Training
# ===========================================================================
def train():
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

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

    joblib.dump(final, MODEL_PATH)
    lr = results.loc["Linear Regression"]
    info = {"model": "Linear Regression with feature engineering", "features": INPUT_FEATURES,
            "target": "Median house value (USD 100,000s)",
            "metrics": {k: round(float(lr[k]), 4) for k in ["Test_MSE", "Test_RMSE", "Test_MAE", "Test_R2"]},
            "input_ranges": {c: [round(float(X[c].min()), 2), round(float(X[c].max()), 2)] for c in INPUT_FEATURES}}
    (MODEL_DIR / "model_info.json").write_text(json.dumps(info, indent=2))
    print("\nModel saved to model/house_price_model.joblib")


# ===========================================================================
# Prediction
# ===========================================================================
EXAMPLE_HOUSES = {
    "Bay Area, high income": dict(MedInc=8.3, HouseAge=41, AveRooms=6.9, AveBedrms=1.0,
                                  Population=322, AveOccup=2.6, Latitude=37.88, Longitude=-122.23),
    "Los Angeles, middle income": dict(MedInc=4.0, HouseAge=30, AveRooms=5.0, AveBedrms=1.1,
                                       Population=1200, AveOccup=3.0, Latitude=34.05, Longitude=-118.25),
    "Central Valley, low income": dict(MedInc=2.0, HouseAge=25, AveRooms=5.2, AveBedrms=1.1,
                                       Population=1500, AveOccup=3.2, Latitude=36.7, Longitude=-119.8),
}


def predict(model, house):
    row = pd.DataFrame([house])[INPUT_FEATURES]
    if (row[["MedInc", "AveRooms", "AveOccup"]] <= 0).any(axis=None):
        raise ValueError("MedInc, AveRooms and AveOccup must be positive")
    # target is in units of $100,000; the floor avoids negative prices from the linear model
    return max(float(model.predict(row)[0]) * 100_000, 15_000)


def predict_houses(args):
    model = joblib.load(MODEL_PATH)
    values = {name: getattr(args, name.lower(), None) for name in INPUT_FEATURES}
    if all(v is not None for v in values.values()):
        houses = {"Your house": values}
    else:
        houses = EXAMPLE_HOUSES
    print()
    for name, house in houses.items():
        print(f"{name:28s} -> predicted price ${predict(model, house):,.0f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("train", help="train and save the model")
    p = sub.add_parser("predict", help="predict prices with the saved model")
    for name in INPUT_FEATURES:
        p.add_argument(f"--{name.lower()}", type=float)
    args = parser.parse_args()

    if args.command in (None, "train"):
        train()
    if args.command in (None, "predict"):
        predict_houses(args)


if __name__ == "__main__":
    # Run through the imported module so FeatureEngineer is pickled as
    # house_price_prediction.FeatureEngineer (not __main__.FeatureEngineer) and can be loaded anywhere.
    import house_price_prediction
    house_price_prediction.main()
