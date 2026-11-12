"""
Experiment 2: Develop a basic machine learning model (Linear Regression) for prediction
---------------------------------------------------------------------------------------
Aim:
    * Implement a linear regression model using Scikit-learn.
    * Train the model on a sample dataset (house price prediction).
    * Evaluate the model's performance using metrics like Mean Squared Error (MSE).
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "house_price.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

NUMERIC = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt"]
CATEGORICAL = ["Location", "Condition", "Garage"]
TARGET = "Price"

# 1. Load data -----------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print(df.head(), "\n")
print("Missing values:\n", df.isna().sum(), "\n")
print("Duplicate rows:", df.duplicated().sum())

# 2. Feature engineering ------------------------------------------------------
df = df.drop(columns=["Id"]).drop_duplicates()
df["HouseAge"] = 2025 - df["YearBuilt"]
NUMERIC = NUMERIC + ["HouseAge"]
print("\nCorrelation of numeric features with Price:")
print(df[NUMERIC + [TARGET]].corr()[TARGET].sort_values(ascending=False))

X = df[NUMERIC + CATEGORICAL]
y = df[TARGET]

# 3. Train / test split -----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# 4. Build pipeline: scaling + one-hot encoding + linear regression --------------
preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ]
)
model = Pipeline([("preprocess", preprocessor), ("regressor", LinearRegression())])

# 5. Train ----------------------------------------------------------------------
model.fit(X_train, y_train)

# Show the learned coefficients
feature_names = model.named_steps["preprocess"].get_feature_names_out()
coefs = pd.Series(model.named_steps["regressor"].coef_, index=feature_names)
print("\nIntercept:", round(model.named_steps["regressor"].intercept_, 2))
print("Coefficients:\n", coefs.round(2).sort_values())


# 6. Evaluate -------------------------------------------------------------------
def evaluate(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "Model": name,
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


y_pred = model.predict(X_test)
result = evaluate("LinearRegression", y_test, y_pred)
print("\n--- Linear Regression performance on test set ---")
for k, v in result.items():
    if k != "Model":
        print(f"{k:5s}: {v:,.4f}")

# Baseline: always predict the mean price. A useful model must beat this.
baseline = evaluate("Baseline (mean)", y_test, np.full(len(y_test), y_train.mean()))

# 7. Compare with regularised linear models -----------------------------------
results = [baseline, result]
for name, reg in [("Ridge (L2)", Ridge(alpha=1.0)), ("Lasso (L1)", Lasso(alpha=100, max_iter=10000))]:
    m = Pipeline([("preprocess", preprocessor), ("regressor", reg)]).fit(X_train, y_train)
    results.append(evaluate(name, y_test, m.predict(X_test)))

comparison = pd.DataFrame(results).set_index("Model")
print("\n--- Model comparison ---")
print(comparison.round(4))
comparison.to_csv(OUTPUT_DIR / "metrics.csv")

# 8. Plots ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(y_test, y_pred, alpha=0.5)
lims = [y_test.min(), y_test.max()]
axes[0].plot(lims, lims, "r--", label="Perfect prediction")
axes[0].set_xlabel("Actual price")
axes[0].set_ylabel("Predicted price")
axes[0].set_title("Actual vs Predicted")
axes[0].legend()

residuals = y_test - y_pred
axes[1].hist(residuals, bins=30, edgecolor="black")
axes[1].set_title("Residual distribution")
axes[1].set_xlabel("Residual (actual - predicted)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png")
plt.close()

# 9. Save model and predict on a new house ------------------------------------
joblib.dump(model, OUTPUT_DIR / "linear_regression_model.joblib")
new_house = pd.DataFrame(
    [{"Area": 2500, "Bedrooms": 3, "Bathrooms": 2, "Floors": 2, "YearBuilt": 2010,
      "HouseAge": 15, "Location": "Urban", "Condition": "Good", "Garage": "Yes"}]
)
print("\nPrediction for a new house:", round(model.predict(new_house)[0], 2))
print("\nNote: this Kaggle dataset is largely synthetic/random, so R2 stays close to 0.")
print("The workflow (split -> fit -> MSE evaluation) is what the experiment demonstrates.")
print("\nExperiment 2 completed successfully.")
