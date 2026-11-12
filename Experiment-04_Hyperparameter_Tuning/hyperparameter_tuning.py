"""
Experiment 4: Improve model performance by tuning hyperparameters
-----------------------------------------------------------------
Aim:
    * Use GridSearchCV to tune hyperparameters for the Decision Tree classifier.
    * Compare the performance of the tuned model with the default model.
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "titanic.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# 1. Load and preprocess (same steps as Experiment 3) --------------------------
def preprocess(data):
    data = data.copy()
    data["Title"] = data["Name"].str.extract(r",\s*([^\.]+)\.")[0]
    data["Title"] = data["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    data.loc[~data["Title"].isin(["Mr", "Mrs", "Miss", "Master"]), "Title"] = "Rare"
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)
    data["Age"] = data["Age"].fillna(data.groupby("Title")["Age"].transform("median"))
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
    data["Gender"] = data["Gender"].map({"male": 0, "female": 1})
    data = pd.get_dummies(data, columns=["Embarked", "Title"], dtype=int)
    return data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])


data = preprocess(pd.read_csv(DATA_PATH))
X = data.drop(columns=["Survived"])
y = data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 2. Default model (no tuning) --------------------------------------------------
default_model = DecisionTreeClassifier(random_state=42)
default_model.fit(X_train, y_train)
default_cv = cross_val_score(default_model, X_train, y_train, cv=cv, scoring="accuracy")
print("Default Decision Tree")
print("  Depth of tree:", default_model.get_depth(), "| leaves:", default_model.get_n_leaves())
print(f"  5-fold CV accuracy: {default_cv.mean():.4f} (+/- {default_cv.std():.4f})")

# 3. Grid search -----------------------------------------------------------------
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 4, 5, 6, 8, 10, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": [None, "sqrt"],
}
n_combos = np.prod([len(v) for v in param_grid.values()])
print(f"\nRunning GridSearchCV over {n_combos} combinations x 5 folds = {n_combos * 5} fits ...")

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    return_train_score=True,
)
grid.fit(X_train, y_train)
tuned_model = grid.best_estimator_

print("\nBest hyperparameters:", grid.best_params_)
print(f"Best 5-fold CV accuracy: {grid.best_score_:.4f}")

cv_results = pd.DataFrame(grid.cv_results_).sort_values("rank_test_score")
cv_results.to_csv(OUTPUT_DIR / "grid_search_results.csv", index=False)
print("\nTop 5 parameter combinations:")
print(cv_results[["params", "mean_train_score", "mean_test_score"]].head().to_string(index=False))


# 4. Compare default vs tuned on the held-out test set -------------------------
def metrics(model):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        "Train accuracy": model.score(X_train, y_train),
        "Test accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, prob),
    }


comparison = pd.DataFrame({"Default model": metrics(default_model), "Tuned model": metrics(tuned_model)})
comparison["Improvement"] = comparison["Tuned model"] - comparison["Default model"]
print("\n--- Default vs Tuned (test set) ---")
print(comparison.round(4))
comparison.to_csv(OUTPUT_DIR / "default_vs_tuned.csv")

gap_default = comparison.loc["Train accuracy", "Default model"] - comparison.loc["Test accuracy", "Default model"]
gap_tuned = comparison.loc["Train accuracy", "Tuned model"] - comparison.loc["Test accuracy", "Tuned model"]
print(f"\nTrain-test gap (overfitting indicator): default={gap_default:.3f}, tuned={gap_tuned:.3f}")

# 5. Plots --------------------------------------------------------------------
# 5a. Validation curve for max_depth (shows under/overfitting)
depths = range(1, 16)
train_scores, val_scores = [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=42)
    val_scores.append(cross_val_score(m, X_train, y_train, cv=cv).mean())
    train_scores.append(m.fit(X_train, y_train).score(X_train, y_train))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(depths, train_scores, "o-", label="Training accuracy")
axes[0].plot(depths, val_scores, "o-", label="Cross-validation accuracy")
axes[0].set_xlabel("max_depth")
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Underfitting vs Overfitting")
axes[0].legend()

RocCurveDisplay.from_estimator(default_model, X_test, y_test, name="Default", ax=axes[1])
RocCurveDisplay.from_estimator(tuned_model, X_test, y_test, name="Tuned", ax=axes[1])
axes[1].set_title("ROC Curve: Default vs Tuned")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tuning_comparison.png")
plt.close()

# 6. Save the best model ---------------------------------------------------------
joblib.dump({"model": tuned_model, "columns": list(X.columns), "params": grid.best_params_},
            OUTPUT_DIR / "tuned_decision_tree.joblib")
print("\nOutputs saved to:", OUTPUT_DIR)
print("Experiment 4 completed successfully.")
