"""
Experiment 17: Sample project - Titanic Survival Prediction using Decision Tree Classifier
==========================================================================================
Steps: EDA -> feature-engineering pipeline -> Decision Tree -> GridSearchCV tuning
       -> evaluation (accuracy, precision, recall, F1, ROC-AUC, confusion matrix)
       -> tree visualisation and rules -> save pipeline for predict.py
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, classification_report,
                             f1_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from features import RAW_COLUMNS, TitanicFeatures

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# 1. Data + quick EDA -----------------------------------------------------------
df = pd.read_csv(BASE_DIR / "data" / "titanic.csv")
print(f"Passengers: {len(df)} | survival rate: {df['Survived'].mean():.1%}")
print("\nSurvival rate by gender:\n", df.groupby("Gender")["Survived"].mean().round(3))
print("\nSurvival rate by class:\n", df.groupby("Pclass")["Survived"].mean().round(3))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
df.groupby("Gender")["Survived"].mean().plot(kind="bar", ax=axes[0], title="Survival by gender", rot=0)
df.groupby("Pclass")["Survived"].mean().plot(kind="bar", ax=axes[1], title="Survival by class", rot=0)
df[df.Survived == 1]["Age"].plot(kind="hist", bins=30, alpha=0.6, ax=axes[2], label="Survived")
df[df.Survived == 0]["Age"].plot(kind="hist", bins=30, alpha=0.6, ax=axes[2], label="Died")
axes[2].set_title("Age distribution")
axes[2].legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eda.png")
plt.close()

X, y = df[RAW_COLUMNS], df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Pipeline + tuning -------------------------------------------------------------
pipe = Pipeline([("features", TitanicFeatures()), ("tree", DecisionTreeClassifier(random_state=42))])
grid = GridSearchCV(
    pipe,
    {"tree__criterion": ["gini", "entropy"], "tree__max_depth": [3, 4, 5, 6, 8],
     "tree__min_samples_leaf": [1, 5, 10], "tree__class_weight": [None, "balanced"]},
    cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="accuracy", n_jobs=-1,
)
grid.fit(X_train, y_train)
model = grid.best_estimator_
print("\nBest parameters:", grid.best_params_)
print(f"Cross-validation accuracy: {grid.best_score_:.4f}")

# 3. Evaluation -----------------------------------------------------------------
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]
metrics = {"accuracy": accuracy_score(y_test, pred), "f1": f1_score(y_test, pred),
           "roc_auc": roc_auc_score(y_test, prob), "cv_accuracy": grid.best_score_}
print(f"\nTest accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
print(classification_report(y_test, pred, target_names=["Died", "Survived"]))

tree = model.named_steps["tree"]
feature_names = list(model.named_steps["features"].transform(X_test.head(1)).columns)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Died", "Survived"], cmap="Blues", ax=axes[0])
RocCurveDisplay.from_predictions(y_test, prob, ax=axes[1])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "evaluation.png")
plt.close()

plt.figure(figsize=(24, 10))
plot_tree(tree, feature_names=feature_names, class_names=["Died", "Survived"], filled=True, max_depth=3, fontsize=9)
plt.savefig(OUTPUT_DIR / "decision_tree.png", bbox_inches="tight")
plt.close()
(OUTPUT_DIR / "tree_rules.txt").write_text(export_text(tree, feature_names=feature_names))

importance = pd.Series(tree.feature_importances_, index=feature_names).sort_values(ascending=False)
print("Most important features:\n", importance.head(5).round(3))

# 4. Save -----------------------------------------------------------------------
joblib.dump(model, MODEL_DIR / "titanic_tree.joblib")
(MODEL_DIR / "model_info.json").write_text(json.dumps(
    {"model": "DecisionTreeClassifier", "params": {k.replace("tree__", ""): v for k, v in grid.best_params_.items()},
     "metrics": {k: round(v, 4) for k, v in metrics.items()}}, indent=2))
print("\nModel saved -> run `python predict.py`")
