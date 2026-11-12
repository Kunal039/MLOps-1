"""
Experiment 3: Build a decision tree model for classification tasks
------------------------------------------------------------------
Aim:
    * Create a decision tree classifier using Scikit-learn.
    * Train and test the model using a classification dataset (Titanic dataset).
    * Evaluate the model with accuracy, confusion matrix and precision-recall.
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "titanic.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Load data -----------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isna().sum())
print("\nSurvival rate: {:.2%}".format(df["Survived"].mean()))


# 2. Preprocessing -------------------------------------------------------------
def preprocess(data):
    data = data.copy()
    # Feature engineering
    data["Title"] = data["Name"].str.extract(r",\s*([^\.]+)\.")[0]
    data["Title"] = data["Title"].replace(
        {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    )
    data.loc[~data["Title"].isin(["Mr", "Mrs", "Miss", "Master"]), "Title"] = "Rare"
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

    # Missing values
    data["Age"] = data["Age"].fillna(data.groupby("Title")["Age"].transform("median"))
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())

    # Encoding
    data["Gender"] = data["Gender"].map({"male": 0, "female": 1})
    data = pd.get_dummies(data, columns=["Embarked", "Title"], dtype=int)

    # Drop columns that are identifiers or have too many missing values
    return data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])


data = preprocess(df)
X = data.drop(columns=["Survived"])
y = data["Survived"]
print("\nFeatures used:", list(X.columns))

# 3. Split ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train decision tree ---------------------------------------------------------
model = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate --------------------------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- Evaluation on test set ---")
print(f"Train accuracy : {model.score(X_train, y_train):.4f}")
print(f"Test accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision      : {precision_score(y_test, y_pred):.4f}")
print(f"Recall         : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score       : {f1_score(y_test, y_pred):.4f}")
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n",
      classification_report(y_test, y_pred, target_names=["Died", "Survived"]))

# 6. Plots: confusion matrix, precision-recall curve, tree, feature importance ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["Died", "Survived"], cmap="Blues", ax=axes[0]
)
axes[0].set_title("Confusion Matrix")
PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=axes[1])
axes[1].set_title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix_pr_curve.png")
plt.close()

plt.figure(figsize=(22, 10))
plot_tree(model, feature_names=X.columns, class_names=["Died", "Survived"],
          filled=True, max_depth=3, fontsize=9)
plt.title("Decision Tree (top 3 levels)")
plt.savefig(OUTPUT_DIR / "decision_tree.png", bbox_inches="tight")
plt.close()

importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()
importance.plot(kind="barh", figsize=(8, 6), title="Feature importance")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance.png")
plt.close()
print("Top features:\n", importance.sort_values(ascending=False).head(5).round(3))

# Text version of the tree rules
(OUTPUT_DIR / "tree_rules.txt").write_text(export_text(model, feature_names=list(X.columns)))

# 7. Save model --------------------------------------------------------------
joblib.dump({"model": model, "columns": list(X.columns)}, OUTPUT_DIR / "decision_tree_model.joblib")
print("\nOutputs saved to:", OUTPUT_DIR)
print("Experiment 3 completed successfully.")
