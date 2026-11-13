"""
Experiment 17: Sample project - Titanic Survival Prediction using Decision Tree Classifier
==========================================================================================
Steps: EDA -> feature-engineering pipeline -> Decision Tree -> GridSearchCV tuning
       -> evaluation (accuracy, precision, recall, F1, ROC-AUC, confusion matrix)
       -> tree visualisation and rules -> save pipeline -> predict survival

Usage:
    python titanic_survival_prediction.py            -> train, then predict a few example passengers
    python titanic_survival_prediction.py train      -> train and save the model only
    python titanic_survival_prediction.py predict    -> predict the example passengers with the saved model
    python titanic_survival_prediction.py predict --pclass 1 --gender female --age 29 --fare 100
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, classification_report,
                             f1_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "titanic_tree.joblib"

RAW_COLUMNS = ["Pclass", "Name", "Gender", "Age", "SibSp", "Parch", "Fare", "Embarked"]


# ===========================================================================
# Feature engineering (part of the saved pipeline)
# ===========================================================================
class TitanicFeatures(BaseEstimator, TransformerMixin):
    """Cleans raw passenger records and creates model features.
    Medians/modes are learnt in fit() from the training data only (no leakage)."""

    def fit(self, X, y=None):
        X = self._titles(X)
        self.age_by_title_ = X.groupby("Title")["Age"].median().to_dict()
        self.age_median_ = X["Age"].median()
        self.fare_median_ = X["Fare"].median()
        self.embarked_mode_ = X["Embarked"].mode()[0]
        return self

    @staticmethod
    def _titles(X):
        X = X.copy()
        title = X["Name"].fillna("").str.extract(r",\s*([^\.]+)\.")[0]
        title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
        X["Title"] = title.where(title.isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
        return X

    def transform(self, X):
        X = self._titles(X)
        X["Age"] = X["Age"].fillna(X["Title"].map(self.age_by_title_)).fillna(self.age_median_)
        X["Fare"] = X["Fare"].fillna(self.fare_median_)
        X["Embarked"] = X["Embarked"].fillna(self.embarked_mode_)
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
        X["IsFemale"] = (X["Gender"].str.lower() == "female").astype(int)
        X["IsChild"] = (X["Age"] < 13).astype(int)
        out = X[["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "IsFemale", "IsChild"]].copy()
        for title in ["Mr", "Mrs", "Miss", "Master", "Rare"]:
            out[f"Title_{title}"] = (X["Title"] == title).astype(int)
        for port in ["C", "Q", "S"]:
            out[f"Embarked_{port}"] = (X["Embarked"] == port).astype(int)
        return out


def passenger_frame(passenger):
    """Convert a passenger dict into a one-row DataFrame with the raw columns."""
    return pd.DataFrame([{
        "Pclass": int(passenger["Pclass"]),
        "Name": passenger.get("Name") or f"Doe, {passenger.get('Title', 'Mr')}. John",
        "Gender": passenger["Gender"],
        "Age": float(passenger["Age"]) if passenger.get("Age") not in (None, "") else None,
        "SibSp": int(passenger.get("SibSp", 0)),
        "Parch": int(passenger.get("Parch", 0)),
        "Fare": float(passenger.get("Fare", 15)),
        "Embarked": passenger.get("Embarked", "S"),
    }])


# ===========================================================================
# Training
# ===========================================================================
def train():
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
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Died", "Survived"], cmap="Blues",
                                            ax=axes[0])
    RocCurveDisplay.from_predictions(y_test, prob, ax=axes[1])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "evaluation.png")
    plt.close()

    plt.figure(figsize=(24, 10))
    plot_tree(tree, feature_names=feature_names, class_names=["Died", "Survived"], filled=True, max_depth=3,
              fontsize=9)
    plt.savefig(OUTPUT_DIR / "decision_tree.png", bbox_inches="tight")
    plt.close()
    (OUTPUT_DIR / "tree_rules.txt").write_text(export_text(tree, feature_names=feature_names))

    importance = pd.Series(tree.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("Most important features:\n", importance.head(5).round(3))

    # 4. Save -----------------------------------------------------------------------
    joblib.dump(model, MODEL_PATH)
    (MODEL_DIR / "model_info.json").write_text(json.dumps(
        {"model": "DecisionTreeClassifier",
         "params": {k.replace("tree__", ""): v for k, v in grid.best_params_.items()},
         "metrics": {k: round(v, 4) for k, v in metrics.items()}}, indent=2))
    print("\nModel saved to model/titanic_tree.joblib")


# ===========================================================================
# Prediction
# ===========================================================================
EXAMPLE_PASSENGERS = {
    "Rose (1st class woman, 17)": {"Pclass": 1, "Gender": "female", "Age": 17, "SibSp": 0, "Parch": 1,
                                   "Fare": 100, "Embarked": "S", "Title": "Miss"},
    "Jack (3rd class man, 20)": {"Pclass": 3, "Gender": "male", "Age": 20, "Fare": 8, "Title": "Mr"},
    "Boy (3rd class, 6)": {"Pclass": 3, "Gender": "male", "Age": 6, "SibSp": 1, "Parch": 2,
                           "Fare": 21, "Title": "Master"},
}


def predict(model, passenger):
    prob = float(model.predict_proba(passenger_frame(passenger))[0, 1])
    return ("Survived" if prob >= 0.5 else "Did not survive"), prob


def predict_passengers(a):
    model = joblib.load(MODEL_PATH)
    if getattr(a, "pclass", None) and getattr(a, "gender", None):
        title = a.title or ("Mr" if a.gender == "male" else "Miss")
        people = {"You": {"Pclass": a.pclass, "Gender": a.gender, "Age": a.age, "SibSp": a.sibsp,
                          "Parch": a.parch, "Fare": a.fare, "Embarked": a.embarked, "Title": title}}
    else:
        people = EXAMPLE_PASSENGERS
    print()
    for name, p in people.items():
        label, prob = predict(model, p)
        print(f"{name:30s} -> {label:16s} (survival probability {prob:.0%})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("train", help="train and save the model")
    p = sub.add_parser("predict", help="predict survival with the saved model")
    p.add_argument("--pclass", type=int)
    p.add_argument("--gender", choices=["male", "female"])
    p.add_argument("--age", type=float)
    p.add_argument("--sibsp", type=int, default=0)
    p.add_argument("--parch", type=int, default=0)
    p.add_argument("--fare", type=float, default=15)
    p.add_argument("--embarked", choices=["C", "Q", "S"], default="S")
    p.add_argument("--title", choices=["Mr", "Mrs", "Miss", "Master", "Rare"])
    args = parser.parse_args()

    if args.command in (None, "train"):
        train()
    if args.command in (None, "predict"):
        predict_passengers(args)


if __name__ == "__main__":
    # Run through the imported module so TitanicFeatures is pickled as
    # titanic_survival_prediction.TitanicFeatures (not __main__.TitanicFeatures) and can be loaded anywhere.
    import titanic_survival_prediction
    titanic_survival_prediction.main()
