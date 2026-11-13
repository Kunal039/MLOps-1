"""
Experiment 18: Sample project - Student Performance Prediction System using Machine Learning
=============================================================================================
Two prediction tasks on the same data:
    A. Regression     - predict the FINAL SCORE (0-100)
                        Linear Regression, Decision Tree, Random Forest, SVR
    B. Classification - predict PASS / FAIL (early-warning system for at-risk students)
                        Logistic Regression, Decision Tree, Random Forest, SVM, KNN
Pipeline: cleaning -> imputation + encoding + scaling (ColumnTransformer) -> 5-fold CV
          -> test evaluation -> best models saved -> predictions with explainable study advice

Data: data/student_performance.csv (1,000 students). The scores follow realistic relationships
(study hours, attendance and previous marks matter most; sleep, tutoring and parental education
help; heavy social-media use hurts), plus random noise, a few missing values and some inconsistent
entries to clean. Replace the CSV with your own college data if available (keep the same column names).

Usage:
    python student_performance_prediction.py                       -> (generate data if missing) train + predict
    python student_performance_prediction.py generate              -> re-create the dataset
    python student_performance_prediction.py train                 -> train and save the models
    python student_performance_prediction.py predict               -> three example students
    python student_performance_prediction.py predict --interactive -> enter your own details
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (RocCurveDisplay, accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_performance.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "student_models.joblib"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

NUMERIC = ["age", "study_hours_per_week", "attendance_pct", "previous_score", "assignments_completed",
           "sleep_hours", "social_media_hours"]
BINARY = ["gender", "internet_access", "extracurricular", "tutoring"]
ORDINAL = ["parental_education"]
EDU_LEVELS = [["High School", "Bachelor", "Master", "PhD"]]


# ===========================================================================
# Dataset generation
# ===========================================================================
def generate(n=1000, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "student_id": [f"S{1000 + i}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "age": rng.integers(17, 23, n),
        "study_hours_per_week": np.clip(rng.normal(14, 6, n), 0, 40).round(1),
        "attendance_pct": np.clip(rng.normal(82, 11, n), 35, 100).round(1),
        "previous_score": np.clip(rng.normal(66, 13, n), 25, 100).round(0),
        "assignments_completed": rng.integers(3, 11, n),
        "sleep_hours": np.clip(rng.normal(7, 1.1, n), 4, 10).round(1),
        "social_media_hours": np.clip(rng.gamma(2, 1.4, n), 0, 10).round(1),
        "parental_education": rng.choice(["High School", "Bachelor", "Master", "PhD"], n, p=[.35, .4, .2, .05]),
        "internet_access": rng.choice(["Yes", "No"], n, p=[.88, .12]),
        "extracurricular": rng.choice(["Yes", "No"], n, p=[.45, .55]),
        "tutoring": rng.choice(["Yes", "No"], n, p=[.3, .7]),
    })
    edu = df["parental_education"].map({"High School": 0, "Bachelor": 1.5, "Master": 3, "PhD": 4})
    score = (0
             + 0.95 * df["study_hours_per_week"]
             + 0.22 * df["attendance_pct"]
             + 0.38 * df["previous_score"]
             + 1.1 * df["assignments_completed"]
             - 1.6 * (df["sleep_hours"] - 7.5) ** 2
             - 1.3 * df["social_media_hours"]
             + edu
             + 3.5 * (df["tutoring"] == "Yes")
             + 2.0 * (df["internet_access"] == "Yes")
             + 1.0 * (df["extracurricular"] == "Yes")
             + rng.normal(0, 5, n))
    df["final_score"] = np.clip(score, 0, 100).round(1)
    df["result"] = np.where(df["final_score"] >= 55, "Pass", "Fail")

    # data-quality problems
    for col in ["study_hours_per_week", "attendance_pct", "sleep_hours", "parental_education"]:
        df.loc[rng.choice(n, 25, replace=False), col] = np.nan
    df.loc[rng.choice(n, 15, replace=False), "gender"] = rng.choice(["male", "F", "M ", "female"], 15)
    return pd.concat([df, df.sample(8, random_state=1)])     # duplicates


def save_dataset():
    DATA_PATH.parent.mkdir(exist_ok=True)
    data = generate()
    data.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(data)} rows to {DATA_PATH}")
    print(data["result"].value_counts())


# ===========================================================================
# Training
# ===========================================================================
def train():
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    # 1. Load + clean ----------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows. Missing values:\n{df.isna().sum()[df.isna().sum() > 0]}")
    df = df.drop_duplicates(subset="student_id")
    df["gender"] = df["gender"].str.strip().str.upper().str[0].map({"M": "Male", "F": "Female"})
    print(f"After removing duplicates: {len(df)} rows | pass rate {df['result'].eq('Pass').mean():.1%}")

    print("\nCorrelation with final score:")
    print(df[NUMERIC + ["final_score"]].corr()["final_score"].drop("final_score")
          .sort_values(ascending=False).round(3))

    X = df[NUMERIC + BINARY + ORDINAL]
    preprocess = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("bin", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(drop="if_binary"))]), BINARY),
        ("ord", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("encode", OrdinalEncoder(categories=EDU_LEVELS))]), ORDINAL),
    ])

    # ===============================================================================
    # A. REGRESSION - final score
    # ===============================================================================
    print("\n" + "=" * 70 + "\nA. REGRESSION: predict final score\n" + "=" * 70)
    y_reg = df["final_score"]
    Xtr, Xte, ytr, yte = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    regressors = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "SVR": SVR(C=30, epsilon=1),
    }
    reg_rows, reg_models = [], {}
    for name, est in regressors.items():
        pipe = Pipeline([("prep", preprocess), ("model", est)])
        cv = -cross_val_score(pipe, Xtr, ytr, cv=KFold(5, shuffle=True, random_state=42),
                              scoring="neg_root_mean_squared_error").mean()
        pipe.fit(Xtr, ytr)
        p = pipe.predict(Xte)
        reg_models[name] = pipe
        reg_rows.append({"Model": name, "CV_RMSE": cv, "Test_RMSE": np.sqrt(mean_squared_error(yte, p)),
                         "Test_MAE": mean_absolute_error(yte, p), "Test_R2": r2_score(yte, p)})
    reg_results = pd.DataFrame(reg_rows).set_index("Model").sort_values("Test_RMSE")
    print(reg_results.round(3))
    best_reg = reg_results.index[0]

    # ===============================================================================
    # B. CLASSIFICATION - pass / fail
    # ===============================================================================
    print("\n" + "=" * 70 + "\nB. CLASSIFICATION: predict Pass / Fail\n" + "=" * 70)
    y_clf = (df["result"] == "Pass").astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, class_weight="balanced",
                                                random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                                random_state=42, n_jobs=-1),
        "SVM": SVC(C=3, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=15),
    }
    clf_rows, clf_models = [], {}
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, est in classifiers.items():
        pipe = Pipeline([("prep", preprocess), ("model", est)])
        cv = cross_val_score(pipe, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                             scoring="f1").mean()
        pipe.fit(Xtr, ytr)
        p = pipe.predict(Xte)
        scores = pipe.decision_function(Xte) if hasattr(pipe, "decision_function") else pipe.predict_proba(Xte)[:, 1]
        clf_models[name] = pipe
        # recall of the FAIL class matters most: we want to catch at-risk students
        clf_rows.append({"Model": name, "CV_F1": cv, "Accuracy": accuracy_score(yte, p),
                         "Precision": precision_score(yte, p), "Recall": recall_score(yte, p),
                         "F1": f1_score(yte, p), "Fail_recall": recall_score(yte, p, pos_label=0),
                         "ROC_AUC": roc_auc_score(yte, scores)})
        RocCurveDisplay.from_predictions(yte, scores, name=name, ax=ax)
    ax.set_title("ROC curves - Pass/Fail")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curves.png")
    plt.close()
    clf_results = pd.DataFrame(clf_rows).set_index("Model").sort_values("ROC_AUC", ascending=False)
    print(clf_results.round(3))
    best_clf = clf_results.index[0]

    # Feature importance from the Random Forest regressor
    rf = reg_models["Random Forest"]
    names = [n.split("__")[1] for n in rf.named_steps["prep"].get_feature_names_out()]
    imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names).sort_values()
    imp.plot(kind="barh", figsize=(8, 6), title="What drives student performance? (Random Forest)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png")
    plt.close()

    # Fairness check: is the pass/fail model equally accurate for male and female students?
    pred_all = clf_models[best_clf].predict(Xte)
    fair = pd.DataFrame({"gender": Xte["gender"].fillna("Unknown").values, "correct": pred_all == yte.values})
    print("\nAccuracy by gender (fairness check):\n", fair.groupby("gender")["correct"].mean().round(3))

    reg_results.to_csv(OUTPUT_DIR / "regression_results.csv")
    clf_results.to_csv(OUTPUT_DIR / "classification_results.csv")
    joblib.dump({"regressor": reg_models[best_reg], "classifier": clf_models[best_clf],
                 "features": NUMERIC + BINARY + ORDINAL}, MODEL_PATH)
    (MODEL_DIR / "model_info.json").write_text(json.dumps({
        "regressor": best_reg, "regression_metrics": reg_results.loc[best_reg].round(3).to_dict(),
        "classifier": best_clf, "classification_metrics": clf_results.loc[best_clf].round(3).to_dict()}, indent=2))
    print(f"\nBest regressor: {best_reg} | best classifier: {best_clf} -> saved to model/")


# ===========================================================================
# Prediction with explainable study advice
# ===========================================================================
EXAMPLES = {
    "Hard-working student": dict(gender="Female", age=19, study_hours_per_week=22, attendance_pct=95,
                                 previous_score=78, assignments_completed=10, sleep_hours=7.5,
                                 social_media_hours=1.5, parental_education="Bachelor", internet_access="Yes",
                                 extracurricular="Yes", tutoring="No"),
    "Average student": dict(gender="Male", age=20, study_hours_per_week=12, attendance_pct=80,
                            previous_score=62, assignments_completed=7, sleep_hours=7,
                            social_media_hours=3, parental_education="High School", internet_access="Yes",
                            extracurricular="No", tutoring="No"),
    "At-risk student": dict(gender="Male", age=21, study_hours_per_week=3, attendance_pct=55,
                            previous_score=40, assignments_completed=4, sleep_hours=5,
                            social_media_hours=7, parental_education="High School", internet_access="No",
                            extracurricular="No", tutoring="No"),
}


def advice(s):
    tips = []
    if s["study_hours_per_week"] < 10:
        tips.append("increase self-study to at least 10-15 hours/week")
    if s["attendance_pct"] < 75:
        tips.append("attend more classes (attendance below 75%)")
    if s["social_media_hours"] > 4:
        tips.append("reduce social-media time")
    if not 6.5 <= s["sleep_hours"] <= 8.5:
        tips.append("aim for 7-8 hours of sleep")
    if s["assignments_completed"] < 7:
        tips.append("complete all assignments")
    return tips or ["keep up the good work!"]


def predict(bundle, student):
    row = pd.DataFrame([student])[bundle["features"]]
    score = float(bundle["regressor"].predict(row)[0])
    passes = int(bundle["classifier"].predict(row)[0]) == 1
    return min(max(score, 0), 100), passes


def ask(prompt, cast=str, default=None):
    value = input(f"{prompt} [{default}]: ").strip()
    return cast(value) if value else default


def predict_students(interactive=False):
    bundle = joblib.load(MODEL_PATH)
    if interactive:
        base = EXAMPLES["Average student"]
        students = {"You": {k: ask(k, type(v), v) for k, v in base.items()}}
    else:
        students = EXAMPLES
    for name, s in students.items():
        score, passes = predict(bundle, s)
        status = "PASS" if passes else "AT RISK OF FAILING"
        print(f"\n{name}: predicted final score {score:.1f}/100 -> {status}")
        print("  Advice:", "; ".join(advice(s)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("generate", help="re-create data/student_performance.csv")
    sub.add_parser("train", help="train and save the models")
    p = sub.add_parser("predict", help="predict example students (or your own with --interactive)")
    p.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    if args.command == "generate" or (args.command is None and not DATA_PATH.exists()):
        save_dataset()
    if args.command in (None, "train"):
        train()
    if args.command in (None, "predict"):
        predict_students(getattr(args, "interactive", False))


if __name__ == "__main__":
    main()
