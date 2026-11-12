"""
Experiment 18: Sample project - Student Performance Prediction System using Machine Learning
=============================================================================================
Two prediction tasks on the same data:
    A. Regression     - predict the FINAL SCORE (0-100)
                        Linear Regression, Decision Tree, Random Forest, SVR
    B. Classification - predict PASS / FAIL (early-warning system for at-risk students)
                        Logistic Regression, Decision Tree, Random Forest, SVM, KNN
Pipeline: cleaning -> imputation + encoding + scaling (ColumnTransformer) -> 5-fold CV
          -> test evaluation -> best models saved for predict.py
"""

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

from generate_dataset import OUT as DATA_PATH
from generate_dataset import generate

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

NUMERIC = ["age", "study_hours_per_week", "attendance_pct", "previous_score", "assignments_completed",
           "sleep_hours", "social_media_hours"]
BINARY = ["gender", "internet_access", "extracurricular", "tutoring"]
ORDINAL = ["parental_education"]
EDU_LEVELS = [["High School", "Bachelor", "Master", "PhD"]]

# 1. Load + clean ----------------------------------------------------------------
if not DATA_PATH.exists():
    DATA_PATH.parent.mkdir(exist_ok=True)
    generate().to_csv(DATA_PATH, index=False)
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows. Missing values:\n{df.isna().sum()[df.isna().sum() > 0]}")
df = df.drop_duplicates(subset="student_id")
df["gender"] = df["gender"].str.strip().str.upper().str[0].map({"M": "Male", "F": "Female"})
print(f"After removing duplicates: {len(df)} rows | pass rate {df['result'].eq('Pass').mean():.1%}")

print("\nCorrelation with final score:")
print(df[NUMERIC + ["final_score"]].corr()["final_score"].drop("final_score").sort_values(ascending=False).round(3))

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
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                            random_state=42, n_jobs=-1),
    "SVM": SVC(C=3, class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=15),
}
clf_rows, clf_models = [], {}
fig, ax = plt.subplots(figsize=(7, 6))
for name, est in classifiers.items():
    pipe = Pipeline([("prep", preprocess), ("model", est)])
    cv = cross_val_score(pipe, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="f1").mean()
    pipe.fit(Xtr, ytr)
    p = pipe.predict(Xte)
    scores = pipe.decision_function(Xte) if hasattr(pipe, "decision_function") else pipe.predict_proba(Xte)[:, 1]
    clf_models[name] = pipe
    # recall of the FAIL class matters most: we want to catch at-risk students
    clf_rows.append({"Model": name, "CV_F1": cv, "Accuracy": accuracy_score(yte, p),
                     "Precision": precision_score(yte, p), "Recall": recall_score(yte, p), "F1": f1_score(yte, p),
                     "Fail_recall": recall_score(yte, p, pos_label=0), "ROC_AUC": roc_auc_score(yte, scores)})
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
             "features": NUMERIC + BINARY + ORDINAL}, MODEL_DIR / "student_models.joblib")
(MODEL_DIR / "model_info.json").write_text(json.dumps({
    "regressor": best_reg, "regression_metrics": reg_results.loc[best_reg].round(3).to_dict(),
    "classifier": best_clf, "classification_metrics": clf_results.loc[best_clf].round(3).to_dict()}, indent=2))
print(f"\nBest regressor: {best_reg} | best classifier: {best_clf} -> saved to model/")
