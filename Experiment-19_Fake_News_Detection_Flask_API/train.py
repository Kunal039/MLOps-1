"""
Experiment 19: Sample project - Fake News Detection using Machine Learning and Flask API
========================================================================================
NLP pipeline:  text cleaning -> TF-IDF (word 1-2 grams) -> classifier
Models compared: Multinomial Naive Bayes, Logistic Regression, Passive-Aggressive, Linear SVM
Evaluation: accuracy, precision, recall, F1, confusion matrix, 5-fold CV
The best pipeline is saved and served by app.py.
"""

import json
import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from dataset import load_news
from text_utils import clean_text

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
pd.set_option("display.width", 200)

df = load_news()
print(f"Articles: {len(df)} | fake: {df.label.sum()} | real: {(df.label == 0).sum()}")
print("Example:", df.text.iloc[0][:150], "...")
print("Cleaned:", clean_text(df.text.iloc[0])[:150], "...")

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2,
                                                    random_state=42, stratify=df["label"])


def make(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), min_df=2,
                                  max_df=0.9, max_features=50000, sublinear_tf=True)),
        ("clf", clf),
    ])


models = {
    "Naive Bayes": make(MultinomialNB(alpha=0.5)),
    "Logistic Regression": make(LogisticRegression(max_iter=2000, C=5)),
    "Passive-Aggressive": make(PassiveAggressiveClassifier(max_iter=1000, random_state=42)),
    "Linear SVM": make(LinearSVC(C=1)),
}
rows = []
for name, model in models.items():
    cv = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                         scoring="f1").mean()
    model.fit(X_train, y_train)
    p = model.predict(X_test)
    rows.append({"Model": name, "CV_F1": cv, "Accuracy": accuracy_score(y_test, p),
                 "Precision": precision_score(y_test, p), "Recall": recall_score(y_test, p),
                 "F1": f1_score(y_test, p)})
results = pd.DataFrame(rows).set_index("Model").sort_values("F1", ascending=False)
print("\n=== Model comparison (positive class = FAKE) ===")
print(results.round(4))
results.to_csv(OUTPUT_DIR / "model_comparison.csv")

# Logistic Regression is preferred when it is within 1% of the best: it gives probabilities + explanations
best = results.index[0]
if results.loc["Logistic Regression", "F1"] >= results["F1"].max() - 0.01:
    best = "Logistic Regression"
model = models[best]
pred = model.predict(X_test)
print(f"\nSelected model: {best}")
print(classification_report(y_test, pred, target_names=["REAL", "FAKE"]))

ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["REAL", "FAKE"], cmap="Purples")
plt.title(f"Confusion matrix - {best}")
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", bbox_inches="tight")
plt.close()

# Most informative words (transparency)
if hasattr(model.named_steps["clf"], "coef_"):
    words = np.array(model.named_steps["tfidf"].get_feature_names_out())
    coef = model.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    print("Words that indicate REAL news:", ", ".join(words[order[:12]]))
    print("Words that indicate FAKE news:", ", ".join(words[order[-12:]][::-1]))

# Error analysis
wrong = X_test[pred != y_test]
print(f"\nMisclassified articles: {len(wrong)}")
for text in wrong.head(3):
    print("  -", text[:120], "...")

joblib.dump(model, MODEL_DIR / "fake_news_model.joblib")
(MODEL_DIR / "model_info.json").write_text(json.dumps(
    {"model": best, "vectorizer": "TF-IDF (1-2 grams)", "metrics": results.loc[best].round(4).to_dict(),
     "trained_on": len(X_train)}, indent=2))
print("\nModel saved -> start the API with: python app.py")
