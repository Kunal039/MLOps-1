"""
Experiment 19: Sample project - Fake News Detection using Machine Learning and Flask API
========================================================================================
NLP pipeline:  text cleaning -> TF-IDF (word 1-2 grams) -> classifier
Models compared: Multinomial Naive Bayes, Logistic Regression, Passive-Aggressive, Linear SVM
Evaluation: accuracy, precision, recall, F1, confusion matrix, 5-fold CV
The best pipeline is saved and served by a Flask REST API (JSON only):
    POST /api/predict      {"text": "..."}                       -> label + probability
    POST /api/batch        {"articles": ["...", "..."]}           -> list of results
    GET  /api/health       model information

Dataset priority:
  1. Kaggle "Fake and Real News" dataset   -> data/Fake.csv + data/True.csv
     https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
  2. "fake_or_real_news.csv" (columns title, text, label = FAKE/REAL) -> data/fake_or_real_news.csv
  3. WELFake dataset                       -> data/WELFake_Dataset.csv  (label 1 = real, 0 = fake as in the source)
  4. Otherwise a built-in sample dataset is generated (data/news_sample.csv) so the project runs offline.
     It imitates the writing style of real reporting vs. fake/clickbait news. For your final
     submission, download one of the real datasets above for meaningful results.

Usage:
    python fake_news_detection.py                  -> train, then classify a few example texts
    python fake_news_detection.py train            -> train and save the model only
    python fake_news_detection.py serve            -> start the API on http://127.0.0.1:5000
    python -m pytest fake_news_detection.py -v     -> API tests
"""

import argparse
import json
import logging
import random
import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "fake_news_model.joblib"
MAX_CHARS = 20000
pd.set_option("display.width", 200)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("fake-news-api")


# ===========================================================================
# 1. Dataset
# ===========================================================================
TOPICS = ["the new health policy", "the election results", "the vaccine programme", "the national budget",
          "the climate summit", "the central bank", "the education reform", "the stock market",
          "the space mission", "the trade agreement", "the city metro project", "the cricket board",
          "a new smartphone", "the electricity tariff", "the water supply scheme", "a celebrity wedding"]
OFFICIALS = ["the health minister", "the finance ministry", "a government spokesperson", "the election commission",
             "the World Health Organization", "university researchers", "the police department", "the company CEO",
             "the Reserve Bank", "an independent auditor"]
SOURCES = ["Reuters", "PTI", "Associated Press", "BBC", "The Hindu", "an official press release",
           "a peer-reviewed study", "the annual report"]
NUMBERS = ["2.4 percent", "1.2 million", "340 crore", "15 districts", "7 percent", "12,000 people", "3 years"]

REAL_TEMPLATES = [
    "{official} said on {day} that {topic} will be reviewed next month, according to {source}.",
    "According to {source}, {topic} recorded a change of {number} compared with last year.",
    "{official} confirmed that a committee has been formed to study {topic}. A report is expected by {month}.",
    "Data released by {source} shows {topic} affected about {number}, {official} told reporters on {day}.",
    "In a statement, {official} said the details of {topic} would be published after consultation with experts.",
    "{source} reported that {topic} was discussed in parliament on {day}; opposition members asked for more data.",
    "Officials said {topic} is on schedule. {official} added that funding of {number} has been approved.",
    "A study cited by {source} found mixed results for {topic}, and researchers said more evidence is needed.",
]
FAKE_TEMPLATES = [
    "SHOCKING: {official} secretly admits {topic} is a total scam!!! Share before they delete this!",
    "You won't BELIEVE what they are hiding about {topic}. Doctors HATE this one trick!",
    "BREAKING!!! {topic} is a plot to control your mind, insiders reveal. Mainstream media is silent!",
    "Leaked video PROVES {official} lied about {topic}. Forward to everyone you know!!!",
    "They don't want you to know: {topic} will make you {number} richer overnight. Click now!",
    "Miracle cure discovered in {topic} - big pharma is FURIOUS. 100% guaranteed, no side effects!",
    "URGENT: {topic} cancelled forever, says anonymous source. Government in total panic!!!",
    "EXPOSED: the truth about {topic} that {official} tried to bury. Wake up people!",
]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
MONTHS = ["March", "June", "September", "December"]


def _sample_dataset(n_per_class=700, seed=1):
    rng = random.Random(seed)

    def fill(t):
        return t.format(official=rng.choice(OFFICIALS), topic=rng.choice(TOPICS), source=rng.choice(SOURCES),
                        number=rng.choice(NUMBERS), day=rng.choice(DAYS), month=rng.choice(MONTHS))

    rows = []
    for _ in range(n_per_class):
        real = " ".join(fill(rng.choice(REAL_TEMPLATES)) for _ in range(rng.randint(2, 4)))
        fake = " ".join(fill(rng.choice(FAKE_TEMPLATES)) for _ in range(rng.randint(1, 3)))
        # make it harder, like real data:
        # 30% of fake news is written in a calm, "official" tone with one false claim inside
        if rng.random() < 0.30:
            claim = fill(rng.choice(FAKE_TEMPLATES)).lower().replace("!", "")
            fake = fill(rng.choice(REAL_TEMPLATES)).replace("according to", "an unnamed insider claims") + " " + claim
        # 20% of real news has a dramatic headline
        if rng.random() < 0.20:
            real = rng.choice(["BREAKING: ", "Big shock for markets! ", "Explained: the truth about "]) + real
        # 3% annotation errors (real datasets contain mislabelled articles)
        if rng.random() < 0.03:
            real, fake = fake, real
        rows += [{"text": real, "label": 0}, {"text": fake, "label": 1}]
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


def load_news():
    """Returns a DataFrame with columns: text, label  (label: 1 = FAKE, 0 = REAL)."""
    if (DATA_DIR / "Fake.csv").exists() and (DATA_DIR / "True.csv").exists():
        fake = pd.read_csv(DATA_DIR / "Fake.csv").assign(label=1)
        real = pd.read_csv(DATA_DIR / "True.csv").assign(label=0)
        # True.csv articles start with "CITY (Reuters) -" which would leak the label - remove it
        real["text"] = real["text"].str.replace(r"^.*?\(Reuters\)\s*-\s*", "", regex=True)
        df = pd.concat([fake, real])
        df["text"] = df["title"].fillna("") + ". " + df["text"].fillna("")
        print("Using Kaggle Fake/True news dataset")
    elif (DATA_DIR / "fake_or_real_news.csv").exists():
        df = pd.read_csv(DATA_DIR / "fake_or_real_news.csv")
        df["text"] = df["title"].fillna("") + ". " + df["text"].fillna("")
        df["label"] = (df["label"].str.upper() == "FAKE").astype(int)
        print("Using fake_or_real_news dataset")
    elif (DATA_DIR / "WELFake_Dataset.csv").exists():
        df = pd.read_csv(DATA_DIR / "WELFake_Dataset.csv")
        df["text"] = df["title"].fillna("") + ". " + df["text"].fillna("")
        df["label"] = 1 - df["label"]
        print("Using WELFake dataset")
    else:
        sample = DATA_DIR / "news_sample.csv"
        if not sample.exists():
            DATA_DIR.mkdir(exist_ok=True)
            _sample_dataset().to_csv(sample, index=False)
        df = pd.read_csv(sample)
        print("Using built-in sample dataset (download the Kaggle dataset for real results)")
    return df[["text", "label"]].dropna().drop_duplicates(subset="text").reset_index(drop=True)


# ===========================================================================
# 2. Text cleaning (used inside the TF-IDF vectorizer of the saved pipeline)
# ===========================================================================
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for", "with", "about", "to", "from",
    "in", "on", "is", "are", "was", "were", "be", "been", "it", "its", "this", "that", "these", "those",
    "as", "has", "have", "had", "will", "would", "can", "could", "he", "she", "they", "we", "you", "i",
}


def clean_text(text):
    text = str(text)
    exclamations = " exclamationmark" * min(text.count("!"), 3)       # keep a signal of sensational style
    caps = " allcapsword" * min(len(re.findall(r"\b[A-Z]{4,}\b", text)), 3)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)            # links
    text = re.sub(r"<.*?>", " ", text)                                # html tags
    text = re.sub(r"[^a-z\s]", " ", text)                             # numbers / punctuation
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(words) + exclamations + caps


# ===========================================================================
# 3. Training
# ===========================================================================
def make(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), min_df=2,
                                  max_df=0.9, max_features=50000, sublinear_tf=True)),
        ("clf", clf),
    ])


def train():
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    df = load_news()
    print(f"Articles: {len(df)} | fake: {df.label.sum()} | real: {(df.label == 0).sum()}")
    print("Example:", df.text.iloc[0][:150], "...")
    print("Cleaned:", clean_text(df.text.iloc[0])[:150], "...")

    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2,
                                                        random_state=42, stratify=df["label"])
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

    joblib.dump(model, MODEL_PATH)
    (MODEL_DIR / "model_info.json").write_text(json.dumps(
        {"model": best, "vectorizer": "TF-IDF (1-2 grams)", "metrics": results.loc[best].round(4).to_dict(),
         "trained_on": len(X_train)}, indent=2))
    print("\nModel saved to model/fake_news_model.joblib")
    _loaded.clear()                   # make the API pick up the new model


# ===========================================================================
# 4. Prediction
# ===========================================================================
_loaded = {}


def load_model():
    """Load the saved model once (training it first if it does not exist yet)."""
    if not _loaded:
        if not MODEL_PATH.exists():
            train()
        _loaded["model"] = joblib.load(MODEL_PATH)
        _loaded["info"] = json.loads((MODEL_DIR / "model_info.json").read_text())
    return _loaded["model"], _loaded["info"]


def classify(text):
    if not isinstance(text, str) or len(text.strip()) < 20:
        raise ValueError("Text must be at least 20 characters long")
    model, _ = load_model()
    text = text[:MAX_CHARS]
    if hasattr(model, "predict_proba"):
        p_fake = float(model.predict_proba([text])[0, 1])
    else:
        p_fake = float(1 / (1 + np.exp(-model.decision_function([text])[0])))
    label = "FAKE" if p_fake >= 0.5 else "REAL"
    log.info("Predicted %s (p_fake=%.3f) for text of %d chars", label, p_fake, len(text))
    return {"label": label, "fake_probability": round(p_fake, 4),
            "confidence": round(max(p_fake, 1 - p_fake), 4)}


# ===========================================================================
# 5. Flask REST API
# ===========================================================================
app = Flask(__name__)


@app.post("/api/predict")
def api_predict():
    try:
        return jsonify(classify((request.get_json(silent=True) or {}).get("text")))
    except ValueError as err:
        return jsonify(error=str(err)), 400


@app.post("/api/batch")
def api_batch():
    articles = (request.get_json(silent=True) or {}).get("articles")
    if not isinstance(articles, list) or not 1 <= len(articles) <= 100:
        return jsonify(error="'articles' must be a list of 1-100 strings"), 400
    results = []
    for a in articles:
        try:
            results.append(classify(a))
        except ValueError as err:
            results.append({"error": str(err)})
    return jsonify(results=results)


@app.get("/api/health")
def health():
    _, info = load_model()
    return jsonify(status="ok", **info)


# ===========================================================================
# 6. API tests  (python -m pytest fake_news_detection.py -v)
# ===========================================================================
REAL = ("The finance ministry said on Tuesday that the national budget will be reviewed next month, "
        "according to Reuters. A committee has been formed to study the proposal.")
FAKE = ("SHOCKING!!! The health minister secretly admits the vaccine programme is a total SCAM. "
        "Share before they delete this! Mainstream media is silent!!!")


@pytest.fixture
def client():
    return app.test_client()


def test_health(client):
    assert client.get("/api/health").json["status"] == "ok"


def test_real_article(client):
    assert client.post("/api/predict", json={"text": REAL}).json["label"] == "REAL"


def test_fake_article(client):
    assert client.post("/api/predict", json={"text": FAKE}).json["label"] == "FAKE"


def test_batch(client):
    labels = [r["label"] for r in client.post("/api/batch", json={"articles": [REAL, FAKE]}).json["results"]]
    assert labels == ["REAL", "FAKE"]


def test_short_text_rejected(client):
    assert client.post("/api/predict", json={"text": "hi"}).status_code == 400


# ===========================================================================
# Command line
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("train", help="train and save the model")
    sub.add_parser("serve", help="start the REST API on port 5000")
    args = parser.parse_args()

    if args.command in (None, "train"):
        train()
    if args.command is None:
        print("\n=== Example predictions ===")
        for text in [REAL, FAKE]:
            r = classify(text)
            print(f"{r['label']} ({r['confidence']:.0%}) <- {text[:70]}...")
    if args.command == "serve":
        load_model()
        app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    # Run through the imported module so clean_text is pickled as fake_news_detection.clean_text
    # (not __main__.clean_text) and the saved model can be loaded by the tests and the API.
    import fake_news_detection
    fake_news_detection.main()
