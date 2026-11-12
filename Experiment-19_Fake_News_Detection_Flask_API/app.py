"""
Fake News Detection - Flask REST API
    POST /api/predict      {"text": "..."}                       -> label + probability
    POST /api/batch        {"articles": ["...", "..."]}           -> list of results
    GET  /api/health       model information
Run: python train.py  then  python app.py   ->  http://127.0.0.1:5000
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request

from text_utils import clean_text  # noqa: F401  (needed to unpickle the pipeline)

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "model" / "fake_news_model.joblib")
info = json.loads((BASE_DIR / "model" / "model_info.json").read_text())
MAX_CHARS = 20000

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("fake-news-api")
app = Flask(__name__)


def classify(text):
    if not isinstance(text, str) or len(text.strip()) < 20:
        raise ValueError("Text must be at least 20 characters long")
    text = text[:MAX_CHARS]
    if hasattr(model, "predict_proba"):
        p_fake = float(model.predict_proba([text])[0, 1])
    else:
        p_fake = float(1 / (1 + np.exp(-model.decision_function([text])[0])))
    label = "FAKE" if p_fake >= 0.5 else "REAL"
    log.info("Predicted %s (p_fake=%.3f) for text of %d chars", label, p_fake, len(text))
    return {"label": label, "fake_probability": round(p_fake, 4),
            "confidence": round(max(p_fake, 1 - p_fake), 4)}


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
    return jsonify(status="ok", **info)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
