"""
House Price Prediction System - web application
    GET  /               HTML form for users
    POST /api/predict    JSON API: {"MedInc": 8.3, "HouseAge": 41, ... }
    GET  /health
Run:  python train.py   then   python app.py   ->  open http://127.0.0.1:5000
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from features import FeatureEngineer  # noqa: F401  (needed to unpickle the pipeline)

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "model" / "house_price_model.joblib")
info = json.loads((BASE_DIR / "model" / "model_info.json").read_text())
FEATURES = info["features"]
LABELS = {
    "MedInc": "Median income of the area (in $10,000s)",
    "HouseAge": "Median house age (years)",
    "AveRooms": "Average rooms per house",
    "AveBedrms": "Average bedrooms per house",
    "Population": "Population of the block",
    "AveOccup": "Average people per house",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
}
DEFAULTS = {"MedInc": 8.3, "HouseAge": 41, "AveRooms": 6.9, "AveBedrms": 1.0,
            "Population": 322, "AveOccup": 2.6, "Latitude": 37.88, "Longitude": -122.23}

app = Flask(__name__)


def predict_price(values):
    missing = [f for f in FEATURES if f not in values]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    row = pd.DataFrame([{f: float(values[f]) for f in FEATURES}])
    if (row[["AveRooms", "AveOccup", "MedInc"]] <= 0).any(axis=None):
        raise ValueError("MedInc, AveRooms and AveOccup must be positive")
    return max(float(model.predict(row)[0]) * 100_000, 15_000)


@app.route("/", methods=["GET", "POST"])
def home():
    values, price, error = dict(DEFAULTS), None, None
    if request.method == "POST":
        values = request.form.to_dict()
        try:
            price = predict_price(values)
        except ValueError as err:
            error = str(err)
    return render_template("index.html", features=FEATURES, labels=LABELS, values=values,
                           price=price, error=error, metrics=info["metrics"])


@app.post("/api/predict")
def api_predict():
    try:
        price = predict_price(request.get_json(silent=True) or {})
    except ValueError as err:
        return jsonify(error=str(err)), 400
    return jsonify(predicted_price_usd=round(price, 2), model=info["model"])


@app.get("/health")
def health():
    return jsonify(status="ok", metrics=info["metrics"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
