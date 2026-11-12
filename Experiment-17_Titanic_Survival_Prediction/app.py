"""
Titanic Survival Prediction - web app + REST API
    GET/POST /          HTML form
    POST /api/predict   {"Pclass": 1, "Gender": "female", "Age": 29, "SibSp": 0, "Parch": 0, "Fare": 100, "Embarked": "S"}
Run: python train.py  then  python app.py  ->  http://127.0.0.1:5000
"""

import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from predict import predict

BASE_DIR = Path(__file__).resolve().parent
info = json.loads((BASE_DIR / "model" / "model_info.json").read_text())
app = Flask(__name__)
DEFAULTS = {"Pclass": "3", "Gender": "male", "Age": "25", "SibSp": "0", "Parch": "0",
            "Fare": "15", "Embarked": "S", "Title": "Mr"}


@app.route("/", methods=["GET", "POST"])
def home():
    values, result, error = dict(DEFAULTS), None, None
    if request.method == "POST":
        values = request.form.to_dict()
        try:
            label, prob = predict(values)
            result = {"label": label, "prob": prob}
        except (KeyError, ValueError) as err:
            error = f"Invalid input: {err}"
    return render_template("index.html", v=values, result=result, error=error, info=info)


@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True) or {}
    if data.get("Gender") not in ("male", "female") or data.get("Pclass") not in (1, 2, 3, "1", "2", "3"):
        return jsonify(error="Pclass must be 1/2/3 and Gender male/female"), 400
    try:
        label, prob = predict(data)
    except (KeyError, ValueError) as err:
        return jsonify(error=str(err)), 400
    return jsonify(prediction=label, survival_probability=round(prob, 4))


@app.get("/health")
def health():
    return jsonify(status="ok", **info)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
