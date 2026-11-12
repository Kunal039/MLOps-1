"""
Minimal prediction service started by the container's default CMD.
    GET  /health   -> container health check
    POST /predict  -> {"features": [5.1, 3.5, 1.4, 0.2]}
"""

import json
from pathlib import Path

from flask import Flask, jsonify, request

from predict import load_model, predict

app = Flask(__name__)
model = load_model()
model_info = json.loads((Path(__file__).resolve().parent / "model" / "model_info.json").read_text())


@app.get("/health")
def health():
    return jsonify(status="ok", model=model_info)


@app.post("/predict")
def predict_endpoint():
    features = (request.get_json(silent=True) or {}).get("features")
    if not isinstance(features, list) or len(features) != 4:
        return jsonify(error="Send JSON like {'features': [5.1, 3.5, 1.4, 0.2]}"), 400
    label, conf = predict(model, [features])[0]
    return jsonify(prediction=label, confidence=round(float(conf), 4))


if __name__ == "__main__":
    # 0.0.0.0 makes the server reachable from outside the container
    app.run(host="0.0.0.0", port=5000)
