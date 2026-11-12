"""
Send realistic traffic to the monitored model so the Grafana dashboard has data.

Phase 1 (normal)  : inputs look like the training data      -> high accuracy, no drift
Phase 2 (drift)   : sensor fault adds +2.5 cm to petal length -> drift alert, accuracy drops
Some invalid requests are also sent to exercise the error metrics.

Usage:
    python simulate_traffic.py            # sends to a running server at http://127.0.0.1:5000
    python simulate_traffic.py --local    # uses Flask's test client (no server needed)
"""

import argparse
import random
import time
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:5000")
parser.add_argument("--local", action="store_true", help="use Flask test client instead of HTTP")
parser.add_argument("--n", type=int, default=150, help="requests per phase")
parser.add_argument("--delay", type=float, default=0.0, help="seconds between requests")
args = parser.parse_args()

if args.local:
    from app import app
    client = app.test_client()

    def post(path, payload):
        r = client.post(path, json=payload)
        return r.status_code, r.get_json()
else:
    import requests

    def post(path, payload):
        r = requests.post(args.url + path, json=payload, timeout=5)
        return r.status_code, r.json()

data = pd.read_csv(BASE_DIR / "data" / "Iris.csv")


def run_phase(name, petal_shift):
    correct = total = 0
    for _ in range(args.n):
        row = data.sample(1).iloc[0]
        payload = {f: round(float(row[f]) + random.gauss(0, 0.1), 2) for f in FEATURES}
        payload["PetalLengthCm"] = round(payload["PetalLengthCm"] + petal_shift, 2)
        status, body = post("/predict", payload)
        if status == 200:
            _, fb = post("/feedback", {"request_id": body["request_id"], "actual": row["Species"]})
            correct += body["prediction"] == row["Species"]
            total += 1
        time.sleep(args.delay)
    print(f"{name:25s} requests={total}  accuracy={correct / total:.2%}  live_accuracy_metric={fb['live_accuracy']}")


run_phase("Phase 1 - normal traffic", petal_shift=0.0)
for bad in [{"SepalLengthCm": 5.0}, {f: "abc" for f in FEATURES}]:
    print("Invalid request ->", post("/predict", bad))
run_phase("Phase 2 - drifted data", petal_shift=2.5)
print("\nDone. Check logs/app.log for drift warnings, logs/predictions.jsonl for the prediction log,")
print("and Grafana (http://localhost:3000) for the dashboards.")
