"""
Simulate production traffic through the monitored model so the logs and Grafana dashboard have data.

Phase 1 (normal)  : inputs look like the training data        -> high accuracy, no drift
Phase 2 (drift)   : sensor fault adds +2.5 cm to petal length -> drift alert, accuracy drops
Some invalid inputs are also sent to exercise the error metrics.

Usage:
    python simulate_traffic.py                  # run both phases once, print results
    python simulate_traffic.py --serve          # also expose metrics on :8000 and keep simulating
                                                # (used by docker-compose so Prometheus can scrape it)
"""

import argparse
import random
import time
from pathlib import Path

import pandas as pd

import monitor

BASE_DIR = Path(__file__).resolve().parent
FEATURES = monitor.FEATURES

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=150, help="predictions per phase")
parser.add_argument("--delay", type=float, default=0.0, help="seconds between predictions")
parser.add_argument("--serve", action="store_true", help="expose Prometheus metrics on port 8000 and loop forever")
args = parser.parse_args()

data = pd.read_csv(BASE_DIR / "data" / "Iris.csv")


def run_phase(name, petal_shift):
    correct = total = 0
    live_accuracy = None
    for _ in range(args.n):
        row = data.sample(1).iloc[0]
        record = {f: round(float(row[f]) + random.gauss(0, 0.1), 2) for f in FEATURES}
        record["PetalLengthCm"] = round(record["PetalLengthCm"] + petal_shift, 2)
        result = monitor.predict(record)
        live_accuracy = monitor.feedback(result["request_id"], row["Species"])   # true label arrives later
        correct += result["prediction"] == row["Species"]
        total += 1
        time.sleep(args.delay)
    print(f"{name:25s} predictions={total}  accuracy={correct / total:.2%}  live_accuracy_metric={live_accuracy}")


def run_once():
    run_phase("Phase 1 - normal traffic", petal_shift=0.0)
    for bad in [{"SepalLengthCm": 5.0}, {f: "abc" for f in FEATURES}]:
        try:
            monitor.predict(bad)
        except ValueError as err:
            print("Invalid input rejected ->", err)
    run_phase("Phase 2 - drifted data", petal_shift=2.5)


if args.serve:
    monitor.start_metrics_server(8000)
    print("Prometheus metrics at http://localhost:8000/metrics (Ctrl+C to stop)")
    while True:
        run_once()
else:
    run_once()
    print("\nDone. Check logs/app.log for drift warnings, logs/predictions.jsonl for the prediction log.")
    print("Run with --serve (or docker compose up) to watch the metrics in Prometheus/Grafana.")
