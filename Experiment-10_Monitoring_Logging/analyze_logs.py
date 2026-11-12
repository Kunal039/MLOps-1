"""
Offline analysis of the prediction log (logs/predictions.jsonl) written by app.py.
Produces a summary and plots of prediction volume, confidence and input drift over time.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "logs" / "predictions.jsonl"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

records = [json.loads(line) for line in LOG_FILE.read_text().splitlines() if line.strip()]
df = pd.json_normalize(records)
print(f"Predictions logged: {len(df)}")
print("\nClass distribution:\n", df["prediction"].value_counts())
print("\nConfidence: mean={:.3f}, min={:.3f}".format(df["confidence"].mean(), df["confidence"].min()))
print("Low-confidence predictions (<0.6):", (df["confidence"] < 0.6).sum())
print("Latency (ms): p50={:.2f}, p95={:.2f}".format(df["latency_ms"].median(), df["latency_ms"].quantile(0.95)))

baseline = json.loads((BASE_DIR / "model" / "baseline.json").read_text())
fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
axes[0].plot(df.index, df["confidence"].rolling(20).mean())
axes[0].set_ylabel("Confidence (rolling 20)")
axes[0].set_title("Model confidence over time")

col = "input.PetalLengthCm"
axes[1].plot(df.index, df[col].rolling(20).mean(), label="Recent mean")
axes[1].axhline(baseline["mean"]["PetalLengthCm"], color="red", linestyle="--", label="Training mean")
axes[1].set_ylabel("Petal length")
axes[1].set_title("Input drift on PetalLengthCm")
axes[1].legend()

for cls in df["prediction"].unique():
    axes[2].plot(df.index, (df["prediction"] == cls).rolling(30).mean(), label=cls)
axes[2].set_ylabel("Share of predictions")
axes[2].set_xlabel("Request number")
axes[2].set_title("Prediction class distribution (rolling 30)")
axes[2].legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "monitoring_report.png")
print("\nPlot saved to outputs/monitoring_report.png")
