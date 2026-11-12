"""
Client that calls the running Flask API (start `python app.py` in another terminal first).
Sends a batch of random requests so Prometheus has metrics to collect.
"""

import random
import sys

import requests

# 127.0.0.1 instead of "localhost" avoids a ~2 s IPv6 lookup delay per request on Windows
URL = "http://127.0.0.1:5000"

SAMPLES = [
    {"SepalLengthCm": 5.1, "SepalWidthCm": 3.5, "PetalLengthCm": 1.4, "PetalWidthCm": 0.2},
    {"SepalLengthCm": 6.0, "SepalWidthCm": 2.7, "PetalLengthCm": 4.1, "PetalWidthCm": 1.0},
    {"SepalLengthCm": 6.7, "SepalWidthCm": 3.0, "PetalLengthCm": 5.2, "PetalWidthCm": 2.3},
]

try:
    print("Health:", requests.get(f"{URL}/health", timeout=5).json())
except requests.ConnectionError:
    sys.exit("API is not running. Start it with:  python app.py")

print("Model info:", requests.get(f"{URL}/model-info").json())

single = requests.post(f"{URL}/predict", json=SAMPLES[0]).json()
print("\nSingle prediction:", single)

batch = requests.post(f"{URL}/predict", json={"instances": SAMPLES}).json()
print("\nBatch predictions:")
for sample, p in zip(SAMPLES, batch["predictions"]):
    print(f"  {list(sample.values())} -> {p['prediction']} ({p['confidence']})")

bad = requests.post(f"{URL}/predict", json={"SepalLengthCm": 5})
print("\nInvalid request ->", bad.status_code, bad.json())

# Generate traffic for Prometheus
for _ in range(50):
    sample = {k: round(v + random.uniform(-0.3, 0.3), 2) for k, v in random.choice(SAMPLES).items()}
    requests.post(f"{URL}/predict", json=sample)
print("\nSent 50 extra requests. View metrics at", f"{URL}/metrics", "or Prometheus at http://localhost:9090")
