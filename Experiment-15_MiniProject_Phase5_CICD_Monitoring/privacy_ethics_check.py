"""
Experiment 15: Mini Project Phase-5 - Ethical considerations: privacy, fairness, transparency, security
======================================================================================================
1. PRIVACY RISK   - show that fine-grained smart-meter data reveals when people are home / asleep
2. PRIVACY FIXES  - pseudonymisation (salted hash), aggregation (minute -> hour) and
                    differential-privacy style Laplace noise; measure the accuracy cost
3. FAIRNESS       - does the model work equally well on weekdays vs weekends, day vs night, all months?
4. TRANSPARENCY   - write a Model Card describing the model, data, metrics and limitations
5. SECURITY       - check the predictor rejects malicious / malformed input
"""

import hashlib
import json
import secrets
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent
RAW = ROOT / "Experiment-11_MiniProject_Phase1_Data_Collection_EDA" / "data" / "household_power_consumption.txt"
PHASE2 = ROOT / "Experiment-12_MiniProject_Phase2_Preprocessing_Features" / "outputs"
REGISTRY = ROOT / "Experiment-14_MiniProject_Phase4_Deployment" / "registry"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def title(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


# ---------------------------------------------------------------------------
title("1. PRIVACY RISK - what minute-level data reveals")
raw = pd.read_csv(RAW, sep=";", na_values=["?"], nrows=7 * 24 * 60, low_memory=False)
raw["Datetime"] = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S")
power = raw.set_index("Datetime")["Global_active_power"].dropna()
idle = power.rolling(60).median() < 0.6                 # an hour with only base load (fridge, standby)
by_hour = idle.groupby(idle.index.hour).mean()
asleep_or_away = by_hour[by_hour > 0.5].index.tolist()
print("Hours of the day when the household is likely asleep/away:", asleep_or_away)
kitchen = raw.set_index("Datetime")["Sub_metering_1"]
print("Cooking events detected (kitchen meter > 10 Wh/min):", int((kitchen > 10).sum()), "minutes in one week")
print("=> Raw readings reveal daily routine, occupancy and appliance use - personal data under GDPR / DPDP Act.")

# ---------------------------------------------------------------------------
title("2. PRIVACY PROTECTION")
salt = secrets.token_hex(16)                             # stored separately, never with the data
household_id = "MeterID-92-04-118-Paris"
pseudo = hashlib.sha256((salt + household_id).encode()).hexdigest()[:16]
print(f"Pseudonymisation : '{household_id}' -> '{pseudo}' (salted SHA-256, irreversible without the salt)")
print("Aggregation      : model uses HOURLY averages only - minute-level appliance signatures are removed")

config = json.loads((PHASE2 / "feature_config.json").read_text())
FEATURES, TARGET = config["features"], config["target"]
train = pd.read_csv(PHASE2 / "train.csv", index_col=0, parse_dates=True)
test = pd.read_csv(PHASE2 / "test.csv", index_col=0, parse_dates=True)
rng = np.random.default_rng(0)
print("\nDifferential-privacy style noise (Laplace) added to the training data:")
print(f"{'epsilon':>8s} {'noise scale':>12s} {'test MAE':>9s}")
sensitivity = 0.5
for eps in [None, 10, 2, 0.5]:
    noisy = train.copy()
    if eps:
        cols = [c for c in FEATURES if c.startswith(("lag", "rolling"))] + [TARGET]
        noisy[cols] += rng.laplace(0, sensitivity / eps, size=noisy[cols].shape)
    m = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    m.fit(noisy[FEATURES], noisy[TARGET])
    mae = mean_absolute_error(test[TARGET], m.predict(test[FEATURES]))
    print(f"{str(eps or 'none'):>8s} {(sensitivity / eps if eps else 0):12.3f} {mae:9.4f}")
print("=> Smaller epsilon = stronger privacy but lower accuracy (privacy-utility trade-off).")

# ---------------------------------------------------------------------------
title("3. FAIRNESS / RELIABILITY ACROSS GROUPS")
version = json.loads((REGISTRY / "latest.json").read_text())["version"]
model = joblib.load(REGISTRY / version / "model.joblib")
meta = json.loads((REGISTRY / version / "metadata.json").read_text())
test = test.assign(pred=model.predict(test[FEATURES]))
test["abs_err"] = (test[TARGET] - test["pred"]).abs()
groups = {
    "Day type": np.where(test["is_weekend"] == 1, "Weekend", "Weekday"),
    "Time of day": pd.cut(test["hour"], [-1, 5, 11, 17, 23], labels=["Night", "Morning", "Afternoon", "Evening"]),
}
fairness_rows = []
overall = test["abs_err"].mean()
for name, g in groups.items():
    stats = test.groupby(g, observed=True)["abs_err"].agg(["mean", "count"])
    for grp, row in stats.iterrows():
        ratio = row["mean"] / overall
        fairness_rows.append({"group": f"{name}: {grp}", "MAE": round(row["mean"], 4),
                              "hours": int(row["count"]), "ratio_to_overall": round(ratio, 2)})
fair = pd.DataFrame(fairness_rows)
print(fair.to_string(index=False))
flagged = fair[fair["ratio_to_overall"] > 1.25]
print("Groups with error > 25% above average:", ", ".join(flagged["group"]) if len(flagged) else "none")
print("=> A fair model has similar error for every group; any flagged group must be reported in the")
print("   model card and investigated (more data for that group, separate model, etc.).")

# ---------------------------------------------------------------------------
title("4. TRANSPARENCY - MODEL CARD")
card = f"""# Model Card - Household Electricity Consumption Predictor

| Item | Value |
|---|---|
| Model version | {meta['version']} |
| Algorithm | {meta['model_name']} {meta['params']} |
| Task | Predict next-hour household active power (kW) |
| Training data | UCI Household Electric Power Consumption (hourly aggregated), {meta['trained_on_rows']} hours |
| Test RMSE / MAE / R2 | {meta['test_metrics']['RMSE']} / {meta['test_metrics']['MAE']} / {meta['test_metrics']['R2']} |
| Baseline RMSE (same hour yesterday) | {meta['baseline_test_rmse']} |

## Intended use
Helping a household plan electricity use and helping utilities balance load.
**Not** for billing, credit decisions, insurance, or inferring the behaviour of residents.

## Inputs
Last 168 hours of consumption, calendar features and last-hour voltage / reactive power / sub-meter.

## Performance across groups
```
{fair.to_string(index=False)}
```

## Limitations
* Trained on one household; does not generalise to other homes without retraining.
* Sudden behaviour changes (new appliance, holidays, guests) cause drift - monitored via live MAE.
* No weather data is used.

## Ethical considerations
* Privacy: data is pseudonymised and aggregated to hourly values; raw minute data is not stored by the prediction service.
* Consent: households must opt in; data retention is limited and deletable on request.
* Security: inputs are validated, the container runs as non-root, model files are checksum-verified.
"""
(OUTPUT_DIR / "MODEL_CARD.md").write_text(card, encoding="utf-8")
print(card[:600], "...")
print("Model card saved to outputs/MODEL_CARD.md")

# ---------------------------------------------------------------------------
title("5. SECURITY - malicious / malformed input")
sys.path.insert(0, str(ROOT / "Experiment-14_MiniProject_Phase4_Deployment"))
from predict import forecast, predict_next_hour  # noqa: E402

attacks = {
    "Empty input": {},
    "Too short history": {"recent_consumption": [1] * 5},
    "Negative values": {"recent_consumption": [-5] * 168},
    "Text instead of numbers": {"recent_consumption": ["DROP TABLE"] * 168},
    "Huge forecast horizon (DoS)": {"recent_consumption": [1] * 168, "hours": 100000},
}
security_ok = True
for name, payload in attacks.items():
    try:
        if "hours" in payload:
            forecast(payload, hours=payload["hours"])
        else:
            predict_next_hour(payload)
        ok, outcome = False, "accepted"
    except (ValueError, TypeError) as err:
        ok, outcome = True, f"rejected ({type(err).__name__})"
    security_ok &= ok
    print(f"{'BLOCKED' if ok else 'NOT BLOCKED':12s} {name:30s} -> {outcome}")
print("All malicious inputs rejected." if security_ok else "Some inputs were not rejected - fix validation!")
print("\nExperiment 15 ethics & privacy check completed.")
