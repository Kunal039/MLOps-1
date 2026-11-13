"""
Experiment 15: Mini Project Phase-5 - Production simulation, monitoring and retraining trigger
==============================================================================================
Replays the test period hour by hour through the monitored model (production_monitor.py):
    1. ask the model for a prediction for the next hour
    2. when the hour is over, record the actual meter reading
Halfway through, concept drift is simulated (e.g. the family buys an electric car that
charges every evening: +1.2 kW from 19:00 to 23:00). The live 24-hour MAE rises, the
service logs a warning and this script writes a retraining trigger - in production this
trigger would start the CI/CD retraining job (see .github/workflows, `workflow_dispatch`).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import production_monitor as monitor

BASE_DIR = Path(__file__).resolve().parent
PHASE2 = BASE_DIR.parent / "Experiment-12_MiniProject_Phase2_Preprocessing_Features" / "outputs"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

data = pd.concat([pd.read_csv(PHASE2 / f, index_col=0, parse_dates=True) for f in ["validation.csv", "test.csv"]])
series = data["Global_active_power"].copy()

DRIFT_START = len(series) - 300          # last ~12 days get the new appliance
evening = (series.index.hour >= 19) & (series.index.hour <= 23)
drift_mask = (pd.Series(range(len(series)), index=series.index) >= DRIFT_START) & evening
series[drift_mask] += 1.2

records, trigger = [], None
for i in range(168, len(series)):
    ts = series.index[i]
    payload = {
        "timestamp": str(ts),
        "recent_consumption": series.iloc[i - 168:i].tolist(),
        "voltage_last_hour": float(data["Voltage_lag_1"].iloc[i]),
        "reactive_power_last_hour": float(data["Global_reactive_power_lag_1"].iloc[i]),
        "sub_metering_3_last_hour": float(data["Sub_metering_3_lag_1"].iloc[i]),
    }
    pred = monitor.predict(payload)
    fb = monitor.record_actual(str(ts), float(series.iloc[i]))
    records.append({"timestamp": ts, "actual": series.iloc[i], "predicted": pred, "live_mae": fb["live_mae_24h"]})
    if fb["retrain_required"] and trigger is None:
        trigger = {"triggered_at_hour": str(ts), "live_mae_24h": fb["live_mae_24h"],
                   "reason": "Rolling 24h MAE exceeded 1.5x the offline test MAE",
                   "action": "Run the retraining pipeline (Phase 2 -> 3 -> register new version -> CI/CD deploy)"}

log = pd.DataFrame(records).set_index("timestamp")
log.to_csv(OUTPUT_DIR / "production_log.csv")
before = log.iloc[: DRIFT_START - 168]
after = log.iloc[DRIFT_START - 168:]
print(f"Hours simulated: {len(log)}")
print(f"MAE before drift: {abs(before.actual - before.predicted).mean():.4f} kW")
print(f"MAE after drift : {abs(after.actual - after.predicted).mean():.4f} kW")

if trigger:
    (OUTPUT_DIR / "retrain_trigger.json").write_text(json.dumps(trigger, indent=2))
    print("\nRETRAINING TRIGGERED:", json.dumps(trigger, indent=2))
else:
    print("\nModel healthy - no retraining needed.")

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(log.index, log["actual"], label="Actual", linewidth=0.8)
axes[0].plot(log.index, log["predicted"], label="Predicted", linewidth=0.8)
axes[0].axvline(series.index[DRIFT_START], color="red", linestyle="--", label="Drift starts (EV charging)")
axes[0].set_ylabel("kW")
axes[0].set_title("Deployed model - actual vs predicted")
axes[0].legend()
axes[1].plot(log.index, log["live_mae"], color="tab:purple", label="Rolling 24h MAE")
axes[1].axhline(monitor.MAE_ALERT_THRESHOLD, color="red", linestyle="--", label="Alert threshold")
axes[1].set_ylabel("MAE (kW)")
axes[1].set_title("Live model performance (energy_live_mae_kw metric)")
axes[1].legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "production_monitoring.png")
print("Plot saved to outputs/production_monitoring.png")
print("Warnings in logs/service.log:", sum("WARNING" in l for l in (BASE_DIR / "logs" / "service.log").open()))
