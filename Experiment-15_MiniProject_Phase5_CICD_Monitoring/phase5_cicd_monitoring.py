"""
Experiment 15: Mini Project Phase-5 - CI/CD, monitoring and ethical considerations
==================================================================================
Project : Household Electricity Consumption Prediction System

1. Production monitoring - wraps the Phase-4 predictor with Prometheus metrics, a prediction log
   and live error tracking once the actual meter reading for an hour is known:
       predict(payload)                 -> predicted kW (logged + counted)
       record_actual(timestamp, kw)     -> absolute error, rolling 24h MAE, retrain flag
       start_metrics_server(port)       -> expose metrics for Prometheus at http://localhost:<port>/metrics
   Metrics: energy_predictions_total, energy_predicted_kw, energy_prediction_latency_seconds,
            energy_abs_error_kw, energy_live_mae_kw (rolling 24h), energy_model_info
   Logs   : logs/service.log, logs/predictions.jsonl
2. Monitor and retrain - replays the test period hour by hour; halfway through, concept drift is
   simulated (the family buys an electric car that charges every evening: +1.2 kW from 19:00 to 23:00).
   The live 24-hour MAE rises, a warning is logged and a retraining trigger is written - in production
   this trigger would start the CI/CD retraining job (`workflow_dispatch` in the GitHub Actions workflow).
3. Ethics - privacy risk, privacy protection, fairness across groups, Model Card, security tests.
4. CI/CD - `pipeline` runs every phase of the project in order, exactly like the GitHub Actions
   workflow (written to outputs/ci/mini-project-cicd.yml) does on each push and on the weekly schedule:
       Phase-1 data -> Phase-2 features -> Phase-3 train & select -> quality gate
       -> register model version -> prediction tests -> project tests -> monitoring -> ethics checks

Commands:
    python phase5_cicd_monitoring.py              -> monitoring simulation + ethics checks + write the CI workflow
    python phase5_cicd_monitoring.py monitor      -> production monitoring simulation and retraining trigger
    python phase5_cicd_monitoring.py ethics       -> privacy / fairness / transparency / security checks
    python phase5_cicd_monitoring.py ci           -> write outputs/ci/mini-project-cicd.yml
    python phase5_cicd_monitoring.py pipeline     -> full end-to-end CI/CD pipeline (Phase 1 -> 5)
    python -m pytest phase5_cicd_monitoring.py    -> data quality, model quality and monitoring tests
"""

import argparse
import hashlib
import json
import logging
import secrets
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent
P1 = ROOT / "Experiment-11_MiniProject_Phase1_Data_Collection_EDA"
P2 = ROOT / "Experiment-12_MiniProject_Phase2_Preprocessing_Features"
P3 = ROOT / "Experiment-13_MiniProject_Phase3_Model_Building"
P4 = ROOT / "Experiment-14_MiniProject_Phase4_Deployment"
RAW = P1 / "data" / "household_power_consumption.txt"
PHASE2 = P2 / "outputs"
PHASE3 = P3 / "outputs" / "models"
REGISTRY = P4 / "registry"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = BASE_DIR / "logs"
PY = sys.executable

sys.path.insert(0, str(P4))
import phase4_deployment as phase4  # noqa: E402  (Phase-4 predictor and model registry)

logging.getLogger("energy-model").setLevel(logging.WARNING)   # keep the console readable


def title(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


# ===========================================================================
# 1. Production monitoring
# ===========================================================================
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("energy-monitor")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(LOG_DIR / "service.log")
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(handler)
prediction_log = (LOG_DIR / "predictions.jsonl").open("a", buffering=1)

PREDICTIONS = Counter("energy_predictions_total", "Predictions served")
PREDICTED_KW = Gauge("energy_predicted_kw", "Last predicted consumption (kW)")
LATENCY = Histogram("energy_prediction_latency_seconds", "Prediction latency")
ABS_ERROR = Histogram("energy_abs_error_kw", "Absolute error once actual is known",
                      buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 1, 2])
LIVE_MAE = Gauge("energy_live_mae_kw", "Rolling 24-hour MAE of the deployed model")
MODEL_INFO = Gauge("energy_model_info", "Deployed model version", ["version", "algorithm"])

pending = {}                    # timestamp -> predicted kW, waiting for the actual value
errors = deque(maxlen=24)       # last 24 hourly absolute errors


def model_metadata():
    metadata = phase4.load_latest_model()[1]
    MODEL_INFO.labels(metadata["version"], metadata["model_name"]).set(1)
    return metadata


def mae_alert_threshold():
    return 1.5 * model_metadata()["test_metrics"]["MAE"]   # 50% worse than offline test MAE


def start_metrics_server(port=8000):
    start_http_server(port)


def predict(payload):
    start = time.perf_counter()
    predicted = round(phase4.predict_next_hour(payload), 4)
    LATENCY.observe(time.perf_counter() - start)
    PREDICTIONS.inc()
    PREDICTED_KW.set(predicted)
    pending[payload["timestamp"]] = predicted
    prediction_log.write(json.dumps({"logged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                     "timestamp": payload["timestamp"], "predicted_kw": predicted,
                                     "model_version": model_metadata()["version"]}) + "\n")
    return predicted


def record_actual(timestamp, actual_kw):
    predicted = pending.pop(timestamp, None)
    if predicted is None:
        raise KeyError(f"No prediction waiting for {timestamp}")
    error = abs(predicted - float(actual_kw))
    errors.append(error)
    ABS_ERROR.observe(error)
    mae = sum(errors) / len(errors)
    LIVE_MAE.set(round(mae, 4))
    prediction_log.write(json.dumps({"timestamp": timestamp, "predicted_kw": predicted,
                                     "actual_kw": actual_kw, "abs_error": round(error, 4)}) + "\n")
    threshold = mae_alert_threshold()
    degraded = len(errors) == errors.maxlen and mae > threshold
    if degraded:
        logger.warning("Model degraded: 24h MAE %.3f kW > threshold %.3f kW - retraining required", mae, threshold)
    return {"abs_error": round(error, 4), "live_mae_24h": round(mae, 4), "retrain_required": degraded}


# ===========================================================================
# 2. Production simulation, monitoring and retraining trigger
# ===========================================================================
def monitor_and_retrain():
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = pd.concat([pd.read_csv(PHASE2 / f, index_col=0, parse_dates=True)
                      for f in ["validation.csv", "test.csv"]])
    series = data["Global_active_power"].copy()

    drift_start = len(series) - 300          # last ~12 days get the new appliance
    evening = (series.index.hour >= 19) & (series.index.hour <= 23)
    drift_mask = (pd.Series(range(len(series)), index=series.index) >= drift_start) & evening
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
        pred = predict(payload)
        fb = record_actual(str(ts), float(series.iloc[i]))
        records.append({"timestamp": ts, "actual": series.iloc[i], "predicted": pred, "live_mae": fb["live_mae_24h"]})
        if fb["retrain_required"] and trigger is None:
            trigger = {"triggered_at_hour": str(ts), "live_mae_24h": fb["live_mae_24h"],
                       "reason": "Rolling 24h MAE exceeded 1.5x the offline test MAE",
                       "action": "Run the retraining pipeline (Phase 2 -> 3 -> register new version -> CI/CD deploy)"}

    log = pd.DataFrame(records).set_index("timestamp")
    log.to_csv(OUTPUT_DIR / "production_log.csv")
    before = log.iloc[: drift_start - 168]
    after = log.iloc[drift_start - 168:]
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
    axes[0].axvline(series.index[drift_start], color="red", linestyle="--", label="Drift starts (EV charging)")
    axes[0].set_ylabel("kW")
    axes[0].set_title("Deployed model - actual vs predicted")
    axes[0].legend()
    axes[1].plot(log.index, log["live_mae"], color="tab:purple", label="Rolling 24h MAE")
    axes[1].axhline(mae_alert_threshold(), color="red", linestyle="--", label="Alert threshold")
    axes[1].set_ylabel("MAE (kW)")
    axes[1].set_title("Live model performance (energy_live_mae_kw metric)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "production_monitoring.png")
    plt.close()
    print("Plot saved to outputs/production_monitoring.png")
    print("Warnings in logs/service.log:", sum("WARNING" in line for line in (LOG_DIR / "service.log").open()))


# ===========================================================================
# 3. Ethical considerations: privacy, fairness, transparency, security
#    1. PRIVACY RISK   - show that fine-grained smart-meter data reveals when people are home / asleep
#    2. PRIVACY FIXES  - pseudonymisation (salted hash), aggregation (minute -> hour) and
#                        differential-privacy style Laplace noise; measure the accuracy cost
#    3. FAIRNESS       - does the model work equally well on weekdays vs weekends, day vs night?
#    4. TRANSPARENCY   - write a Model Card describing the model, data, metrics and limitations
#    5. SECURITY       - check the predictor rejects malicious / malformed input
# ===========================================================================
def privacy_ethics_check():
    OUTPUT_DIR.mkdir(exist_ok=True)

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

    title("2. PRIVACY PROTECTION")
    salt = secrets.token_hex(16)                             # stored separately, never with the data
    household_id = "MeterID-92-04-118-Paris"
    pseudo = hashlib.sha256((salt + household_id).encode()).hexdigest()[:16]
    print(f"Pseudonymisation : '{household_id}' -> '{pseudo}' (salted SHA-256, irreversible without the salt)")
    print("Aggregation      : model uses HOURLY averages only - minute-level appliance signatures are removed")

    config = json.loads((PHASE2 / "feature_config.json").read_text())
    features, target = config["features"], config["target"]
    train = pd.read_csv(PHASE2 / "train.csv", index_col=0, parse_dates=True)
    test = pd.read_csv(PHASE2 / "test.csv", index_col=0, parse_dates=True)
    rng = np.random.default_rng(0)
    print("\nDifferential-privacy style noise (Laplace) added to the training data:")
    print(f"{'epsilon':>8s} {'noise scale':>12s} {'test MAE':>9s}")
    sensitivity = 0.5
    for eps in [None, 10, 2, 0.5]:
        noisy = train.copy()
        if eps:
            cols = [c for c in features if c.startswith(("lag", "rolling"))] + [target]
            noisy[cols] += rng.laplace(0, sensitivity / eps, size=noisy[cols].shape)
        m = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        m.fit(noisy[features], noisy[target])
        mae = mean_absolute_error(test[target], m.predict(test[features]))
        print(f"{str(eps or 'none'):>8s} {(sensitivity / eps if eps else 0):12.3f} {mae:9.4f}")
    print("=> Smaller epsilon = stronger privacy but lower accuracy (privacy-utility trade-off).")

    title("3. FAIRNESS / RELIABILITY ACROSS GROUPS")
    version = json.loads((REGISTRY / "latest.json").read_text())["version"]
    model = joblib.load(REGISTRY / version / "model.joblib")
    meta = json.loads((REGISTRY / version / "metadata.json").read_text())
    test = test.assign(pred=model.predict(test[features]))
    test["abs_err"] = (test[target] - test["pred"]).abs()
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

    title("5. SECURITY - malicious / malformed input")
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
                phase4.forecast(payload, hours=payload["hours"])
            else:
                phase4.predict_next_hour(payload)
            ok, outcome = False, "accepted"
        except (ValueError, TypeError) as err:
            ok, outcome = True, f"rejected ({type(err).__name__})"
        security_ok &= ok
        print(f"{'BLOCKED' if ok else 'NOT BLOCKED':12s} {name:30s} -> {outcome}")
    print("All malicious inputs rejected." if security_ok else "Some inputs were not rejected - fix validation!")
    print("\nExperiment 15 ethics & privacy check completed.")


# ===========================================================================
# 4. CI/CD - GitHub Actions workflow and the end-to-end pipeline
# ===========================================================================
WORKFLOW = """\
# Mini Project - Household Electricity Consumption Prediction - CI/CD
# Copy to <repo-root>/.github/workflows/ (GitHub only reads workflows from the repository root).
# Assumes the repo root is the MLOps-Lab-Experiments folder.
name: Energy Model CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: "0 2 * * 1"          # weekly retraining every Monday 02:00 UTC
  workflow_dispatch:              # manual / monitoring-triggered retraining

jobs:
  train-test-validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run full ML pipeline (data -> features -> train -> quality gate -> tests -> monitoring -> ethics)
        run: python Experiment-15_MiniProject_Phase5_CICD_Monitoring/phase5_cicd_monitoring.py pipeline
      - name: Upload model registry and reports
        uses: actions/upload-artifact@v4
        with:
          name: energy-model-${{ github.sha }}
          path: |
            Experiment-14_MiniProject_Phase4_Deployment/registry/
            Experiment-13_MiniProject_Phase3_Model_Building/outputs/
            Experiment-15_MiniProject_Phase5_CICD_Monitoring/outputs/

  build-and-deploy:
    needs: train-test-validate
    if: github.ref == 'refs/heads/main' && github.event_name != 'pull_request'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install pandas numpy scikit-learn joblib
      - uses: actions/download-artifact@v4
        with:
          name: energy-model-${{ github.sha }}
          path: artifacts
      - name: Put the validated model into place and write the Docker build context
        run: |
          cp -r artifacts/Experiment-14_MiniProject_Phase4_Deployment/registry Experiment-14_MiniProject_Phase4_Deployment/
          python Experiment-14_MiniProject_Phase4_Deployment/phase4_deployment.py docker
      - name: Build image (Phase-4 prediction service)
        run: docker build -t ghcr.io/${{ github.repository_owner }}/energy-model:${{ github.sha }} Experiment-14_MiniProject_Phase4_Deployment/outputs/docker
      - name: Smoke test the container (runs a prediction)
        run: docker run --rm ghcr.io/${{ github.repository_owner }}/energy-model:${{ github.sha }}
      - name: Push image
        run: |
          echo "${{ secrets.GITHUB_TOKEN }}" | docker login ghcr.io -u ${{ github.actor }} --password-stdin
          docker push ghcr.io/${{ github.repository_owner }}/energy-model:${{ github.sha }}
      - name: Deploy
        run: echo "Deploy the new image (AWS ECS / GCP Cloud Run / Azure Container Apps / on-prem)"
"""

MIN_IMPROVEMENT_OVER_BASELINE = 0.10      # new model must be >= 10% better than "same hour yesterday"


def write_workflow():
    ci_dir = OUTPUT_DIR / "ci"
    ci_dir.mkdir(parents=True, exist_ok=True)
    (ci_dir / "mini-project-cicd.yml").write_text(WORKFLOW)
    print(f"GitHub Actions workflow written to {ci_dir / 'mini-project-cicd.yml'}")
    print("Copy it to <repo-root>/.github/workflows/ to run it on GitHub.")


def quality_gate():
    meta = json.loads((PHASE3 / "model_metadata.json").read_text())
    rmse, baseline = meta["test_metrics"]["RMSE"], meta["baseline_test_rmse"]
    improvement = 1 - rmse / baseline
    print(f"Test RMSE {rmse:.4f} vs baseline {baseline:.4f} -> improvement {improvement:.1%} "
          f"(required {MIN_IMPROVEMENT_OVER_BASELINE:.0%})")
    if improvement < MIN_IMPROVEMENT_OVER_BASELINE:
        raise SystemExit("Quality gate FAILED")
    print("Quality gate PASSED")


def run_pipeline():
    this = Path(__file__).name
    pytest = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    stages = [
        ("Phase-1 data collection & EDA", P1, [PY, "phase1_data_collection_eda.py"]),
        ("Phase-2 preprocessing & features", P2, [PY, "phase2_preprocessing.py"]),
        ("Phase-3 model training & selection", P3, [PY, "phase3_model_building.py"]),
        ("Quality gate", BASE_DIR, quality_gate),
        ("Register model version", P4, [PY, "phase4_deployment.py", "register"]),
        ("Prediction tests", P4, pytest + ["phase4_deployment.py"]),
        ("Project tests", BASE_DIR, pytest + [this]),
        ("Monitoring simulation", BASE_DIR, [PY, this, "monitor"]),
        ("Privacy / fairness / security checks", BASE_DIR, [PY, this, "ethics"]),
    ]
    summary = []
    for name, cwd, cmd in stages:
        print("\n" + "=" * 70 + f"\n[STAGE] {name}\n" + "=" * 70)
        start = time.perf_counter()
        if callable(cmd):
            try:
                cmd()
                ok = True
            except SystemExit:
                ok = False
        else:
            result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
            print("\n".join((result.stdout + result.stderr).strip().splitlines()[-6:]))
            ok = result.returncode == 0
        summary.append((name, ok, time.perf_counter() - start))
        if not ok:
            break

    print("\n" + "=" * 70 + "\nPIPELINE SUMMARY\n" + "=" * 70)
    for name, ok, t in summary:
        print(f"{'PASSED' if ok else 'FAILED':7s} {t:6.1f}s  {name}")
    if not all(ok for _, ok, _ in summary):
        sys.exit(1)
    print("\nMini project pipeline completed - model is ready to deploy.")


# ===========================================================================
# 5. Tests (python -m pytest phase5_cicd_monitoring.py): data quality, model quality, monitoring
# ===========================================================================
def test_processed_data_has_no_missing_values():
    for split in ["train", "validation", "test"]:
        df = pd.read_csv(PHASE2 / f"{split}.csv", index_col=0)
        assert not df.isna().any().any(), split


def test_time_split_has_no_overlap():
    train = pd.read_csv(PHASE2 / "train.csv", index_col=0, parse_dates=True)
    test = pd.read_csv(PHASE2 / "test.csv", index_col=0, parse_dates=True)
    assert train.index.max() < test.index.min()


def test_consumption_is_physically_plausible():
    df = pd.read_csv(PHASE2 / "train.csv", index_col=0)
    assert df["Global_active_power"].between(0, 11).all()


def test_model_beats_baseline():
    meta = json.loads((PHASE3 / "model_metadata.json").read_text())
    assert meta["test_metrics"]["RMSE"] < meta["baseline_test_rmse"]
    assert meta["test_metrics"]["R2"] > 0.8


def test_metrics_are_registered():
    from prometheus_client import generate_latest
    model_metadata()
    text = generate_latest().decode()
    assert "energy_live_mae_kw" in text and "energy_model_info" in text


def test_prediction_and_feedback_loop():
    test = pd.read_csv(PHASE2 / "test.csv", index_col=0, parse_dates=True)
    i = 300
    payload = {"timestamp": str(test.index[i]),
               "recent_consumption": test["Global_active_power"].iloc[i - 168:i].tolist()}
    assert predict(payload) >= 0
    fb = record_actual(str(test.index[i]), float(test["Global_active_power"].iloc[i]))
    assert fb["abs_error"] >= 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 15 - CI/CD, monitoring and ethics")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("monitor", help="production monitoring simulation + retraining trigger")
    sub.add_parser("ethics", help="privacy / fairness / transparency / security checks")
    sub.add_parser("ci", help="write the GitHub Actions workflow to outputs/ci/")
    sub.add_parser("pipeline", help="run the whole mini project pipeline (Phase 1 -> 5)")
    args = parser.parse_args()

    if args.command == "monitor":
        monitor_and_retrain()
    elif args.command == "ethics":
        privacy_ethics_check()
    elif args.command == "ci":
        write_workflow()
    elif args.command == "pipeline":
        run_pipeline()
    else:
        title("PRODUCTION MONITORING AND RETRAINING TRIGGER")
        monitor_and_retrain()
        privacy_ethics_check()
        title("CI/CD WORKFLOW")
        write_workflow()
