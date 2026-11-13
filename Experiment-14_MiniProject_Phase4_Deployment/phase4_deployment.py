"""
Experiment 14: Mini Project Phase-4 - Model versioning and deployment (prediction service + Docker)
===================================================================================================
1. Model registry - copies the best model from Phase-3 into a versioned registry:

    registry/
      v1/  model.joblib  metadata.json   (metrics, features, SHA-256 checksum, registered_at)
      v2/  ...
      latest.json  -> {"version": "v2", ...}   <- predictions always use this version

   A new version is only created if the model file actually changed (checksum differs).
   (The data and model files can additionally be tracked with DVC - see Experiment 9.)

2. Prediction service - loads the latest registered version and predicts household consumption:
    predict_next_hour(payload)      -> predicted kW for the requested hour
    forecast(payload, hours=24)     -> recursive forecast for the next N hours

3. Docker - writes outputs/docker/ (Dockerfile, requirements.txt, this script, registry copy).

Commands:
    python phase4_deployment.py                     -> register + prediction demo + write the Docker build context
    python phase4_deployment.py register            -> register the Phase-3 model (then list versions)
    python phase4_deployment.py list                -> list registered versions
    python phase4_deployment.py rollback v1         -> point latest back to v1
    python phase4_deployment.py predict             -> next-hour prediction + 24-hour forecast demo
    python phase4_deployment.py docker              -> write outputs/docker/ for `docker build`
    python -m pytest phase4_deployment.py           -> tests (incl. the training-serving skew test)

payload = {"timestamp": "2009-06-20 18:00:00",
           "recent_consumption": [... at least 168 hourly kW values, oldest first ...],
           "voltage_last_hour": 239.4, "reactive_power_last_hour": 0.15, "sub_metering_3_last_hour": 7.2}
"""

import argparse
import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PHASE3_MODELS = BASE_DIR.parent / "Experiment-13_MiniProject_Phase3_Model_Building" / "outputs" / "models"
PHASE2 = BASE_DIR.parent / "Experiment-12_MiniProject_Phase2_Preprocessing_Features" / "outputs"
REGISTRY = BASE_DIR / "registry"
DOCKER_DIR = BASE_DIR / "outputs" / "docker"
HISTORY_HOURS = 168   # one week of hourly consumption is needed for the lag_168 feature

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("energy-model")


# ===========================================================================
# 1. Model registry (versioning)
# ===========================================================================
def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def versions():
    return sorted((p for p in REGISTRY.glob("v*") if p.is_dir()), key=lambda p: int(p.name[1:]))


def register():
    source = PHASE3_MODELS / "best_model.joblib"
    if not source.exists():
        raise SystemExit("No trained model - run Experiment-13 (Phase-3) first.")
    REGISTRY.mkdir(exist_ok=True)
    checksum = sha256(source)

    for v in versions():
        if json.loads((v / "metadata.json").read_text())["sha256"] == checksum:
            print(f"Model already registered as {v.name} - nothing to do.")
            set_latest(v.name)
            return

    new_version = f"v{len(versions()) + 1}"
    target = REGISTRY / new_version
    target.mkdir()
    shutil.copy2(source, target / "model.joblib")
    metadata = json.loads((PHASE3_MODELS / "model_metadata.json").read_text())
    metadata.update({"version": new_version, "sha256": checksum,
                     "registered_at": datetime.now().isoformat(timespec="seconds")})
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2))
    set_latest(new_version)
    print(f"Registered new model version {new_version} ({metadata['model_name']}, "
          f"test RMSE {metadata['test_metrics']['RMSE']})")


def set_latest(version):
    (REGISTRY / "latest.json").write_text(json.dumps({"version": version}, indent=2))
    print(f"latest -> {version}")


def list_versions():
    latest = json.loads((REGISTRY / "latest.json").read_text())["version"] if (REGISTRY / "latest.json").exists() else None
    print(f"{'Version':8s} {'Model':15s} {'Test RMSE':>10s}  Registered at         SHA-256")
    for v in versions():
        m = json.loads((v / "metadata.json").read_text())
        flag = " <- latest" if v.name == latest else ""
        print(f"{v.name:8s} {m['model_name']:15s} {m['test_metrics']['RMSE']:10.4f}  "
              f"{m['registered_at']}  {m['sha256'][:12]}{flag}")


# ===========================================================================
# 2. Features - must create exactly the same features as Phase-2 (phase2_preprocessing.py),
#    otherwise the deployed model would receive different inputs than it was trained on
#    (training-serving skew).
# ===========================================================================
def build_features(payload, feature_order):
    """
    payload = {
        "timestamp": "2009-06-20 18:00:00",            # hour to predict
        "recent_consumption": [... 168 values ...],     # hourly kW, oldest first, last = previous hour
        "voltage_last_hour": 239.4,
        "reactive_power_last_hour": 0.15,
        "sub_metering_3_last_hour": 7.2
    }
    """
    history = payload.get("recent_consumption")
    if not isinstance(history, list) or len(history) < HISTORY_HOURS:
        raise ValueError(f"'recent_consumption' must contain at least {HISTORY_HOURS} hourly values")
    history = np.asarray(history[-HISTORY_HOURS:], dtype=float)
    if (history < 0).any():
        raise ValueError("Consumption values cannot be negative")

    ts = pd.Timestamp(payload.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:00:00"))
    last_24 = pd.Series(history[-24:])

    features = {
        "hour": ts.hour,
        "day_of_week": ts.dayofweek,
        "month": ts.month,
        "is_weekend": int(ts.dayofweek >= 5),
        "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
        "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
        "lag_1": history[-1],
        "lag_2": history[-2],
        "lag_3": history[-3],
        "lag_24": history[-24],
        "lag_168": history[-168],
        "rolling_mean_3": history[-3:].mean(),
        "rolling_mean_24": last_24.mean(),
        "rolling_std_24": last_24.std(),          # pandas std (ddof=1) - same as in Phase-2
        "Voltage_lag_1": float(payload.get("voltage_last_hour", 240.0)),
        "Global_reactive_power_lag_1": float(payload.get("reactive_power_last_hour", 0.12)),
        "Sub_metering_3_lag_1": float(payload.get("sub_metering_3_last_hour", 6.0)),
    }
    return pd.DataFrame([features])[feature_order]


# ===========================================================================
# 3. Prediction service (latest registered model)
# ===========================================================================
_loaded = {}


def load_latest_model():
    """Load (once) and return the model + metadata of the version that latest.json points to."""
    if not _loaded:
        version = json.loads((REGISTRY / "latest.json").read_text())["version"]
        _loaded["model"] = joblib.load(REGISTRY / version / "model.joblib")
        _loaded["metadata"] = json.loads((REGISTRY / version / "metadata.json").read_text())
        logger.info("Loaded model %s (%s)", version, _loaded["metadata"]["model_name"])
    return _loaded["model"], _loaded["metadata"]


def predict_next_hour(payload):
    """Predicted consumption (kW) for payload['timestamp']. Raises ValueError for invalid input."""
    model, metadata = load_latest_model()
    X = build_features(payload, metadata["features"])
    return max(float(model.predict(X)[0]), 0.0)


def forecast(payload, hours=24):
    """Predict the next N hours, feeding each prediction back as the newest history value."""
    if not 1 <= hours <= 168:
        raise ValueError("'hours' must be between 1 and 168")
    history = list(payload.get("recent_consumption") or [])
    if len(history) < HISTORY_HOURS:
        raise ValueError(f"'recent_consumption' must contain at least {HISTORY_HOURS} values")
    start = pd.Timestamp(payload.get("timestamp", pd.Timestamp.now().floor("h")))
    results = []
    for h in range(hours):
        ts = start + pd.Timedelta(hours=h)
        value = predict_next_hour({**payload, "timestamp": str(ts), "recent_consumption": history})
        results.append({"timestamp": str(ts), "predicted_kw": round(value, 4)})
        history = history[1:] + [value]
    return results


def sample_payload(row=200):
    """A real week of history from the Phase-2 test set, for predicting test-set hour `row`."""
    test = pd.read_csv(PHASE2 / "test.csv", index_col=0, parse_dates=True)
    target_row = test.iloc[row]
    payload = {
        "timestamp": str(test.index[row]),
        "recent_consumption": test["Global_active_power"].iloc[row - 168: row].tolist(),
        "voltage_last_hour": float(target_row["Voltage_lag_1"]),
        "reactive_power_last_hour": float(target_row["Global_reactive_power_lag_1"]),
        "sub_metering_3_last_hour": float(target_row["Sub_metering_3_lag_1"]),
    }
    return payload, test


def demo():
    _, metadata = load_latest_model()
    print(f"Model {metadata['version']}: {metadata['model_name']}, test RMSE {metadata['test_metrics']['RMSE']} kW\n")
    if (PHASE2 / "test.csv").exists():
        payload, test = sample_payload()
        actual = f", actual {test['Global_active_power'].iloc[200]:.3f} kW"
    else:
        # inside the Docker image there is no Phase-2 data: use a typical week (low at night, evening peak)
        day = [0.45] * 6 + [0.9, 1.2, 1.0] + [0.5] * 9 + [1.2, 1.8, 2.1, 1.9, 1.4, 0.8]
        payload = {"timestamp": "2009-06-20 18:00:00", "recent_consumption": day * 7}
        actual = ""
    predicted = predict_next_hour(payload)
    print(f"Next hour {payload['timestamp']}: predicted {predicted:.3f} kW{actual}")

    day = forecast(payload, hours=24)
    print(f"\n24-hour forecast from {day[0]['timestamp']}: total {sum(r['predicted_kw'] for r in day):.2f} kWh")
    for r in day[::4]:
        print(f"  {r['timestamp']}  {r['predicted_kw']:.3f} kW")


# ===========================================================================
# 4. Docker build context
# ===========================================================================
DOCKERFILE = """\
# Mini project - Phase 4: containerised prediction service
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY phase4_deployment.py ./
COPY registry/ registry/
RUN useradd -m appuser
USER appuser
CMD ["python", "phase4_deployment.py", "predict"]
"""


def write_docker_context():
    """Write outputs/docker/ = a complete build context: Dockerfile, requirements, script, registry."""
    import sklearn

    if DOCKER_DIR.exists():
        shutil.rmtree(DOCKER_DIR)
    DOCKER_DIR.mkdir(parents=True)
    (DOCKER_DIR / "Dockerfile").write_text(DOCKERFILE)
    # pin the scikit-learn version the model was trained with, so the pickled model loads
    (DOCKER_DIR / "requirements.txt").write_text(
        f"pandas\nnumpy\njoblib\nscikit-learn=={sklearn.__version__}\n")
    shutil.copy2(Path(__file__), DOCKER_DIR / "phase4_deployment.py")
    shutil.copytree(REGISTRY, DOCKER_DIR / "registry")
    print(f"Docker build context written to {DOCKER_DIR}")
    print("Build and run:")
    print(f'  docker build -t household-energy-model:1.0 "{DOCKER_DIR}"')
    print("  docker run --rm household-energy-model:1.0")


# ===========================================================================
# 5. Tests  (python -m pytest phase4_deployment.py -v)
#    The most important test checks there is no training-serving skew: features built at
#    prediction time from raw history must equal the features Phase-2 produced for the same hour.
# ===========================================================================
def test_features_match_training_pipeline():
    payload, test = sample_payload()
    features = load_latest_model()[1]["features"]
    serving_features = build_features(payload, features).iloc[0].to_numpy(dtype=float)
    training_features = test.iloc[200][features].to_numpy(dtype=float)
    assert np.allclose(serving_features, training_features, atol=1e-6)


def test_registry_metadata():
    metadata = load_latest_model()[1]
    assert metadata["version"].startswith("v") and len(metadata["features"]) == 17


def test_prediction_is_close_to_actual():
    payload, test = sample_payload()
    assert abs(predict_next_hour(payload) - test["Global_active_power"].iloc[200]) < 0.5


def test_forecast_24_hours():
    assert len(forecast(sample_payload()[0], hours=24)) == 24


def test_short_history_rejected():
    import pytest
    with pytest.raises(ValueError):
        predict_next_hour({"recent_consumption": [1.0] * 10})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 14 - model registry, prediction service, Docker")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("register", help="register the Phase-3 model and list versions")
    sub.add_parser("list", help="list registered versions")
    rb = sub.add_parser("rollback", help="point latest to an older version")
    rb.add_argument("version")
    sub.add_parser("predict", help="next-hour prediction + 24-hour forecast demo")
    sub.add_parser("docker", help="write outputs/docker/ build context")
    args = parser.parse_args()

    if args.command == "register":
        register()
        list_versions()
    elif args.command == "list":
        list_versions()
    elif args.command == "rollback":
        if not (REGISTRY / args.version).exists():
            raise SystemExit(f"Version {args.version} does not exist")
        set_latest(args.version)
        list_versions()
    elif args.command == "predict":
        demo()
    elif args.command == "docker":
        write_docker_context()
    else:
        register()
        list_versions()
        print()
        demo()
        print()
        write_docker_context()
