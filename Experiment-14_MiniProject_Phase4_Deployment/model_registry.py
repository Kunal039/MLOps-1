"""
Experiment 14: Mini Project Phase-4 - Model versioning (simple model registry)
==============================================================================
Copies the best model from Phase-3 into a versioned registry:

    registry/
      v1/  model.joblib  metadata.json   (metrics, features, SHA-256 checksum, registered_at)
      v2/  ...
      latest.json  -> {"version": "v2", ...}   <- the API always serves this version

A new version is only created if the model file actually changed (checksum differs).
`python model_registry.py --list` shows all versions; `--rollback v1` points latest back to v1.
(The data and model files can additionally be tracked with DVC - see README.)
"""

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PHASE3_MODELS = BASE_DIR.parent / "Experiment-13_MiniProject_Phase3_Model_Building" / "outputs" / "models"
REGISTRY = BASE_DIR / "registry"


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--rollback", metavar="VERSION")
    args = parser.parse_args()
    if args.rollback:
        if not (REGISTRY / args.rollback).exists():
            raise SystemExit(f"Version {args.rollback} does not exist")
        set_latest(args.rollback)
    elif not args.list:
        register()
    list_versions()
