"""
Experiment 15: Mini Project Phase-5 - End-to-end CI/CD pipeline for the mini project
=====================================================================================
Runs every phase of the project in order, exactly like the GitHub Actions workflow
(.github/workflows/mini-project-cicd.yml) does on each push and on the weekly retraining schedule:

    Phase-1 data  ->  Phase-2 features  ->  Phase-3 train & select  ->  quality gate
    ->  register model version  ->  prediction tests  ->  monitoring simulation  ->  ethics checks
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
P1 = ROOT / "Experiment-11_MiniProject_Phase1_Data_Collection_EDA"
P2 = ROOT / "Experiment-12_MiniProject_Phase2_Preprocessing_Features"
P3 = ROOT / "Experiment-13_MiniProject_Phase3_Model_Building"
P4 = ROOT / "Experiment-14_MiniProject_Phase4_Deployment"
PY = sys.executable
MIN_IMPROVEMENT_OVER_BASELINE = 0.10      # new model must be >= 10% better than "same hour yesterday"


def quality_gate():
    meta = json.loads((P3 / "outputs" / "models" / "model_metadata.json").read_text())
    rmse, baseline = meta["test_metrics"]["RMSE"], meta["baseline_test_rmse"]
    improvement = 1 - rmse / baseline
    print(f"Test RMSE {rmse:.4f} vs baseline {baseline:.4f} -> improvement {improvement:.1%} "
          f"(required {MIN_IMPROVEMENT_OVER_BASELINE:.0%})")
    if improvement < MIN_IMPROVEMENT_OVER_BASELINE:
        raise SystemExit("Quality gate FAILED")
    print("Quality gate PASSED")


STAGES = [
    ("Phase-1 data collection & EDA", P1, [PY, "phase1_data_collection_eda.py"]),
    ("Phase-2 preprocessing & features", P2, [PY, "phase2_preprocessing.py"]),
    ("Phase-3 model training & selection", P3, [PY, "phase3_model_building.py"]),
    ("Quality gate", HERE, quality_gate),
    ("Register model version", P4, [PY, "model_registry.py"]),
    ("Prediction tests", P4, [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_predict.py"]),
    ("Project tests", HERE, [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"]),
    ("Monitoring simulation", HERE, [PY, "monitor_and_retrain.py"]),
    ("Privacy / fairness / security checks", HERE, [PY, "privacy_ethics_check.py"]),
]

summary = []
for name, cwd, cmd in STAGES:
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
