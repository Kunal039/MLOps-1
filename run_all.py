"""
Run every experiment in order and print a pass/fail summary.

    python run_all.py              -> all 20 experiments
    python run_all.py 7 9 13       -> only the listed experiment numbers

Docker steps are not run (they need Docker Desktop).
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

RUNS = {
    1: ("Experiment-01_Data_Preprocessing", [["data_preprocessing.py"]]),
    2: ("Experiment-02_Linear_Regression", [["linear_regression.py"]]),
    3: ("Experiment-03_Decision_Tree_Classification", [["decision_tree.py"]]),
    4: ("Experiment-04_Hyperparameter_Tuning", [["hyperparameter_tuning.py"]]),
    5: ("Experiment-05_DevOps_Git_Setup", [["setup_devops_repo.py"]]),
    6: ("Experiment-06_Docker_Containerization", [["train.py"], ["predict.py"]]),
    7: ("Experiment-07_Flask_REST_API_Prometheus", [["train_model.py"], ["predict.py"]]),
    8: ("Experiment-08_CICD_Pipeline", [["run_pipeline.py"]]),
    9: ("Experiment-09_DVC_Model_Versioning", [["setup_dvc.py"]]),
    10: ("Experiment-10_Monitoring_Logging", [["train_model.py"], ["simulate_traffic.py"], ["analyze_logs.py"]]),
    11: ("Experiment-11_MiniProject_Phase1_Data_Collection_EDA", [["phase1_data_collection_eda.py"]]),
    12: ("Experiment-12_MiniProject_Phase2_Preprocessing_Features", [["phase2_preprocessing.py"]]),
    13: ("Experiment-13_MiniProject_Phase3_Model_Building", [["phase3_model_building.py"]]),
    14: ("Experiment-14_MiniProject_Phase4_Deployment", [["model_registry.py"], ["predict.py"],
                                                     ["-m", "pytest", "-q", "test_predict.py"]]),
    15: ("Experiment-15_MiniProject_Phase5_CICD_Monitoring", [["monitor_and_retrain.py"], ["privacy_ethics_check.py"],
                                                            ["-m", "pytest", "-q", "tests"]]),
    16: ("Experiment-16_House_Price_Prediction_System", [["train.py"], ["predict.py"]]),
    17: ("Experiment-17_Titanic_Survival_Prediction", [["train.py"], ["predict.py"]]),
    18: ("Experiment-18_Student_Performance_Prediction", [["generate_dataset.py"], ["train.py"], ["predict.py"]]),
    19: ("Experiment-19_Fake_News_Detection_Flask_API", [["train.py"], ["-m", "pytest", "-q", "test_api.py"]]),
    20: ("Experiment-20_Employee_Salary_Prediction_Monitoring", [["generate_data.py"], ["generate_data.py", "--new-batch"],
                                                               ["train_mlflow.py"], ["predict.py"], ["monitor_drift.py"],
                                                               ["fairness_audit.py"]]),
}

selected = [int(a) for a in sys.argv[1:]] or list(RUNS)
summary = []
for number in selected:
    folder, commands = RUNS[number]
    start = time.perf_counter()
    ok = True
    for cmd in commands:
        result = subprocess.run([PY, *cmd, *(["-p", "no:cacheprovider"] if "pytest" in cmd else [])],
                                cwd=ROOT / folder, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        if result.returncode != 0:
            ok = False
            print(f"\n[FAILED] Experiment {number}: {' '.join(cmd)}\n" + (result.stdout + result.stderr)[-2000:])
            break
    duration = time.perf_counter() - start
    summary.append((number, folder, ok, duration))
    print(f"{'PASS' if ok else 'FAIL'}  {duration:6.1f}s  {folder}", flush=True)

failed = [s for s in summary if not s[2]]
print(f"\n{len(summary) - len(failed)}/{len(summary)} experiments passed.")
sys.exit(1 if failed else 0)
