"""
Experiment 9: Implement model versioning using DVC for efficient model management
---------------------------------------------------------------------------------
Aim:
    * Install and configure DVC in a project.
    * Version control data, models and experiments for reproducibility.

This one file contains both the ML pipeline stages that DVC runs and the walkthrough:

    python dvc_model_versioning.py                     -> full DVC walkthrough (same as `setup`)
    python dvc_model_versioning.py get-data --version 1|2
    python dvc_model_versioning.py prepare             -> DVC stage 1: clean and split the raw data
    python dvc_model_versioning.py train               -> DVC stage 2: train a Random Forest (params.yaml)
    python dvc_model_versioning.py evaluate            -> DVC stage 3: metrics.json + confusion-matrix data

`setup` writes dvc.yaml, params.yaml and .gitattributes (defined below) plus a copy of this script
into outputs/dvc-workspace (a fresh Git + DVC repo) and runs every DVC command step-by-step, printing each one like a terminal:

    1. git init + dvc init + configure a (local) DVC remote storage
    2. Version data v1        -> dvc add, git commit, dvc push, git tag
    3. Run the ML pipeline     -> dvc repro (prepare -> train -> evaluate), dvc metrics show
    4. Track experiments       -> dvc exp run --set-param ..., dvc exp show
    5. Version data v2 + model -> dvc add, dvc repro, dvc metrics diff
    6. Reproducibility         -> git checkout v1 + dvc checkout restores the old data & model

Prerequisite:  pip install dvc
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent      # the DVC workspace when DVC runs the stages
SCRIPT = Path(__file__).name
WORKSPACE = PROJECT_DIR / "outputs" / "dvc-workspace"
REMOTE = PROJECT_DIR / "outputs" / "dvc-remote-storage"

# ---------------------------------------------------------------------------
# Project files written into the DVC workspace by `setup`
# ---------------------------------------------------------------------------
PARAMS_YAML = """# Hyper-parameters tracked by DVC. Changing a value here makes `dvc repro` re-run only the affected stages.
prepare:
  test_size: 0.2
  random_state: 42

train:
  n_estimators: 100
  max_depth: 5
  random_state: 42
"""

DVC_YAML = """# DVC pipeline: data -> prepare -> train -> evaluate
# Each stage lists its dependencies (deps), parameters (params) and outputs (outs).
# All stages are subcommands of dvc_model_versioning.py (copied next to this file by `setup`).
stages:
  prepare:
    cmd: python dvc_model_versioning.py prepare
    deps:
      - data/raw/data.csv
      - dvc_model_versioning.py
    params:
      - prepare
    outs:
      - data/processed

  train:
    cmd: python dvc_model_versioning.py train
    deps:
      - data/processed
      - dvc_model_versioning.py
    params:
      - train
    outs:
      - models/model.joblib

  evaluate:
    cmd: python dvc_model_versioning.py evaluate
    deps:
      - data/processed
      - models/model.joblib
      - dvc_model_versioning.py
    metrics:
      - metrics.json:
          cache: false
    plots:
      - plots/confusion_matrix.csv:
          template: confusion
          x: actual
          y: predicted
          cache: false
"""

GITATTRIBUTES = """# Keep files byte-identical so DVC hashes do not change on Windows checkouts
* -text
"""

GITIGNORE = "__pycache__/\n"


def load_params(section):
    return yaml.safe_load((PROJECT_DIR / "params.yaml").read_text())[section]


# ---------------------------------------------------------------------------
# Data: Breast Cancer Wisconsin
#   --version 1 : original data (first 400 rows)
#   --version 2 : full data (569 rows) -> simulates "new data arrived"
# DVC tracks data/raw/data.csv, so each version can be restored later.
# ---------------------------------------------------------------------------
def get_data(version):
    df = load_breast_cancer(as_frame=True).frame
    if version == 1:
        df = df.sample(400, random_state=0)
    out = PROJECT_DIR / "data" / "raw" / "data.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Data version {version}: {len(df)} rows written to {out.relative_to(PROJECT_DIR)}")


# DVC stage 1 - prepare: clean and split the raw data
def prepare():
    params = load_params("prepare")
    df = pd.read_csv(PROJECT_DIR / "data" / "raw" / "data.csv").drop_duplicates().dropna()
    train_df, test_df = train_test_split(df, test_size=params["test_size"],
                                         random_state=params["random_state"], stratify=df["target"])
    out = PROJECT_DIR / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out / "train.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)
    print(f"prepare: train={len(train_df)} rows, test={len(test_df)} rows")


# DVC stage 2 - train: fit a Random Forest using the hyper-parameters in params.yaml
def train():
    params = load_params("train")
    train_df = pd.read_csv(PROJECT_DIR / "data" / "processed" / "train.csv")
    model = RandomForestClassifier(**params)
    model.fit(train_df.drop(columns=["target"]), train_df["target"])
    out = PROJECT_DIR / "models"
    out.mkdir(exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    print(f"train: RandomForest {params} saved to models/model.joblib")


# DVC stage 3 - evaluate: write metrics.json (tracked by `dvc metrics`) and data for `dvc plots`
def evaluate():
    test_df = pd.read_csv(PROJECT_DIR / "data" / "processed" / "test.csv")
    X, y = test_df.drop(columns=["target"]), test_df["target"]
    model = joblib.load(PROJECT_DIR / "models" / "model.joblib")
    pred = model.predict(X)
    metrics = {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, model.predict_proba(X)[:, 1]), 4),
        "test_rows": len(test_df),
    }
    (PROJECT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plots = PROJECT_DIR / "plots"
    plots.mkdir(exist_ok=True)
    pd.DataFrame({"actual": y, "predicted": pred}).to_csv(plots / "confusion_matrix.csv", index=False)
    print("evaluate:", metrics)


# ---------------------------------------------------------------------------
# DVC walkthrough
# ---------------------------------------------------------------------------
# UTF-8 output (dvc prints tree characters) and no analytics prompt
ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "DVC_NO_ANALYTICS": "1"}


def run(*cmd, check=True):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=WORKSPACE, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", env=ENV)
    output = (result.stdout + result.stderr).strip()
    if output:
        print("  " + output.replace("\n", "\n  "))
    if check and result.returncode != 0:
        sys.exit(f"Command failed: {' '.join(cmd)}")
    return result


def step(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def force_remove(path):
    def on_error(func, p, _):
        Path(p).chmod(0o777)
        func(p)
    if path.exists():
        shutil.rmtree(path, onerror=on_error)


def setup():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console fix
    if shutil.which("dvc") is None:
        sys.exit("DVC is not installed. Run:  pip install dvc")

    force_remove(WORKSPACE)
    force_remove(REMOTE)
    WORKSPACE.mkdir(parents=True)
    for name, text in {"params.yaml": PARAMS_YAML, "dvc.yaml": DVC_YAML,
                       ".gitattributes": GITATTRIBUTES, ".gitignore": GITIGNORE}.items():
        (WORKSPACE / name).write_text(text)
    shutil.copy2(Path(__file__), WORKSPACE / SCRIPT)
    py = sys.executable

    # -----------------------------------------------------------------------
    step("STEP 1: Initialise Git and DVC, configure remote storage")
    run("git", "init", "-b", "main")
    if not run("git", "config", "user.name", check=False).stdout.strip():
        run("git", "config", "user.name", "MLOps Student")
        run("git", "config", "user.email", "student@example.com")
    run("dvc", "init")
    # A local folder acts as remote storage. In real projects: S3 / GCS / Azure Blob / Google Drive
    run("dvc", "remote", "add", "-d", "localstorage", str(REMOTE))
    run("git", "add", ".")
    run("git", "commit", "-m", "Initialise DVC project with pipeline and params")

    # -----------------------------------------------------------------------
    step("STEP 2: Version the dataset (data v1)")
    run(py, SCRIPT, "get-data", "--version", "1")
    run("dvc", "add", "data/raw/data.csv")          # creates data/raw/data.csv.dvc (small pointer file)
    print("\nContent of the pointer file committed to Git:")
    print((WORKSPACE / "data" / "raw" / "data.csv.dvc").read_text())
    run("git", "add", "data/raw/data.csv.dvc", "data/raw/.gitignore")
    run("git", "commit", "-m", "Add raw data v1 (tracked by DVC)")

    # -----------------------------------------------------------------------
    step("STEP 3: Run the reproducible pipeline and version model v1")
    run("dvc", "dag")
    run("dvc", "repro")
    run("dvc", "metrics", "show")
    run("git", "add", "-A")
    run("git", "commit", "-m", "Model v1: trained on data v1")
    run("git", "tag", "-a", "v1.0", "-m", "data v1 + model v1")
    run("dvc", "push")                              # upload data + model to remote storage
    run("dvc", "repro")                             # nothing changed -> every stage is skipped (cached)

    # -----------------------------------------------------------------------
    step("STEP 4: Experiment tracking with dvc exp")
    run("dvc", "exp", "run", "-n", "trees-200", "--set-param", "train.n_estimators=200")
    run("dvc", "exp", "run", "-n", "depth-2", "--set-param", "train.max_depth=2")
    run("dvc", "exp", "show", "--no-pager", "--md", "--drop", "data|dvc_model|models|prepare|Created")
    # Experiments are applied to the workspace - restore the committed v1 state before continuing
    run("git", "reset", "--hard", "HEAD")
    run("dvc", "checkout", "--force")

    # -----------------------------------------------------------------------
    step("STEP 5: New data arrives -> data v2 + model v2")
    run(py, SCRIPT, "get-data", "--version", "2")
    run("dvc", "status")                            # DVC detects that the data changed
    run("dvc", "add", "data/raw/data.csv")
    run("dvc", "repro")
    run("dvc", "metrics", "diff", "v1.0")           # compare with the v1 model
    run("git", "add", "-A")
    run("git", "commit", "-m", "Model v2: retrained on data v2")
    run("git", "tag", "-a", "v2.0", "-m", "data v2 + model v2")
    run("dvc", "push")

    # -----------------------------------------------------------------------
    step("STEP 6: Reproducibility - go back to version 1")
    run("git", "checkout", "v1.0")
    run("dvc", "checkout")                          # restores data v1 + model v1 from the cache
    rows = sum(1 for _ in open(WORKSPACE / "data" / "raw" / "data.csv")) - 1
    print(f"\nRows in data/raw/data.csv after checkout v1.0: {rows}  (v1 had 400)")
    print("metrics.json at v1.0:", (WORKSPACE / "metrics.json").read_text())
    run("git", "checkout", "main")
    run("dvc", "checkout")
    run("git", "log", "--oneline", "--decorate")

    print("\nWorkspace:", WORKSPACE)
    print("Experiment 9 completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 9 - model versioning with DVC")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("setup", help="run the full DVC walkthrough (default)")
    get = sub.add_parser("get-data", help="write data/raw/data.csv")
    get.add_argument("--version", type=int, default=1, choices=[1, 2])
    for name in ["prepare", "train", "evaluate"]:
        sub.add_parser(name, help=f"DVC stage: {name}")
    args = parser.parse_args()

    if args.command == "get-data":
        get_data(args.version)
    elif args.command in ("prepare", "train", "evaluate"):
        {"prepare": prepare, "train": train, "evaluate": evaluate}[args.command]()
    else:
        setup()
