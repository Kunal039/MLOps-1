"""
Experiment 9: Implement model versioning using DVC for efficient model management
---------------------------------------------------------------------------------
Aim:
    * Install and configure DVC in a project.
    * Version control data, models and experiments for reproducibility.

The script copies dvc-project/ into outputs/dvc-workspace (a fresh Git + DVC repo)
and runs every DVC command step-by-step, printing each one like a terminal session:

    1. git init + dvc init + configure a (local) DVC remote storage
    2. Version data v1        -> dvc add, git commit, dvc push, git tag
    3. Run the ML pipeline     -> dvc repro (prepare -> train -> evaluate), dvc metrics show
    4. Track experiments       -> dvc exp run --set-param ..., dvc exp show
    5. Version data v2 + model -> dvc add, dvc repro, dvc metrics diff
    6. Reproducibility         -> git checkout v1 + dvc checkout restores the old data & model

Prerequisite:  pip install dvc
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console fix

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE = BASE_DIR / "dvc-project"
WORKSPACE = BASE_DIR / "outputs" / "dvc-workspace"
REMOTE = BASE_DIR / "outputs" / "dvc-remote-storage"


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


if shutil.which("dvc") is None:
    sys.exit("DVC is not installed. Run:  pip install dvc")

force_remove(WORKSPACE)
force_remove(REMOTE)
shutil.copytree(TEMPLATE, WORKSPACE, ignore=shutil.ignore_patterns("__pycache__"))
PY = sys.executable
# UTF-8 output (dvc prints tree characters) and no analytics prompt
ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "DVC_NO_ANALYTICS": "1"}

# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
step("STEP 2: Version the dataset (data v1)")
run(PY, "src/get_data.py", "--version", "1")
run("dvc", "add", "data/raw/data.csv")          # creates data/raw/data.csv.dvc (small pointer file)
print("\nContent of the pointer file committed to Git:")
print((WORKSPACE / "data" / "raw" / "data.csv.dvc").read_text())
run("git", "add", "data/raw/data.csv.dvc", "data/raw/.gitignore")
run("git", "commit", "-m", "Add raw data v1 (tracked by DVC)")

# ---------------------------------------------------------------------------
step("STEP 3: Run the reproducible pipeline and version model v1")
run("dvc", "dag")
run("dvc", "repro")
run("dvc", "metrics", "show")
run("git", "add", "dvc.lock", "metrics.json", "plots", "data/.gitignore", ".gitignore")
run("git", "add", "-A")
run("git", "commit", "-m", "Model v1: trained on data v1")
run("git", "tag", "-a", "v1.0", "-m", "data v1 + model v1")
run("dvc", "push")                              # upload data + model to remote storage
run("dvc", "repro")                             # nothing changed -> every stage is skipped (cached)

# ---------------------------------------------------------------------------
step("STEP 4: Experiment tracking with dvc exp")
run("dvc", "exp", "run", "-n", "trees-200", "--set-param", "train.n_estimators=200")
run("dvc", "exp", "run", "-n", "depth-2", "--set-param", "train.max_depth=2")
run("dvc", "exp", "show", "--no-pager", "--md", "--drop", "data|src|models|prepare|Created")
# Experiments are applied to the workspace - restore the committed v1 state before continuing
run("git", "reset", "--hard", "HEAD")
run("dvc", "checkout", "--force")

# ---------------------------------------------------------------------------
step("STEP 5: New data arrives -> data v2 + model v2")
run(PY, "src/get_data.py", "--version", "2")
run("dvc", "status")                            # DVC detects that the data changed
run("dvc", "add", "data/raw/data.csv")
run("dvc", "repro")
run("dvc", "metrics", "diff", "v1.0")           # compare with the v1 model
run("git", "add", "-A")
run("git", "commit", "-m", "Model v2: retrained on data v2")
run("git", "tag", "-a", "v2.0", "-m", "data v2 + model v2")
run("dvc", "push")

# ---------------------------------------------------------------------------
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
