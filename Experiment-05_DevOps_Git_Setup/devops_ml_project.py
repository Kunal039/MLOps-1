"""
Experiment 5: Set up a DevOps environment for ML deployment
-----------------------------------------------------------
Aim:
    * Set up a version-controlled project repository using Git.
    * Learn about the integration of ML workflows into the DevOps pipeline.

This one file is both the ML project (an Iris classifier) and the script that sets up its repository:

    python devops_ml_project.py                  -> full experiment (same as `setup`)
    python devops_ml_project.py train            -> build step: train, save models/model.joblib + metrics.json
    python devops_ml_project.py predict [4 nums] -> serve step: load the model and predict
    python -m pytest devops_ml_project.py        -> quality gates run in CI

What `setup` does:
    1. Creates outputs/ml-devops-repo with this file plus .gitignore, requirements.txt and a
       GitHub Actions workflow (written from the text below), so a fresh Git repository can be
       created without touching this folder.
    2. Initialises Git, makes the first commit and follows a branching workflow
       (main -> develop -> feature/*) like a real team.
    3. Runs a local DevOps pipeline: Code -> Build -> Test -> Train -> Validate -> Package -> Release
       which is exactly what the GitHub Actions workflow automates.
    4. Tags the released model version (v1.0.0) and prints the Git history.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent
SCRIPT = Path(__file__).name
MODEL_DIR = PROJECT_DIR / "models"
REPO = PROJECT_DIR / "outputs" / "ml-devops-repo"
CLASSES = ["setosa", "versicolor", "virginica"]
COLUMNS = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]


# ---------------------------------------------------------------------------
# Build step: train an Iris classifier and save the model + metrics
# ---------------------------------------------------------------------------
def load_data():
    iris = load_iris(as_frame=True)
    return train_test_split(iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)


def train(n_estimators=100, max_depth=None):
    X_train, X_test, y_train, y_test = load_data()
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "f1_macro": round(f1_score(y_test, pred, average="macro"), 4),
    }
    return model, metrics


def train_and_save():
    model, metrics = train()
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Model trained. Metrics:", metrics)


# ---------------------------------------------------------------------------
# Serve step: load the saved model and make a prediction
# ---------------------------------------------------------------------------
def predict(features):
    model = joblib.load(MODEL_DIR / "model.joblib")
    row = pd.DataFrame([features], columns=COLUMNS)
    return CLASSES[int(model.predict(row)[0])]


# ---------------------------------------------------------------------------
# Automated tests run by the CI pipeline before a model can be merged / deployed
# ---------------------------------------------------------------------------
def test_data_has_expected_shape():
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[1] == 4
    assert len(X_train) + len(X_test) == 150
    assert not X_train.isna().any().any()


def test_model_meets_accuracy_threshold():
    _, metrics = train()
    assert metrics["accuracy"] >= 0.90, "Model quality gate failed"


def test_model_predicts_valid_classes():
    model, _ = train()
    _, X_test, _, _ = load_data()
    assert set(model.predict(X_test)).issubset({0, 1, 2})


# ---------------------------------------------------------------------------
# Repository files written into outputs/ml-devops-repo by `setup`
# ---------------------------------------------------------------------------
GITIGNORE = """# Python
__pycache__/
*.pyc
.venv/
venv/
.pytest_cache/

# ML artifacts (models and large data are versioned with DVC / artifact store, not Git)
models/
*.joblib
*.pkl
data/raw/

# Environment / secrets
.env
"""

REQUIREMENTS_TXT = """scikit-learn>=1.4
pandas>=2.0
joblib>=1.3
pytest>=8.0
"""

CI_WORKFLOW_YML = """# Continuous Integration for the ML project.
# Every push / pull request: install deps -> run tests -> train -> upload model artifact.
name: ML CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build-test-train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run unit tests
        run: pytest -v devops_ml_project.py
      - name: Train model
        run: python devops_ml_project.py train
      - name: Upload model artifact
        uses: actions/upload-artifact@v4
        with:
          name: model
          path: models/
"""


# ---------------------------------------------------------------------------
# Repository setup + local DevOps pipeline
# ---------------------------------------------------------------------------
def run(cmd, check=True):
    """Run a shell command inside the repository and print it like a terminal."""
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    if output:
        print("   " + output.replace("\n", "\n   "))
    if check and result.returncode != 0:
        sys.exit(f"Command failed: {' '.join(cmd)}")
    return result


def git(*args, check=True):
    return run(["git", *args], check=check)


def stage(name):
    print("\n" + "-" * 70 + f"\n[PIPELINE STAGE] {name}\n" + "-" * 70)


def setup():
    # -----------------------------------------------------------------------
    # 1. Create a fresh project folder
    # -----------------------------------------------------------------------
    if REPO.exists():
        shutil.rmtree(REPO, onerror=lambda f, p, e: (Path(p).chmod(0o777), f(p)))
    (REPO / ".github" / "workflows").mkdir(parents=True)
    (REPO / ".gitignore").write_text(GITIGNORE)
    (REPO / "requirements.txt").write_text(REQUIREMENTS_TXT)
    (REPO / ".github" / "workflows" / "ml-ci.yml").write_text(CI_WORKFLOW_YML)
    shutil.copy2(Path(__file__), REPO / SCRIPT)
    print("Project created in:", REPO)

    # -----------------------------------------------------------------------
    # 2. Initialise Git repository
    # -----------------------------------------------------------------------
    stage("CODE - initialise version control")
    git("init", "-b", "main")
    # Use the student's global identity if set, otherwise a local placeholder identity
    if not git("config", "user.name", check=False).stdout.strip():
        git("config", "user.name", "MLOps Student")
        git("config", "user.email", "student@example.com")
    git("add", ".")
    git("commit", "-m", "Initial commit: ML project, tests and CI workflow")

    # develop branch for integration, feature branch for new work
    git("checkout", "-b", "develop")
    git("checkout", "-b", "feature/tune-model")

    # A change on the feature branch: tune hyper-parameters (only the first match is the train() definition)
    project_file = REPO / SCRIPT
    project_file.write_text(project_file.read_text().replace(
        "def train(n_estimators=100, max_depth=None)", "def train(n_estimators=200, max_depth=5)", 1))
    git("add", SCRIPT)
    git("commit", "-m", "feat: tune RandomForest (n_estimators=200, max_depth=5)")

    # -----------------------------------------------------------------------
    # 3. Run the ML pipeline on the feature branch (what CI does on every push)
    # -----------------------------------------------------------------------
    stage("BUILD - check the code compiles")
    run([sys.executable, "-m", "py_compile", SCRIPT])

    stage("TEST - run automated unit tests")
    run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", SCRIPT])

    stage("TRAIN - train the model")
    run([sys.executable, SCRIPT, "train"])

    stage("VALIDATE - model quality gate")
    metrics = json.loads((REPO / "models" / "metrics.json").read_text())
    threshold = 0.90
    print(f"accuracy={metrics['accuracy']} threshold={threshold}")
    if metrics["accuracy"] < threshold:
        sys.exit("Quality gate FAILED - model will not be released")
    print("Quality gate PASSED")

    stage("PACKAGE - smoke test the packaged model")
    run([sys.executable, SCRIPT, "predict", "6.7", "3.0", "5.2", "2.3"])

    # Note: models/ is in .gitignore, so the binary is NOT committed to Git
    git("status", "--short")

    # -----------------------------------------------------------------------
    # 4. Merge (pull request) and release
    # -----------------------------------------------------------------------
    stage("RELEASE - merge feature -> develop -> main and tag version")
    git("checkout", "develop")
    git("merge", "--no-ff", "feature/tune-model", "-m", "Merge feature/tune-model into develop")
    git("checkout", "main")
    git("merge", "--no-ff", "develop", "-m", "Release v1.0.0")
    git("tag", "-a", "v1.0.0", "-m", f"Model release v1.0.0 accuracy={metrics['accuracy']}")
    git("branch", "-d", "feature/tune-model")

    stage("HISTORY")
    git("log", "--oneline", "--graph", "--all", "--decorate")
    git("branch", "-a")
    git("tag", "-n")

    print("\nRepository ready at:", REPO)
    print("To push to GitHub:  git remote add origin <your-repo-url>  &&  git push -u origin main --tags")
    print("\nExperiment 5 completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 5 - DevOps environment for ML")
    parser.add_argument("command", nargs="?", default="setup", choices=["setup", "train", "predict"])
    parser.add_argument("features", nargs="*", type=float, help="4 flower measurements for `predict`")
    args = parser.parse_args()

    if args.command == "train":
        train_and_save()
    elif args.command == "predict":
        values = args.features if len(args.features) == 4 else [5.1, 3.5, 1.4, 0.2]
        print(f"Input: {values} -> Predicted species: {predict(values)}")
    else:
        setup()
