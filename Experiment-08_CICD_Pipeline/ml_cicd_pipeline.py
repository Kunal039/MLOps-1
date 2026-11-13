"""
Experiment 8: Set up a CI/CD pipeline for automatic deployment of ML models
---------------------------------------------------------------------------
Aim:
    * Use Jenkins or GitHub Actions to automate testing and deployment of ML models.
    * Set up continuous integration for the training and validation of models.

Everything for the wine classifier lives in this one file. Each CI/CD stage is a subcommand:

    python ml_cicd_pipeline.py              -> run the whole pipeline locally (same as `pipeline`)
    python ml_cicd_pipeline.py pipeline     -> Write CI files -> Lint -> Test -> Train -> Validate -> Deploy
    python ml_cicd_pipeline.py write-ci     -> write the GitHub Actions workflow, Jenkinsfile, Dockerfile and
                                               requirements.txt for this project into outputs/ci/
    python ml_cicd_pipeline.py train        -> train with params.json, save model/ + reports/metrics.json
    python ml_cicd_pipeline.py validate     -> model quality gate (exit code 1 if below thresholds)
    python ml_cicd_pipeline.py predict      -> load the deployed model and predict sample wines
    python -m pytest ml_cicd_pipeline.py    -> the automated tests (data, model, prediction)

The generated GitHub Actions workflow and Jenkinsfile run exactly these commands on a CI server
(copy outputs/ci/ next to this script at the root of a repository to use them).
A stage that fails stops the pipeline, so a bad model is never deployed.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
REPORT_DIR = PROJECT_DIR / "reports"
SCRIPT = Path(__file__).name

EXPECTED_FEATURES = 13
EXPECTED_CLASSES = {0, 1, 2}
CLASSES = ["class_0", "class_1", "class_2"]


# Hyper-parameters and quality-gate thresholds used by training and validation
PARAMS = {
    "test_size": 0.2,
    "random_state": 42,
    "n_estimators": 150,
    "max_depth": 6,
    "quality_gate": {
        "min_accuracy": 0.90,
        "min_f1_macro": 0.90,
    },
}


def load_params():
    return PARAMS


# ---------------------------------------------------------------------------
# Data loading and data validation (first stage of the CI pipeline)
# ---------------------------------------------------------------------------
def load_data():
    wine = load_wine(as_frame=True)
    df = wine.frame
    return df.drop(columns=["target"]), df["target"]


def validate_data(X: pd.DataFrame, y: pd.Series):
    """Raise an error if the data does not look like the data the model was designed for."""
    assert X.shape[1] == EXPECTED_FEATURES, f"Expected {EXPECTED_FEATURES} features, got {X.shape[1]}"
    assert not X.isna().any().any(), "Data contains missing values"
    assert set(y.unique()) == EXPECTED_CLASSES, "Unexpected target classes"
    assert (X >= 0).all().all(), "Negative values found in chemical measurements"
    assert len(X) >= 100, "Not enough rows to train"
    return True


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# ---------------------------------------------------------------------------
# Train (CI 'train' stage)
# ---------------------------------------------------------------------------
def build_model(params):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=params["n_estimators"],
                                       max_depth=params["max_depth"],
                                       random_state=params["random_state"])),
    ])


def train():
    params = load_params()
    X, y = load_data()
    validate_data(X, y)
    X_train, X_test, y_train, y_test = split_data(X, y, params["test_size"], params["random_state"])

    model = build_model(params)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "cv_accuracy_mean": round(cv_scores.mean(), 4),
        "test_accuracy": round(accuracy_score(y_test, pred), 4),
        "test_f1_macro": round(f1_score(y_test, pred, average="macro"), 4),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Training complete:", json.dumps(metrics, indent=2))


# ---------------------------------------------------------------------------
# Model quality gate (CI 'validate' stage)
# Exits with code 1 if the new model is below the thresholds in params.json,
# which makes the GitHub Actions / Jenkins job fail and stops deployment.
# ---------------------------------------------------------------------------
def validate():
    gate = load_params()["quality_gate"]
    metrics = json.loads((REPORT_DIR / "metrics.json").read_text())

    checks = {
        "accuracy": (metrics["test_accuracy"], gate["min_accuracy"]),
        "f1_macro": (metrics["test_f1_macro"], gate["min_f1_macro"]),
    }
    passed = True
    for name, (value, minimum) in checks.items():
        ok = value >= minimum
        passed &= ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {value:.4f} (minimum {minimum})")

    if not passed:
        print("Quality gate FAILED - model will NOT be deployed.")
        sys.exit(1)
    print("Quality gate PASSED - model can be deployed.")


# ---------------------------------------------------------------------------
# Predict (the artifact deployed by the CD stage)
# ---------------------------------------------------------------------------
def predict(features, model=None):
    model = model or joblib.load(MODEL_DIR / "model.joblib")
    feature_names = list(model.feature_names_in_)
    if not isinstance(features, list) or len(features) != len(feature_names):
        raise ValueError(f"features must be a list of {len(feature_names)} numbers")
    row = pd.DataFrame([features], columns=feature_names)
    return CLASSES[int(model.predict(row)[0])], round(float(model.predict_proba(row).max()), 4)


def predict_samples():
    metrics = json.loads((REPORT_DIR / "metrics.json").read_text())
    print("Deployed model metrics:", metrics)
    model = joblib.load(MODEL_DIR / "model.joblib")
    X, y = load_data()
    for i in [0, 70, 150]:
        label, confidence = predict(X.iloc[i].tolist(), model)
        print(f"Wine #{i}: predicted {label} (confidence {confidence}), actual class_{y.iloc[i]}")


# ---------------------------------------------------------------------------
# Tests executed automatically by the CI pipeline on every push / pull request
#   * data tests       - is the input data valid?
#   * model tests      - does training produce a good enough model?
#   * prediction tests - does the deployed model predict correctly?
# ---------------------------------------------------------------------------
def test_data_is_valid():
    X, y = load_data()
    assert validate_data(X, y)


def test_validation_catches_missing_values():
    X, y = load_data()
    X.iloc[0, 0] = None
    with pytest.raises(AssertionError):
        validate_data(X, y)


def test_split_is_stratified():
    X, y = load_data()
    _, _, y_train, y_test = split_data(X, y)
    assert abs(y_train.mean() - y_test.mean()) < 0.1


def test_model_beats_quality_gate():
    params = load_params()
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = build_model(params).fit(X_train, y_train)
    assert model.score(X_test, y_test) >= params["quality_gate"]["min_accuracy"]


def test_model_is_deterministic():
    params = load_params()
    X, y = load_data()
    X_train, X_test, y_train, _ = split_data(X, y)
    p1 = build_model(params).fit(X_train, y_train).predict(X_test)
    p2 = build_model(params).fit(X_train, y_train).predict(X_test)
    assert (p1 == p2).all()


@pytest.fixture(scope="module")
def trained_model():
    if not (MODEL_DIR / "model.joblib").exists():
        train()
    return joblib.load(MODEL_DIR / "model.joblib")


def test_prediction_is_correct(trained_model):
    X, _ = load_data()
    label, confidence = predict(X.iloc[0].tolist(), trained_model)
    assert label == "class_0" and 0 <= confidence <= 1


def test_prediction_rejects_bad_input(trained_model):
    with pytest.raises(ValueError):
        predict([1, 2], trained_model)


# ---------------------------------------------------------------------------
# CI/CD configuration (written to outputs/ci/ by `write-ci`)
# ---------------------------------------------------------------------------
REQUIREMENTS_TXT = """pandas>=2.0
scikit-learn>=1.4
joblib>=1.3
pytest>=8.0
flake8>=7.0
"""

GITHUB_ACTIONS_YML = """# Experiment 8: CI/CD pipeline for automatic testing and deployment of ML models
# Place at <repo-root>/.github/workflows/ml-cicd.yml next to ml_cicd_pipeline.py and requirements.txt.
name: ML CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:          # allows manual runs from the Actions tab

env:
  IMAGE_NAME: wine-classifier

jobs:
  # ---------------- Continuous Integration ----------------
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Lint (code quality)
        run: flake8 --max-line-length 120 .
      - name: Unit tests (data, model, prediction)
        run: pytest -v ml_cicd_pipeline.py

  train-and-validate:
    needs: lint-and-test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -r requirements.txt
      - name: Train model
        run: python ml_cicd_pipeline.py train
      - name: Model quality gate
        run: python ml_cicd_pipeline.py validate
      - name: Upload model + metrics as build artifact
        uses: actions/upload-artifact@v4
        with:
          name: model-${{ github.sha }}
          path: |
            model/
            reports/

  # ---------------- Continuous Deployment ----------------
  build-and-deploy:
    needs: train-and-validate
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - name: Download validated model
        uses: actions/download-artifact@v4
        with:
          name: model-${{ github.sha }}
      - name: Build Docker image
        run: docker build -t ghcr.io/${{ github.repository_owner }}/${{ env.IMAGE_NAME }}:${{ github.sha }} .
      - name: Smoke test the container (runs a prediction)
        run: docker run --rm ghcr.io/${{ github.repository_owner }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
      - name: Push image to GitHub Container Registry
        run: |
          echo "${{ secrets.GITHUB_TOKEN }}" | docker login ghcr.io -u ${{ github.actor }} --password-stdin
          docker push ghcr.io/${{ github.repository_owner }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
      - name: Deploy
        run: echo "Deploy step - e.g. update the image tag on AWS ECS / GCP Cloud Run / Azure Container Apps"
"""

JENKINSFILE = """// Experiment 8 - the same CI/CD pipeline written for Jenkins (alternative to GitHub Actions)
pipeline {
    agent any
    environment {
        IMAGE = "wine-classifier:${env.BUILD_NUMBER}"
    }
    stages {
        stage('Checkout') {
            steps { checkout scm }
        }
        stage('Install') {
            steps { sh 'python3 -m venv venv && . venv/bin/activate && pip install -r requirements.txt' }
        }
        stage('Lint') {
            steps { sh '. venv/bin/activate && flake8 --max-line-length 120 .' }
        }
        stage('Test') {
            steps { sh '. venv/bin/activate && pytest -v ml_cicd_pipeline.py --junitxml=reports/junit.xml' }
        }
        stage('Train') {
            steps { sh '. venv/bin/activate && python ml_cicd_pipeline.py train' }
        }
        stage('Validate model') {
            steps { sh '. venv/bin/activate && python ml_cicd_pipeline.py validate' }
        }
        stage('Build image') {
            steps { sh 'docker build -t $IMAGE .' }
        }
        stage('Deploy') {
            when { branch 'main' }
            steps {
                sh 'docker run --rm $IMAGE'
                sh 'docker tag $IMAGE wine-classifier:latest'
            }
        }
    }
    post {
        always { archiveArtifacts artifacts: 'reports/**', allowEmptyArchive: true }
        success { echo 'Model trained, validated and deployed.' }
        failure { echo 'Pipeline failed - deployment skipped.' }
    }
}
"""

DOCKERFILE = """FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir pandas scikit-learn joblib pytest
COPY ml_cicd_pipeline.py ./
# model/ and reports/ come from the CI 'train-and-validate' job (validated model only)
COPY model/ model/
COPY reports/ reports/
CMD ["python", "ml_cicd_pipeline.py", "predict"]
"""


def write_ci():
    ci_dir = PROJECT_DIR / "outputs" / "ci"
    files = {
        ci_dir / ".github" / "workflows" / "ml-cicd.yml": GITHUB_ACTIONS_YML,
        ci_dir / "Jenkinsfile": JENKINSFILE,
        ci_dir / "Dockerfile": DOCKERFILE,
        ci_dir / "requirements.txt": REQUIREMENTS_TXT,
    }
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print("Written:", path.relative_to(PROJECT_DIR))


# ---------------------------------------------------------------------------
# Local CI/CD pipeline - the same stages GitHub Actions / Jenkins run on a CI server
# ---------------------------------------------------------------------------
STAGES = [
    ("Write CI/CD config files", [sys.executable, SCRIPT, "write-ci"]),
    ("Lint (code quality)", [sys.executable, "-m", "flake8", "--max-line-length", "120", SCRIPT]),
    ("Unit tests (data, model, prediction)", [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", SCRIPT]),
    ("Train model", [sys.executable, SCRIPT, "train"]),
    ("Validate model (quality gate)", [sys.executable, SCRIPT, "validate"]),
    ("Deploy - smoke test the model", [sys.executable, SCRIPT, "predict"]),
]


def run_pipeline():
    print("=" * 70 + "\nML CI/CD PIPELINE\n" + "=" * 70)
    summary = []
    for i, (name, cmd) in enumerate(STAGES, start=1):
        print(f"\n[Stage {i}/{len(STAGES)}] {name}\n$ {' '.join(cmd[1:])}")
        start = time.perf_counter()
        result = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True)
        duration = time.perf_counter() - start
        output = (result.stdout + result.stderr).strip()
        print(output[-1500:] if output else "(no output - OK)")

        status = "PASSED" if result.returncode == 0 else "FAILED"
        summary.append((name, status, duration))
        if result.returncode != 0:
            print(f"\nStage '{name}' FAILED - pipeline stopped, deployment skipped.")
            break

    print("\n" + "=" * 70 + "\nPIPELINE SUMMARY\n" + "=" * 70)
    for name, status, duration in summary:
        print(f"{status:7s} {duration:6.2f}s  {name}")
    if any(s == "FAILED" for _, s, _ in summary):
        sys.exit(1)
    print("\nAll stages passed - model deployed. Experiment 8 completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 8 - ML CI/CD pipeline")
    parser.add_argument("command", nargs="?", default="pipeline",
                        choices=["pipeline", "write-ci", "train", "validate", "predict"])
    command = parser.parse_args().command
    commands = {"pipeline": run_pipeline, "write-ci": write_ci, "train": train,
                "validate": validate, "predict": predict_samples}
    commands[command]()
