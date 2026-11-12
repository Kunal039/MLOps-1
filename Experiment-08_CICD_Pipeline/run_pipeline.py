"""
Experiment 8: Set up a CI/CD pipeline for automatic deployment of ML models
---------------------------------------------------------------------------
Aim:
    * Use Jenkins or GitHub Actions to automate testing and deployment of ML models.
    * Set up continuous integration for the training and validation of models.

This script runs, on your own PC, exactly the same stages that
.github/workflows/ml-cicd.yml (GitHub Actions) and Jenkinsfile (Jenkins) run on a CI server:

    Lint -> Test -> Train -> Validate (quality gate) -> Deploy (smoke test)

A stage that fails stops the pipeline, so a bad model is never deployed.
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

SMOKE_TEST = (
    "from app import app; c = app.test_client(); r = c.get('/health'); "
    "print('Health check:', r.status_code, r.json); assert r.status_code == 200"
)

STAGES = [
    ("Lint (code quality)", [sys.executable, "-m", "flake8", "."]),
    ("Unit tests (data, model, API)", [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]),
    ("Train model", [sys.executable, "src/train.py"]),
    ("Validate model (quality gate)", [sys.executable, "src/validate_model.py"]),
    ("Deploy - smoke test service", [sys.executable, "-c", SMOKE_TEST]),
]


def main():
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
    main()
