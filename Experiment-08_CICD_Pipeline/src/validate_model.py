"""
Model quality gate (CI 'validate' stage).
Exits with code 1 if the new model is below the thresholds in params.json,
which makes the GitHub Actions / Jenkins job fail and stops deployment.
"""

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]


def main():
    gate = json.loads((PROJECT_DIR / "params.json").read_text())["quality_gate"]
    metrics = json.loads((PROJECT_DIR / "reports" / "metrics.json").read_text())

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


if __name__ == "__main__":
    main()
