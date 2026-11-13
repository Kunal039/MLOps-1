"""
Experiment 20 - Step 2: Salary prediction with input validation and prediction logging
    python predict.py          -> predict salaries for a few example employees

    predict_salary({"experience_years": 5, "skills_count": 8, "certifications": 1, "performance_rating": 4,
                    "job_role": "Data Scientist", "city": "Bengaluru", "company_size": "Enterprise",
                    "education": "Master"})
Every prediction is appended to logs/predictions.jsonl - the input data for drift monitoring.
"""

import json
import logging
import time
from pathlib import Path

import joblib
import pandas as pd

from generate_data import CITY, EDUCATION, ROLES, SIZE

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
model = joblib.load(BASE_DIR / "model" / "salary_model.joblib")
info = json.loads((BASE_DIR / "model" / "model_info.json").read_text())
FEATURES = info["features"]
ALLOWED = {"job_role": set(ROLES), "city": set(CITY), "company_size": set(SIZE), "education": set(EDUCATION)}
RANGES = {"experience_years": (0, 45), "skills_count": (0, 50), "certifications": (0, 30), "performance_rating": (1, 5)}

log = logging.getLogger("salary-model")
log.setLevel(logging.INFO)
_handler = logging.FileHandler(LOG_DIR / "service.log")
_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
log.addHandler(_handler)
prediction_log = (LOG_DIR / "predictions.jsonl").open("a", buffering=1)


def validate(employee):
    """Return an error message for invalid input, else None."""
    for field in FEATURES:
        if field not in employee:
            return f"Missing field '{field}'"
    for field, allowed in ALLOWED.items():
        if employee[field] not in allowed:
            return f"'{field}' must be one of {sorted(allowed)}"
    for field, (lo, hi) in RANGES.items():
        try:
            value = float(employee[field])
        except (TypeError, ValueError):
            return f"'{field}' must be a number"
        if not lo <= value <= hi:
            return f"'{field}' must be between {lo} and {hi}"
    return None


def predict_salary(employee):
    """Predicted salary (lakh INR per year) with a 95% range. Raises ValueError for invalid input."""
    error = validate(employee)
    if error:
        log.warning("Rejected input: %s", error)
        raise ValueError(error)

    salary = float(model.predict(pd.DataFrame([{f: employee[f] for f in FEATURES}]))[0])
    prediction_log.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                     **{f: employee[f] for f in FEATURES}, "predicted_salary_lpa": round(salary, 2),
                                     "model_version": info["registry_version"]}) + "\n")
    rmse = info["metrics"]["test_rmse"]
    return {"predicted_salary_lpa": round(salary, 2),
            "expected_range_lpa": [round(max(salary - 1.96 * rmse, 0), 2), round(salary + 1.96 * rmse, 2)]}


if __name__ == "__main__":
    print(f"Model: {info['algorithm']} (MLflow registry version {info['registry_version']}, "
          f"test RMSE {info['metrics']['test_rmse']} LPA)\n")
    examples = {
        "Fresher data analyst, Jaipur": dict(experience_years=0.5, skills_count=4, certifications=0,
                                             performance_rating=3, job_role="Data Analyst", city="Jaipur",
                                             company_size="Startup", education="Bachelor"),
        "Data scientist, 5 yrs, Bengaluru": dict(experience_years=5, skills_count=8, certifications=1,
                                                 performance_rating=4, job_role="Data Scientist", city="Bengaluru",
                                                 company_size="Enterprise", education="Master"),
        "Product manager, 12 yrs, Mumbai": dict(experience_years=12, skills_count=12, certifications=2,
                                                performance_rating=5, job_role="Product Manager", city="Mumbai",
                                                company_size="Enterprise", education="Master"),
    }
    for name, employee in examples.items():
        r = predict_salary(employee)
        low, high = r["expected_range_lpa"]
        print(f"{name:34s} -> {r['predicted_salary_lpa']:6.2f} LPA  (95% range {low}-{high})")
    try:
        predict_salary({"experience_years": 5})
    except ValueError as err:
        print("\nInvalid input rejected:", err)
