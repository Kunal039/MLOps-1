"""
Generate the employee salary dataset (annual salary in lakh INR).

    python generate_data.py                 -> data/employees.csv          (historical training data)
    python generate_data.py --new-batch     -> data/employees_new_batch.csv (newer data with drift:
                                               more senior hires, 12% salary inflation, more AI roles)

The historical data deliberately contains a small unexplained gender pay gap, so that the
fairness audit (fairness_audit.py) has something real to detect.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
ROLES = {"Data Analyst": 6, "Software Engineer": 9, "Data Scientist": 12, "ML Engineer": 14,
         "DevOps Engineer": 10, "Product Manager": 16, "HR Executive": 5, "Sales Executive": 5.5}
EDUCATION = {"Diploma": 0.85, "Bachelor": 1.0, "Master": 1.15, "PhD": 1.3}
CITY = {"Bengaluru": 1.15, "Mumbai": 1.12, "Delhi NCR": 1.08, "Hyderabad": 1.05, "Pune": 1.0, "Chennai": 0.98,
        "Jaipur": 0.85, "Chandigarh": 0.88}
SIZE = {"Startup": 0.95, "Mid-size": 1.0, "Enterprise": 1.1}


def generate(n=3000, seed=0, drift=False):
    rng = np.random.default_rng(seed)
    role_p = np.array([.15, .25, .12, .08, .1, .08, .11, .11])
    if drift:                                      # new hiring trend: more ML roles
        role_p = np.array([.1, .2, .15, .2, .1, .08, .08, .09])
    df = pd.DataFrame({
        "employee_id": [f"E{seed}{i:05d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n, p=[.62, .38]),
        "job_role": rng.choice(list(ROLES), n, p=role_p / role_p.sum()),
        "education": rng.choice(list(EDUCATION), n, p=[.1, .55, .3, .05]),
        "city": rng.choice(list(CITY), n),
        "company_size": rng.choice(list(SIZE), n, p=[.3, .4, .3]),
        "experience_years": np.clip(rng.gamma(2.2 if not drift else 3.2, 2.3, n), 0, 30).round(1),
        "skills_count": rng.integers(2, 16, n),
        "certifications": rng.poisson(1.2, n),
        "performance_rating": rng.choice([1, 2, 3, 4, 5], n, p=[.05, .15, .4, .3, .1]),
    })
    df["age"] = (22 + df["experience_years"] + rng.normal(1.5, 1.5, n)).round().astype(int)
    salary = (df["job_role"].map(ROLES)
              * df["education"].map(EDUCATION)
              * df["city"].map(CITY)
              * df["company_size"].map(SIZE)
              * (1 + 0.085 * df["experience_years"] - 0.0012 * df["experience_years"] ** 2)
              * (1 + 0.012 * df["skills_count"] + 0.02 * df["certifications"])
              * (0.9 + 0.05 * df["performance_rating"])
              * np.where(df["gender"] == "Female", 0.94, 1.0)          # historical bias (6% gap)
              * rng.lognormal(0, 0.08, n))
    if drift:
        salary *= 1.12                                                  # inflation / market correction
    df["salary_lpa"] = salary.round(2)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-batch", action="store_true")
    args = parser.parse_args()
    DATA_DIR.mkdir(exist_ok=True)
    if args.new_batch:
        out, df = DATA_DIR / "employees_new_batch.csv", generate(800, seed=1, drift=True)
    else:
        out, df = DATA_DIR / "employees.csv", generate()
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} employees to {out.name}; mean salary {df.salary_lpa.mean():.2f} LPA")
