"""
Generate the student performance dataset (1,000 students) used in Experiment 18.

The scores follow realistic relationships (study hours, attendance and previous marks
matter most; sleep, tutoring and parental education help; heavy social-media use hurts),
plus random noise, a few missing values and some inconsistent entries to clean.
Replace data/student_performance.csv with your own college data if available
(keep the same column names).
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "data" / "student_performance.csv"


def generate(n=1000, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "student_id": [f"S{1000 + i}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "age": rng.integers(17, 23, n),
        "study_hours_per_week": np.clip(rng.normal(14, 6, n), 0, 40).round(1),
        "attendance_pct": np.clip(rng.normal(82, 11, n), 35, 100).round(1),
        "previous_score": np.clip(rng.normal(66, 13, n), 25, 100).round(0),
        "assignments_completed": rng.integers(3, 11, n),
        "sleep_hours": np.clip(rng.normal(7, 1.1, n), 4, 10).round(1),
        "social_media_hours": np.clip(rng.gamma(2, 1.4, n), 0, 10).round(1),
        "parental_education": rng.choice(["High School", "Bachelor", "Master", "PhD"], n, p=[.35, .4, .2, .05]),
        "internet_access": rng.choice(["Yes", "No"], n, p=[.88, .12]),
        "extracurricular": rng.choice(["Yes", "No"], n, p=[.45, .55]),
        "tutoring": rng.choice(["Yes", "No"], n, p=[.3, .7]),
    })
    edu = df["parental_education"].map({"High School": 0, "Bachelor": 1.5, "Master": 3, "PhD": 4})
    score = (0
             + 0.95 * df["study_hours_per_week"]
             + 0.22 * df["attendance_pct"]
             + 0.38 * df["previous_score"]
             + 1.1 * df["assignments_completed"]
             - 1.6 * (df["sleep_hours"] - 7.5) ** 2
             - 1.3 * df["social_media_hours"]
             + edu
             + 3.5 * (df["tutoring"] == "Yes")
             + 2.0 * (df["internet_access"] == "Yes")
             + 1.0 * (df["extracurricular"] == "Yes")
             + rng.normal(0, 5, n))
    df["final_score"] = np.clip(score, 0, 100).round(1)
    df["result"] = np.where(df["final_score"] >= 55, "Pass", "Fail")

    # data-quality problems
    for col in ["study_hours_per_week", "attendance_pct", "sleep_hours", "parental_education"]:
        df.loc[rng.choice(n, 25, replace=False), col] = np.nan
    df.loc[rng.choice(n, 15, replace=False), "gender"] = rng.choice(["male", "F", "M ", "female"], 15)
    return pd.concat([df, df.sample(8, random_state=1)])     # duplicates


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    data = generate()
    data.to_csv(OUT, index=False)
    print(f"Saved {len(data)} rows to {OUT}")
    print(data["result"].value_counts())
