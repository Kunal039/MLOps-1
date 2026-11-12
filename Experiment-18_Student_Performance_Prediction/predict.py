"""
Student Performance Prediction System - predict a student's final score and pass/fail risk,
and give simple, explainable study advice.

    python predict.py                 -> three example students
    python predict.py --interactive   -> enter your own details
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
bundle = joblib.load(BASE_DIR / "model" / "student_models.joblib")

EXAMPLES = {
    "Hard-working student": dict(gender="Female", age=19, study_hours_per_week=22, attendance_pct=95,
                                 previous_score=78, assignments_completed=10, sleep_hours=7.5,
                                 social_media_hours=1.5, parental_education="Bachelor", internet_access="Yes",
                                 extracurricular="Yes", tutoring="No"),
    "Average student": dict(gender="Male", age=20, study_hours_per_week=12, attendance_pct=80,
                            previous_score=62, assignments_completed=7, sleep_hours=7,
                            social_media_hours=3, parental_education="High School", internet_access="Yes",
                            extracurricular="No", tutoring="No"),
    "At-risk student": dict(gender="Male", age=21, study_hours_per_week=3, attendance_pct=55,
                            previous_score=40, assignments_completed=4, sleep_hours=5,
                            social_media_hours=7, parental_education="High School", internet_access="No",
                            extracurricular="No", tutoring="No"),
}


def advice(s):
    tips = []
    if s["study_hours_per_week"] < 10:
        tips.append("increase self-study to at least 10-15 hours/week")
    if s["attendance_pct"] < 75:
        tips.append("attend more classes (attendance below 75%)")
    if s["social_media_hours"] > 4:
        tips.append("reduce social-media time")
    if not 6.5 <= s["sleep_hours"] <= 8.5:
        tips.append("aim for 7-8 hours of sleep")
    if s["assignments_completed"] < 7:
        tips.append("complete all assignments")
    return tips or ["keep up the good work!"]


def predict(student):
    row = pd.DataFrame([student])[bundle["features"]]
    score = float(bundle["regressor"].predict(row)[0])
    passes = int(bundle["classifier"].predict(row)[0]) == 1
    return min(max(score, 0), 100), passes


def ask(prompt, cast=str, default=None):
    value = input(f"{prompt} [{default}]: ").strip()
    return cast(value) if value else default


if __name__ == "__main__":
    if "--interactive" in sys.argv:
        base = EXAMPLES["Average student"]
        students = {"You": {k: ask(k, type(v), v) for k, v in base.items()}}
    else:
        students = EXAMPLES
    for name, s in students.items():
        score, passes = predict(s)
        status = "PASS" if passes else "AT RISK OF FAILING"
        print(f"\n{name}: predicted final score {score:.1f}/100 -> {status}")
        print("  Advice:", "; ".join(advice(s)))
