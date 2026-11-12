"""
Command-line survival prediction.
    python predict.py                                   -> a few example passengers
    python predict.py --pclass 1 --gender female --age 29 --fare 100
"""

import argparse
from pathlib import Path

import joblib

from features import passenger_frame

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "model" / "titanic_tree.joblib")


def predict(passenger):
    prob = float(model.predict_proba(passenger_frame(passenger))[0, 1])
    return ("Survived" if prob >= 0.5 else "Did not survive"), prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pclass", type=int)
    parser.add_argument("--gender", choices=["male", "female"])
    parser.add_argument("--age", type=float)
    parser.add_argument("--sibsp", type=int, default=0)
    parser.add_argument("--parch", type=int, default=0)
    parser.add_argument("--fare", type=float, default=15)
    parser.add_argument("--embarked", choices=["C", "Q", "S"], default="S")
    parser.add_argument("--title", choices=["Mr", "Mrs", "Miss", "Master", "Rare"])
    a = parser.parse_args()

    if a.pclass and a.gender:
        title = a.title or ("Mr" if a.gender == "male" else "Miss")
        people = {"You": {"Pclass": a.pclass, "Gender": a.gender, "Age": a.age, "SibSp": a.sibsp,
                          "Parch": a.parch, "Fare": a.fare, "Embarked": a.embarked, "Title": title}}
    else:
        people = {
            "Rose (1st class woman, 17)": {"Pclass": 1, "Gender": "female", "Age": 17, "SibSp": 0, "Parch": 1,
                                           "Fare": 100, "Embarked": "S", "Title": "Miss"},
            "Jack (3rd class man, 20)": {"Pclass": 3, "Gender": "male", "Age": 20, "Fare": 8, "Title": "Mr"},
            "Boy (3rd class, 6)": {"Pclass": 3, "Gender": "male", "Age": 6, "SibSp": 1, "Parch": 2,
                                   "Fare": 21, "Title": "Master"},
        }
    for name, p in people.items():
        label, prob = predict(p)
        print(f"{name:30s} -> {label:16s} (survival probability {prob:.0%})")
