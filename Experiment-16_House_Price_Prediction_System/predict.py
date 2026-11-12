"""
Command-line house price prediction.
    python predict.py                                  -> a few example houses
    python predict.py --medinc 8.3 --houseage 41 --averooms 6.9 --avebedrms 1.0
                      --population 322 --aveoccup 2.6 --latitude 37.88 --longitude -122.23
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from features import FeatureEngineer  # noqa: F401  (needed to unpickle the pipeline)

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "model" / "house_price_model.joblib")
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]


def predict(house):
    row = pd.DataFrame([house])[FEATURES]
    if (row[["MedInc", "AveRooms", "AveOccup"]] <= 0).any(axis=None):
        raise ValueError("MedInc, AveRooms and AveOccup must be positive")
    # target is in units of $100,000; the floor avoids negative prices from the linear model
    return max(float(model.predict(row)[0]) * 100_000, 15_000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in FEATURES:
        parser.add_argument(f"--{name.lower()}", type=float)
    a = vars(parser.parse_args())

    if all(a[name.lower()] is not None for name in FEATURES):
        houses = {"Your house": {name: a[name.lower()] for name in FEATURES}}
    else:
        houses = {
            "Bay Area, high income": dict(MedInc=8.3, HouseAge=41, AveRooms=6.9, AveBedrms=1.0,
                                          Population=322, AveOccup=2.6, Latitude=37.88, Longitude=-122.23),
            "Los Angeles, middle income": dict(MedInc=4.0, HouseAge=30, AveRooms=5.0, AveBedrms=1.1,
                                               Population=1200, AveOccup=3.0, Latitude=34.05, Longitude=-118.25),
            "Central Valley, low income": dict(MedInc=2.0, HouseAge=25, AveRooms=5.2, AveBedrms=1.1,
                                               Population=1500, AveOccup=3.2, Latitude=36.7, Longitude=-119.8),
        }
    for name, house in houses.items():
        print(f"{name:28s} -> predicted price ${predict(house):,.0f}")
