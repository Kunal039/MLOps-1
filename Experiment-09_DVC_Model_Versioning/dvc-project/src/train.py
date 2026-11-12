"""DVC stage 2 - train: fit a Random Forest using the hyper-parameters in params.yaml."""

from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

PROJECT_DIR = Path(__file__).resolve().parents[1]
params = yaml.safe_load((PROJECT_DIR / "params.yaml").read_text())["train"]

train = pd.read_csv(PROJECT_DIR / "data" / "processed" / "train.csv")
model = RandomForestClassifier(**params)
model.fit(train.drop(columns=["target"]), train["target"])

out = PROJECT_DIR / "models"
out.mkdir(exist_ok=True)
joblib.dump(model, out / "model.joblib")
print(f"train: RandomForest {params} saved to models/model.joblib")
