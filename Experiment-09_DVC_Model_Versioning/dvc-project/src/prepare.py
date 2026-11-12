"""DVC stage 1 - prepare: clean and split the raw data."""

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
params = yaml.safe_load((PROJECT_DIR / "params.yaml").read_text())["prepare"]

df = pd.read_csv(PROJECT_DIR / "data" / "raw" / "data.csv").drop_duplicates().dropna()
train, test = train_test_split(df, test_size=params["test_size"],
                               random_state=params["random_state"], stratify=df["target"])

out = PROJECT_DIR / "data" / "processed"
out.mkdir(parents=True, exist_ok=True)
train.to_csv(out / "train.csv", index=False)
test.to_csv(out / "test.csv", index=False)
print(f"prepare: train={len(train)} rows, test={len(test)} rows")
