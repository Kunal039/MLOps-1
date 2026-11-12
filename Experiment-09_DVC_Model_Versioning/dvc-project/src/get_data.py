"""
Create the raw dataset (Breast Cancer Wisconsin).
    --version 1 : original data (first 400 rows)
    --version 2 : full data (569 rows) -> simulates "new data arrived"
DVC tracks data/raw/data.csv, so each version can be restored later.
"""

import argparse
from pathlib import Path

from sklearn.datasets import load_breast_cancer

PROJECT_DIR = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--version", type=int, default=1, choices=[1, 2])
args = parser.parse_args()

df = load_breast_cancer(as_frame=True).frame
if args.version == 1:
    df = df.sample(400, random_state=0)

out = PROJECT_DIR / "data" / "raw" / "data.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Data version {args.version}: {len(df)} rows written to {out.relative_to(PROJECT_DIR)}")
