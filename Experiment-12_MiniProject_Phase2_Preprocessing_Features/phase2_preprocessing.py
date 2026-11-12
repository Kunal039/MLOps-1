"""
Experiment 12: Mini Project Phase-2 - Data preprocessing and feature engineering
================================================================================
Project : Household Electricity Consumption Prediction System
Input   : ../Experiment-11_.../data/household_power_consumption.txt  (run Phase-1 first)

Steps
    1. Parse date/time, convert '?' to missing values
    2. Treat outliers (faulty meter readings)
    3. Fill missing values (time interpolation + same-time-last-week for long gaps)
    4. Resample minute data -> hourly consumption (prediction target)
    5. Feature engineering: calendar features + lag features + rolling statistics
       (only PAST values are used, so there is no data leakage)
    6. Time-based split into train / validation / test (70 / 15 / 15)
    7. Normalise with StandardScaler fitted on the training set only
Output  : outputs/train.csv, validation.csv, test.csv, scaler.joblib, feature_config.json
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
RAW_FILE = PROJECT_ROOT / "Experiment-11_MiniProject_Phase1_Data_Collection_EDA" / "data" / "household_power_consumption.txt"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
TARGET = "Global_active_power"

if not RAW_FILE.exists():
    raise SystemExit("Raw data not found - run Experiment-11 (Phase-1) first.")

# 1. Load --------------------------------------------------------------------------
df = pd.read_csv(RAW_FILE, sep=";", na_values=["?"], low_memory=False)
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
df = df.drop(columns=["Date", "Time"]).set_index("Datetime").sort_index()
df = df[~df.index.duplicated()].asfreq("min")       # make the minute index complete
print(f"Loaded {len(df):,} minute readings, missing rows: {df[TARGET].isna().sum():,}")

# 2. Outliers -------------------------------------------------------------------
# A household connection cannot exceed ~11 kW; values far above the IQR fence are faulty readings.
q1, q3 = df[TARGET].quantile([0.25, 0.75])
limit = min(q3 + 5 * (q3 - q1), 11.0)
faulty = df[TARGET] > limit
print(f"Faulty readings above {limit:.2f} kW set to missing: {faulty.sum()}")
df.loc[faulty, TARGET] = np.nan

# 3. Missing values -------------------------------------------------------------
before = df[TARGET].isna().sum()
df = df.interpolate(method="time", limit=60)              # short gaps (< 1 hour): interpolation
week_ago = df.shift(7 * 24 * 60)                           # long gaps: same minute last week
df = df.fillna(week_ago).bfill()                           # very start of data: back-fill
print(f"Missing values filled: {before:,} -> {df[TARGET].isna().sum()}")

# 4. Resample to hourly ------------------------------------------------------------
hourly = df.resample("h").mean()
hourly["Sub_metering_other"] = (hourly[TARGET] * 1000 / 60
                                - hourly[["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]].sum(axis=1))
print(f"Hourly rows: {len(hourly):,}")

# 5. Feature engineering --------------------------------------------------------
feat = pd.DataFrame(index=hourly.index)
feat[TARGET] = hourly[TARGET]

# Calendar features (+ cyclic encoding so hour 23 is close to hour 0)
feat["hour"] = feat.index.hour
feat["day_of_week"] = feat.index.dayofweek
feat["month"] = feat.index.month
feat["is_weekend"] = (feat.index.dayofweek >= 5).astype(int)
feat["hour_sin"] = np.sin(2 * np.pi * feat["hour"] / 24)
feat["hour_cos"] = np.cos(2 * np.pi * feat["hour"] / 24)

# Lag features: consumption 1, 2, 3, 24 hours and 1 week ago
for lag in [1, 2, 3, 24, 168]:
    feat[f"lag_{lag}"] = feat[TARGET].shift(lag)

# Rolling statistics of PAST values (shift(1) so the current hour is not included)
past = feat[TARGET].shift(1)
feat["rolling_mean_3"] = past.rolling(3).mean()
feat["rolling_mean_24"] = past.rolling(24).mean()
feat["rolling_std_24"] = past.rolling(24).std()

# Previous-hour readings of the other meters
for col in ["Voltage", "Global_reactive_power", "Sub_metering_3"]:
    feat[f"{col}_lag_1"] = hourly[col].shift(1)

feat = feat.dropna()
FEATURES = [c for c in feat.columns if c != TARGET]
print(f"Feature rows after removing incomplete lags: {len(feat):,}")
print("Features:", FEATURES)

# 6. Time-based split (never shuffle time series!) ----------------------------------
n = len(feat)
train = feat.iloc[: int(n * 0.70)]
val = feat.iloc[int(n * 0.70): int(n * 0.85)]
test = feat.iloc[int(n * 0.85):]
for name, part in [("Train", train), ("Validation", val), ("Test", test)]:
    print(f"{name:10s}: {len(part):5d} rows  {part.index.min()} -> {part.index.max()}")

# 7. Scaling (fit on train only) --------------------------------------------------
scaler = StandardScaler().fit(train[FEATURES])
joblib.dump(scaler, OUTPUT_DIR / "scaler.joblib")

train.to_csv(OUTPUT_DIR / "train.csv")
val.to_csv(OUTPUT_DIR / "validation.csv")
test.to_csv(OUTPUT_DIR / "test.csv")
(OUTPUT_DIR / "feature_config.json").write_text(json.dumps(
    {"target": TARGET, "features": FEATURES, "frequency": "hourly",
     "lags": [1, 2, 3, 24, 168], "split": {"train": len(train), "validation": len(val), "test": len(test)}},
    indent=2))

# Plot the split
plt.figure(figsize=(14, 4))
for name, part in [("Train", train), ("Validation", val), ("Test", test)]:
    part[TARGET].resample("D").mean().plot(label=name)
plt.title("Hourly consumption (daily mean) - time-based split")
plt.ylabel("kW")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "train_val_test_split.png")
plt.close()

corr = train[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
corr.plot(kind="barh", figsize=(8, 6), title="Correlation of features with consumption")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_correlation.png")
plt.close()
print("\nTop correlated features:\n", corr.abs().sort_values(ascending=False).head(6).round(3))
print("\nExperiment 12 (Mini Project Phase-2) completed successfully.")
