"""
Experiment 1: Data preprocessing for training
------------------------------------------------
Aim:
    * Load a dataset (Iris dataset).
    * Perform data cleaning, handle missing values and normalize the data.
    * Split the dataset into training and test sets.

The original Iris dataset is already clean, so a few missing values, duplicate
rows and outliers are injected on purpose to demonstrate every cleaning step.
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Iris.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
TARGET = "Species"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Step 1: Load the dataset
# ---------------------------------------------------------------------------
section("STEP 1: LOAD THE DATASET")
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.head())
print("\nData types:\n", df.dtypes)
print("\nClass distribution:\n", df[TARGET].value_counts())

# ---------------------------------------------------------------------------
# Step 2: Introduce data quality problems (to simulate real-world raw data)
# ---------------------------------------------------------------------------
section("STEP 2: SIMULATE A RAW (DIRTY) DATASET")
rng = np.random.default_rng(42)
raw = df.copy()

# 5% missing values in each feature column
for col in FEATURES:
    idx = rng.choice(raw.index, size=int(0.05 * len(raw)), replace=False)
    raw.loc[idx, col] = np.nan

# 5 duplicate rows
raw = pd.concat([raw, raw.sample(5, random_state=1)], ignore_index=True)

# 3 extreme outliers
raw.loc[raw.sample(3, random_state=7).index, "SepalWidthCm"] = [15.0, 0.1, 12.5]

# Inconsistent text labels (extra spaces / different case)
raw.loc[raw.sample(4, random_state=3).index, TARGET] = " iris-setosa "
raw.to_csv(OUTPUT_DIR / "iris_raw_dirty.csv", index=False)
print("Dirty dataset shape:", raw.shape)

# ---------------------------------------------------------------------------
# Step 3: Data cleaning
# ---------------------------------------------------------------------------
section("STEP 3: DATA CLEANING")
clean = raw.copy()

# 3a. Drop the Id column - it carries no information for the model
clean = clean.drop(columns=["Id"])

# 3b. Fix inconsistent category labels
clean[TARGET] = clean[TARGET].str.strip().str.lower().str.replace("iris-", "Iris-")
print("Labels after fixing:", sorted(clean[TARGET].unique()))

# 3c. Remove duplicate rows
print("Duplicate rows found:", clean.duplicated().sum())
clean = clean.drop_duplicates().reset_index(drop=True)

# ---------------------------------------------------------------------------
# Step 4: Handle missing values
# ---------------------------------------------------------------------------
section("STEP 4: HANDLE MISSING VALUES")
print("Missing values before:\n", clean.isna().sum())
# Median is robust to outliers, so it is used to fill numeric gaps
imputer = SimpleImputer(strategy="median")
clean[FEATURES] = imputer.fit_transform(clean[FEATURES])
print("\nMissing values after:\n", clean.isna().sum())

# ---------------------------------------------------------------------------
# Step 5: Detect and treat outliers (IQR method)
# ---------------------------------------------------------------------------
section("STEP 5: OUTLIER DETECTION (IQR METHOD)")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
clean[FEATURES].boxplot(ax=axes[0])
axes[0].set_title("Before outlier treatment")

for col in FEATURES:
    q1, q3 = clean[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((clean[col] < low) | (clean[col] > high)).sum()
    print(f"{col:15s} -> range [{low:.2f}, {high:.2f}], outliers: {n_out}")
    # Capping (winsorization) keeps the row but limits the extreme value
    clean[col] = clean[col].clip(low, high)

clean[FEATURES].boxplot(ax=axes[1])
axes[1].set_title("After outlier treatment")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "outlier_boxplots.png")
plt.close()

# ---------------------------------------------------------------------------
# Step 6: Encode the target label
# ---------------------------------------------------------------------------
section("STEP 6: ENCODE TARGET LABEL")
encoder = LabelEncoder()
clean["SpeciesLabel"] = encoder.fit_transform(clean[TARGET])
for i, name in enumerate(encoder.classes_):
    print(f"{name} -> {i}")

# ---------------------------------------------------------------------------
# Step 7: Split into training and test sets (stratified 80/20)
# ---------------------------------------------------------------------------
section("STEP 7: TRAIN / TEST SPLIT")
X = clean[FEATURES]
y = clean["SpeciesLabel"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training set:", X_train.shape, " Test set:", X_test.shape)

# ---------------------------------------------------------------------------
# Step 8: Normalize the data
# The scaler is fitted ONLY on the training data to avoid data leakage.
# ---------------------------------------------------------------------------
section("STEP 8: NORMALIZATION / STANDARDIZATION")
minmax = MinMaxScaler()
X_train_mm = pd.DataFrame(minmax.fit_transform(X_train), columns=FEATURES)
X_test_mm = pd.DataFrame(minmax.transform(X_test), columns=FEATURES)
print("Min-Max scaled (train) - min:\n", X_train_mm.min().round(2).to_dict())
print("Min-Max scaled (train) - max:\n", X_train_mm.max().round(2).to_dict())

standard = StandardScaler()
X_train_std = pd.DataFrame(standard.fit_transform(X_train), columns=FEATURES)
X_test_std = pd.DataFrame(standard.transform(X_test), columns=FEATURES)
print("\nStandard scaled (train) - mean:\n", X_train_std.mean().round(2).to_dict())
print("Standard scaled (train) - std:\n", X_train_std.std().round(2).to_dict())

# ---------------------------------------------------------------------------
# Step 9: Save processed data and preprocessing objects
# ---------------------------------------------------------------------------
section("STEP 9: SAVE OUTPUTS")
clean.to_csv(OUTPUT_DIR / "iris_cleaned.csv", index=False)
train_df = X_train_std.assign(Species=y_train.values)
test_df = X_test_std.assign(Species=y_test.values)
train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)
joblib.dump(
    {"imputer": imputer, "scaler": standard, "label_encoder": encoder},
    OUTPUT_DIR / "preprocessors.joblib",
)
for f in sorted(OUTPUT_DIR.iterdir()):
    print("Saved:", f.name)

print("\nExperiment 1 completed successfully.")
