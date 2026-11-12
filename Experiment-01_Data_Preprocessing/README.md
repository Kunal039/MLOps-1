# Experiment 1 – Data Preprocessing for Training

**Unit 1 · CO2** | Syllabus: *Load a dataset (Iris), perform data cleaning, handle missing values, normalize the data, split into training and test sets.*

## Aim
To prepare a raw dataset for machine learning by cleaning it, handling missing values and outliers, encoding labels, normalizing features and splitting it into training and test sets.

## Theory
- **Data cleaning:** fix inconsistent labels, remove duplicates, drop identifier columns (`Id`).
- **Missing values:** fill numeric gaps with the **median**, because it is robust to outliers. `SimpleImputer` does this.
- **Outliers (IQR rule):** values outside `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]` are capped (winsorization).
- **Normalization:** Min-Max scaling maps values to [0, 1]. Standardization gives mean 0 and std 1. Fit the scaler **only on training data** to avoid *data leakage*.
- **Train/test split:** a stratified 80/20 split keeps the class ratio the same in both sets.

## Files
| File | Purpose |
|---|---|
| `data_preprocessing.py` | Full preprocessing workflow (9 steps) |
| `data/Iris.csv` | Iris dataset (150 rows, 4 features, 3 classes) |
| `outputs/` | Dirty and cleaned data, `train.csv`/`test.csv`, boxplots, saved preprocessors |

The Iris dataset is already clean, so the script **injects** 5% missing values, duplicates, outliers and messy labels to demonstrate each step.

## Run
```bash
python data_preprocessing.py
```

## Result
Missing values go from 7 per column to 0, 5 duplicates are removed and outliers are capped. Data is split into 120 training and 30 test rows. Min-Max scaled features lie in [0, 1]; standardized features have mean 0 and std 1.
