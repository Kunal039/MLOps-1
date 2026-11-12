"""Data loading and data validation (first stage of the CI pipeline)."""

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

EXPECTED_FEATURES = 13
EXPECTED_CLASSES = {0, 1, 2}


def load_data():
    wine = load_wine(as_frame=True)
    df = wine.frame
    return df.drop(columns=["target"]), df["target"]


def validate_data(X: pd.DataFrame, y: pd.Series):
    """Raise an error if the data does not look like the data the model was designed for."""
    assert X.shape[1] == EXPECTED_FEATURES, f"Expected {EXPECTED_FEATURES} features, got {X.shape[1]}"
    assert not X.isna().any().any(), "Data contains missing values"
    assert set(y.unique()) == EXPECTED_CLASSES, "Unexpected target classes"
    assert (X >= 0).all().all(), "Negative values found in chemical measurements"
    assert len(X) >= 100, "Not enough rows to train"
    return True


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
