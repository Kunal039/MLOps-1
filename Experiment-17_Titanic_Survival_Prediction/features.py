"""Titanic feature engineering, shared by training and the web app (importable for unpickling)."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

RAW_COLUMNS = ["Pclass", "Name", "Gender", "Age", "SibSp", "Parch", "Fare", "Embarked"]


class TitanicFeatures(BaseEstimator, TransformerMixin):
    """Cleans raw passenger records and creates model features.
    Medians/modes are learnt in fit() from the training data only (no leakage)."""

    def fit(self, X, y=None):
        X = self._titles(X)
        self.age_by_title_ = X.groupby("Title")["Age"].median().to_dict()
        self.age_median_ = X["Age"].median()
        self.fare_median_ = X["Fare"].median()
        self.embarked_mode_ = X["Embarked"].mode()[0]
        return self

    @staticmethod
    def _titles(X):
        X = X.copy()
        title = X["Name"].fillna("").str.extract(r",\s*([^\.]+)\.")[0]
        title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
        X["Title"] = title.where(title.isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
        return X

    def transform(self, X):
        X = self._titles(X)
        X["Age"] = X["Age"].fillna(X["Title"].map(self.age_by_title_)).fillna(self.age_median_)
        X["Fare"] = X["Fare"].fillna(self.fare_median_)
        X["Embarked"] = X["Embarked"].fillna(self.embarked_mode_)
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
        X["IsFemale"] = (X["Gender"].str.lower() == "female").astype(int)
        X["IsChild"] = (X["Age"] < 13).astype(int)
        out = X[["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "IsFemale", "IsChild"]].copy()
        for title in ["Mr", "Mrs", "Miss", "Master", "Rare"]:
            out[f"Title_{title}"] = (X["Title"] == title).astype(int)
        for port in ["C", "Q", "S"]:
            out[f"Embarked_{port}"] = (X["Embarked"] == port).astype(int)
        return out


def passenger_frame(passenger):
    """Convert a dict from the web form / API into a one-row DataFrame with the raw columns."""
    return pd.DataFrame([{
        "Pclass": int(passenger["Pclass"]),
        "Name": passenger.get("Name") or f"Doe, {passenger.get('Title', 'Mr')}. John",
        "Gender": passenger["Gender"],
        "Age": float(passenger["Age"]) if passenger.get("Age") not in (None, "") else None,
        "SibSp": int(passenger.get("SibSp", 0)),
        "Parch": int(passenger.get("Parch", 0)),
        "Fare": float(passenger.get("Fare", 15)),
        "Embarked": passenger.get("Embarked", "S"),
    }])
