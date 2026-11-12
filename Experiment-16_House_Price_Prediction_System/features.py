"""Feature engineering step shared by train.py and predict.py (must be importable to load the saved pipeline)."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds engineered features. Part of the pipeline, so predict.py gets them automatically."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["RoomsPerPerson"] = X["AveRooms"] / X["AveOccup"]
        X["BedroomRatio"] = X["AveBedrms"] / X["AveRooms"]
        X["LogPopulation"] = np.log1p(X["Population"])
        X["LogIncome"] = np.log(X["MedInc"])
        # distance to the two big cities (Los Angeles, San Francisco) - location drives price
        X["DistLA"] = np.hypot(X["Latitude"] - 34.05, X["Longitude"] + 118.24)
        X["DistSF"] = np.hypot(X["Latitude"] - 37.77, X["Longitude"] + 122.42)
        return X
