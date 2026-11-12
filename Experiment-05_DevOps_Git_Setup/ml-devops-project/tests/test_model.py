"""Automated tests run by the CI pipeline before a model can be merged / deployed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import load_data, train  # noqa: E402


def test_data_has_expected_shape():
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[1] == 4
    assert len(X_train) + len(X_test) == 150
    assert not X_train.isna().any().any()


def test_model_meets_accuracy_threshold():
    _, metrics = train()
    assert metrics["accuracy"] >= 0.90, "Model quality gate failed"


def test_model_predicts_valid_classes():
    model, _ = train()
    _, X_test, _, _ = load_data()
    assert set(model.predict(X_test)).issubset({0, 1, 2})
