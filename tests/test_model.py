import os
import pytest
import joblib
import numpy as np
import pandas as pd

MODELS_DIR    = os.path.join('models')
PROCESSED_DIR = os.path.join('data', 'processed')

FINAL_MODEL = os.path.join(MODELS_DIR, 'xgboost_tuned.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

ALL_MODELS = [
    'logistic_regression.pkl',
    'decision_tree.pkl',
    'random_forest.pkl',
    'extra_trees.pkl',
    'adaboost.pkl',
    'gradient_boosting.pkl',
    'xgboost.pkl',
    'lightgbm.pkl',
    'catboost.pkl',
    'xgboost_tuned.pkl',
    'catboost_final.pkl',
]


def test_final_model_exists():
    """The model served by the API must exist."""
    assert os.path.exists(FINAL_MODEL), f"Final model not found: {FINAL_MODEL}"


def test_all_models_loadable():
    """Every trained model must load without errors."""
    for fname in ALL_MODELS:
        path = os.path.join(MODELS_DIR, fname)
        assert os.path.exists(path), f"Missing model file: {fname}"
        model = joblib.load(path)
        assert model is not None


def test_final_model_has_predict_proba():
    """Final model must support probability output — required by the API."""
    model = joblib.load(FINAL_MODEL)
    assert hasattr(model, 'predict_proba'), "Model must have predict_proba method"


def test_prediction_shape():
    """Model must return one prediction per input row."""
    model  = joblib.load(FINAL_MODEL)
    scaler = joblib.load(SCALER_PATH)
    X_test = joblib.load(os.path.join(PROCESSED_DIR, 'X_test.pkl'))

    sample = X_test.iloc[:10].copy()
    sample[['Amount', 'Time']] = scaler.transform(sample[['Amount', 'Time']])

    preds   = model.predict(sample)
    probas  = model.predict_proba(sample)

    assert len(preds)        == 10
    assert probas.shape      == (10, 2)


def test_prediction_values_are_binary():
    """Predictions must only be 0 or 1."""
    model  = joblib.load(FINAL_MODEL)
    scaler = joblib.load(SCALER_PATH)
    X_test = joblib.load(os.path.join(PROCESSED_DIR, 'X_test.pkl'))

    sample = X_test.iloc[:50].copy()
    sample[['Amount', 'Time']] = scaler.transform(sample[['Amount', 'Time']])

    preds = model.predict(sample)
    assert set(preds).issubset({0, 1}), f"Unexpected prediction values: {set(preds)}"


def test_probabilities_between_0_and_1():
    """Fraud probabilities must always be in [0, 1]."""
    model  = joblib.load(FINAL_MODEL)
    scaler = joblib.load(SCALER_PATH)
    X_test = joblib.load(os.path.join(PROCESSED_DIR, 'X_test.pkl'))

    sample = X_test.iloc[:50].copy()
    sample[['Amount', 'Time']] = scaler.transform(sample[['Amount', 'Time']])

    probas = model.predict_proba(sample)[:, 1]
    assert (probas >= 0).all() and (probas <= 1).all(), "Probabilities out of [0,1] range"


def test_model_catches_some_fraud():
    """Model must detect at least some fraud on the real test set (sanity check)."""
    model  = joblib.load(FINAL_MODEL)
    scaler = joblib.load(SCALER_PATH)
    X_test = joblib.load(os.path.join(PROCESSED_DIR, 'X_test.pkl'))
    y_test = joblib.load(os.path.join(PROCESSED_DIR, 'y_test.pkl'))

    X = X_test.copy()
    X[['Amount', 'Time']] = scaler.transform(X[['Amount', 'Time']])

    preds       = model.predict(X)
    fraud_caught = ((preds == 1) & (y_test == 1)).sum()
    assert fraud_caught > 0, "Model caught zero fraud cases — something is wrong"
