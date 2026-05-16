import os
import pytest
import joblib
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join('data', 'processed')
MODELS_DIR    = os.path.join('models')


def test_processed_files_exist():
    """All 6 split files must exist after preprocess.py runs."""
    files = [
        'X_train_raw.pkl', 'y_train_raw.pkl',
        'X_train_smote.pkl', 'y_train_smote.pkl',
        'X_test.pkl', 'y_test.pkl',
    ]
    for f in files:
        path = os.path.join(PROCESSED_DIR, f)
        assert os.path.exists(path), f"Missing processed file: {f}"


def test_scaler_exists():
    """scaler.pkl must exist — API needs it at inference time."""
    assert os.path.exists(os.path.join(MODELS_DIR, 'scaler.pkl'))


def test_train_test_shapes():
    """Train and test feature sets must have the same number of columns."""
    X_train = joblib.load(os.path.join(PROCESSED_DIR, 'X_train_smote.pkl'))
    X_test  = joblib.load(os.path.join(PROCESSED_DIR, 'X_test.pkl'))
    assert X_train.shape[1] == X_test.shape[1], "Column count mismatch between train and test"
    assert X_train.shape[1] == 30, f"Expected 30 features, got {X_train.shape[1]}"


def test_smote_balanced():
    """After SMOTE, fraud and normal classes must be equal in the training set."""
    y_train = joblib.load(os.path.join(PROCESSED_DIR, 'y_train_smote.pkl'))
    counts  = y_train.value_counts()
    assert counts[0] == counts[1], "SMOTE did not balance the classes equally"


def test_test_set_untouched():
    """Test labels must still be imbalanced — SMOTE should never touch the test set."""
    y_test = joblib.load(os.path.join(PROCESSED_DIR, 'y_test.pkl'))
    fraud_ratio = y_test.mean()
    assert fraud_ratio < 0.01, f"Test set fraud ratio {fraud_ratio:.4f} is too high — SMOTE may have leaked"


def test_no_missing_values():
    """No NaN values in any split."""
    for fname in ['X_train_smote.pkl', 'X_test.pkl']:
        df = joblib.load(os.path.join(PROCESSED_DIR, fname))
        assert not df.isnull().any().any(), f"NaN values found in {fname}"


def test_scaler_transforms_correctly():
    """Scaler must transform Amount and Time without errors and change their values."""
    scaler   = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    X_test   = joblib.load(os.path.join(PROCESSED_DIR, 'X_test.pkl'))
    original = X_test[['Amount', 'Time']].values.copy()
    scaled   = scaler.transform(X_test[['Amount', 'Time']])
    assert scaled.shape == original.shape
    assert not np.allclose(scaled, original), "Scaler did not change the values"
