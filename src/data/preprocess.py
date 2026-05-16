import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
import joblib

RAW_PATH       = os.path.join('data', 'raw', 'creditcard.csv')
PROCESSED_DIR  = os.path.join('data', 'processed')
SCALER_PATH    = os.path.join('models', 'scaler.pkl')
RANDOM_STATE   = 42
TEST_SIZE      = 0.20


def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded  : {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def scale_features(df: pd.DataFrame, fit: bool = True, scaler=None):
    """
    Scale Amount and Time with RobustScaler (resistant to outliers).
    V1-V28 are already PCA-scaled — leave them untouched.
    """
    df = df.copy()
    if fit:
        scaler = RobustScaler()
        df[['Amount', 'Time']] = scaler.fit_transform(df[['Amount', 'Time']])
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"Scaler  : fitted and saved to {SCALER_PATH}")
    else:
        df[['Amount', 'Time']] = scaler.transform(df[['Amount', 'Time']])
    return df, scaler


def split_data(df: pd.DataFrame):
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y          # preserves 0.17% fraud ratio in both splits
    )
    print(f"Train   : {X_train.shape[0]:,} rows  (fraud: {y_train.sum():,})")
    print(f"Test    : {X_test.shape[0]:,} rows  (fraud: {y_test.sum():,})")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series):
    """
    SMOTE applied ONLY to training data — never to test data.
    Creates synthetic fraud examples so the model sees a balanced dataset.
    """
    print(f"Before SMOTE — normal: {(y_train==0).sum():,}  fraud: {(y_train==1).sum():,}")
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"After  SMOTE — normal: {(y_res==0).sum():,}  fraud: {(y_res==1).sum():,}")
    return X_res, y_res


def save_splits(X_train, X_test, y_train, y_test, X_train_sm, y_train_sm):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    joblib.dump(X_train,    os.path.join(PROCESSED_DIR, 'X_train_raw.pkl'))
    joblib.dump(X_test,     os.path.join(PROCESSED_DIR, 'X_test.pkl'))
    joblib.dump(y_train,    os.path.join(PROCESSED_DIR, 'y_train_raw.pkl'))
    joblib.dump(y_test,     os.path.join(PROCESSED_DIR, 'y_test.pkl'))
    joblib.dump(X_train_sm, os.path.join(PROCESSED_DIR, 'X_train_smote.pkl'))
    joblib.dump(y_train_sm, os.path.join(PROCESSED_DIR, 'y_train_smote.pkl'))
    print(f"Saved   : all splits to {PROCESSED_DIR}/")


def run():
    print("=" * 45)
    print("  Preprocessing Pipeline")
    print("=" * 45)

    df = load_data()
    df, scaler = scale_features(df, fit=True)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)
    save_splits(X_train, X_test, y_train, y_test, X_train_sm, y_train_sm)

    print("=" * 45)
    print("  Done — preprocessing complete")
    print("=" * 45)
    return X_train, X_test, y_train, y_test, X_train_sm, y_train_sm, scaler


if __name__ == '__main__':
    run()
