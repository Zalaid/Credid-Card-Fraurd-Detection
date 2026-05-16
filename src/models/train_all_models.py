import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, GradientBoostingClassifier,
)
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.mlflow_setup import init_mlflow, log_model_run

warnings.filterwarnings('ignore')

PROCESSED_DIR = os.path.join('data', 'processed')
MODELS_DIR    = os.path.join('models')


# ---------------------------------------------------------------------------
# Model definitions — (name, estimator, params_to_log)
# ---------------------------------------------------------------------------
MODELS = [
    (
        "Logistic Regression",
        LogisticRegression(max_iter=1000, random_state=42),
        {"max_iter": 1000, "solver": "lbfgs"},
    ),
    (
        "Decision Tree",
        DecisionTreeClassifier(max_depth=10, random_state=42),
        {"max_depth": 10},
    ),
    (
        "Random Forest",
        RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        {"n_estimators": 100},
    ),
    (
        "Extra Trees",
        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        {"n_estimators": 100},
    ),
    (
        "AdaBoost",
        AdaBoostClassifier(n_estimators=100, random_state=42),
        {"n_estimators": 100},
    ),
    (
        "Gradient Boosting",
        GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
        {"n_estimators": 100, "max_depth": 4},
    ),
    (
        "XGBoost",
        XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=1, eval_metric='logloss',
            random_state=42, n_jobs=-1,
        ),
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
    ),
    (
        "LightGBM",
        LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbose=-1,
        ),
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
    ),
    (
        "CatBoost",
        CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.1,
            random_seed=42, verbose=0,
        ),
        {"iterations": 200, "depth": 6, "learning_rate": 0.1},
    ),
]


def load_data():
    X_train = joblib.load(os.path.join(PROCESSED_DIR, 'X_train_smote.pkl'))
    y_train = joblib.load(os.path.join(PROCESSED_DIR, 'y_train_smote.pkl'))
    X_test  = joblib.load(os.path.join(PROCESSED_DIR, 'X_test.pkl'))
    y_test  = joblib.load(os.path.join(PROCESSED_DIR, 'y_test.pkl'))
    return X_train, y_train, X_test, y_test


def evaluate(model, X_test, y_test):
    y_pred      = model.predict(X_test)
    y_proba     = model.predict_proba(X_test)[:, 1]
    return {
        "auc_roc"   : round(roc_auc_score(y_test, y_proba), 6),
        "f1"        : round(f1_score(y_test, y_pred), 6),
        "precision" : round(precision_score(y_test, y_pred), 6),
        "recall"    : round(recall_score(y_test, y_pred), 6),
        "accuracy"  : round(accuracy_score(y_test, y_pred), 6),
    }


def run():
    print("=" * 65)
    print("  Training & Benchmarking 9 Models")
    print("=" * 65)

    init_mlflow()
    X_train, y_train, X_test, y_test = load_data()

    os.makedirs(MODELS_DIR, exist_ok=True)
    results = []

    for name, model, params in MODELS:
        print(f"\n  Training : {name}")
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)

        # log to MLflow
        log_model_run(
            model_name=name,
            model=model,
            params=params,
            metrics=metrics,
            X_sample=X_test.iloc[:5],
        )

        # save model locally
        safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        joblib.dump(model, os.path.join(MODELS_DIR, f'{safe_name}.pkl'))

        results.append({"Model": name, **metrics})

    # -----------------------------------------------------------------------
    # Print comparison table ranked by AUC-ROC
    # -----------------------------------------------------------------------
    df = pd.DataFrame(results).sort_values("auc_roc", ascending=False).reset_index(drop=True)
    df.index += 1

    print("\n")
    print("=" * 65)
    print("  Model Benchmark Results (ranked by AUC-ROC)")
    print("=" * 65)
    print(df.to_string())
    print("=" * 65)
    print(f"\n  Best model : {df.iloc[0]['Model']}  (AUC-ROC = {df.iloc[0]['auc_roc']})")
    print("=" * 65)

    return df


if __name__ == '__main__':
    run()
