import os
import sys
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.mlflow_setup import init_mlflow, log_model_run

PROCESSED_DIR = os.path.join('data', 'processed')
MODELS_DIR    = os.path.join('models')

PARAM_GRID = {
    'n_estimators' : [100, 200, 300],
    'max_depth'    : [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2],
}


def load_data():
    X_train = joblib.load(os.path.join(PROCESSED_DIR, 'X_train_smote.pkl'))
    y_train = joblib.load(os.path.join(PROCESSED_DIR, 'y_train_smote.pkl'))
    X_test  = joblib.load(os.path.join(PROCESSED_DIR, 'X_test.pkl'))
    y_test  = joblib.load(os.path.join(PROCESSED_DIR, 'y_test.pkl'))
    return X_train, y_train, X_test, y_test


def tune():
    print("=" * 55)
    print("  XGBoost Hyperparameter Tuning (GridSearchCV)")
    print("=" * 55)

    X_train, y_train, X_test, y_test = load_data()

    base = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
    )

    print(f"\n  Grid size : {3*3*3} combinations  (3 x 3 x 3)")
    print("  Searching — this may take a few minutes ...\n")

    grid = GridSearchCV(
        base,
        PARAM_GRID,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    best_model  = grid.best_estimator_

    print(f"\n  Best params : {best_params}")
    print(f"  CV AUC-ROC  : {grid.best_score_:.4f}")

    # evaluate on held-out test set
    y_pred  = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    metrics = {
        "auc_roc"   : round(roc_auc_score(y_test, y_proba), 6),
        "f1"        : round(f1_score(y_test, y_pred), 6),
        "precision" : round(precision_score(y_test, y_pred), 6),
        "recall"    : round(recall_score(y_test, y_pred), 6),
        "accuracy"  : round(accuracy_score(y_test, y_pred), 6),
    }

    print(f"\n  Test AUC-ROC : {metrics['auc_roc']}")
    print(f"  Test Recall  : {metrics['recall']}  (% of fraud caught)")
    print(f"  Test F1      : {metrics['f1']}")

    # log tuned run to MLflow
    init_mlflow()
    log_model_run(
        model_name="XGBoost (Tuned)",
        model=best_model,
        params=best_params,
        metrics=metrics,
        X_sample=X_test.iloc[:5],
    )

    # save final model
    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, 'xgboost_tuned.pkl')  # final model served by the API
    joblib.dump(best_model, out_path)
    print(f"\n  Saved : {out_path}")
    print("=" * 55)

    return best_model, best_params, metrics


if __name__ == '__main__':
    tune()
