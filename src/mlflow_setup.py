import os
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

EXPERIMENT_NAME = "fraud-detection-benchmark"
# SQLite backend — recommended over filesystem tracking (deprecated Feb 2026)
TRACKING_URI    = "sqlite:///mlruns/mlflow.db"


def init_mlflow():
    """Set tracking URI and create/get the experiment. Call once before training."""
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
        print(f"Created experiment : '{EXPERIMENT_NAME}'")
    else:
        print(f"Using experiment   : '{EXPERIMENT_NAME}' (id={experiment.experiment_id})")
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_model_run(
    model_name: str,
    model,
    params: dict,
    metrics: dict,
    X_sample=None,
):
    """
    Log a single model training run to MLflow.

    Parameters
    ----------
    model_name : str   — display name shown in the MLflow UI
    model      : fitted sklearn-compatible estimator
    params     : dict  — hyperparameters to log (e.g. {'n_estimators': 100})
    metrics    : dict  — evaluation scores  (e.g. {'auc_roc': 0.98, 'f1': 0.87})
    X_sample   : optional small DataFrame/array used to infer the model signature
    """
    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("model_name", model_name)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_sample, model.predict(X_sample)) if X_sample is not None else None
        mlflow.sklearn.log_model(model, name="model", signature=signature)

        run_id = mlflow.active_run().info.run_id
        print(
            f"  [{model_name:<25}] "
            f"AUC={metrics.get('auc_roc', 0):.4f}  "
            f"F1={metrics.get('f1', 0):.4f}  "
            f"Recall={metrics.get('recall', 0):.4f}  "
            f"run_id={run_id[:8]}"
        )
        return run_id


if __name__ == '__main__':
    init_mlflow()
    print(f"\nMLflow UI  : run  'mlflow ui'  then open  http://localhost:5000")
    print(f"Tracking   : {TRACKING_URI}")
