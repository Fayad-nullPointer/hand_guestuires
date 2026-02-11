# MLflow Utility Functions for Hand Landmark Recognition
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV


def setup_mlflow(experiment_name="Hand-Landmark-Recognition"):
    """Set up MLflow experiment."""
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)


def track_model(model, model_name, X_train, y_train, X_val, y_val,
                experiment_name="Hand-Landmark-Recognition"):
    """
    Track a single model: log params, metrics, and model artifact.

    Args:
        model: sklearn estimator (fitted or unfitted).
        model_name: Name for the MLflow run.
        X_train, y_train: Training data.
        X_val, y_val: Validation data.

    Returns:
        run_id (str)
    """
    setup_mlflow(experiment_name)

    with mlflow.start_run(run_name=model_name) as run:
        # Fit if not already fitted
        try:
            model.predict(X_val[:1])
        except Exception:
            model.fit(X_train, y_train)

        # Log parameters
        mlflow.log_param("model_type", type(model).__name__)
        for k, v in model.get_params().items():
            mlflow.log_param(k, str(v)[:250])

        # Predict & log metrics
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)

        # Log model artifact
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.set_tag("model_name", model_name)

        print(f"[MLflow] '{model_name}' | accuracy={acc:.4f} | f1={f1:.4f}")
        return run.info.run_id


def track_grid_search(estimator, param_grid, X_train, y_train, X_val, y_val,
                      model_name="GridSearch", cv=5,
                      experiment_name="Hand-Landmark-Recognition"):
    """
    Run GridSearchCV, log best params, metrics, and model artifact.

    Returns:
        (best_estimator, run_id)
    """
    grid = GridSearchCV(estimator, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    setup_mlflow(experiment_name)

    with mlflow.start_run(run_name=model_name) as run:
        # Log best params
        for k, v in grid.best_params_.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("best_cv_score", grid.best_score_)

        # Evaluate on validation
        y_pred = grid.best_estimator_.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="weighted")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)

        # Log model
        mlflow.sklearn.log_model(grid.best_estimator_, artifact_path="model")
        mlflow.set_tag("model_name", model_name)

        print(f"[MLflow] '{model_name}' best_params={grid.best_params_} | accuracy={acc:.4f}")
        return grid.best_estimator_, run.info.run_id
def add_artifact(run_id, artifact_path, local_path):
    """Add an additional artifact (e.g., confusion matrix) to an existing run."""
    mlflow.log_artifact(local_path, artifact_path=artifact_path, run_id=run_id)
    print(f"[MLflow] Added artifact '{local_path}' to run {run_id} under '{artifact_path}'")


def load_best_model(experiment_name="Hand-Landmark-Recognition", metric="f1_weighted"):
    """
    Load the model with the highest metric from the experiment.

    Returns:
        loaded sklearn model
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    best = runs.iloc[0]
    model = mlflow.sklearn.load_model(f"runs:/{best['run_id']}/model")
    print(f"[MLflow] Loaded '{best['tags.model_name']}' | {metric}={best[f'metrics.{metric}']:.4f}")
    return model


def register_model(run_id, model_name="HandLandmarkModel"):
    """Register a logged model to the MLflow Model Registry."""
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)
    print(f"[MLflow] Registered '{model_name}' version {result.version}")
    return result