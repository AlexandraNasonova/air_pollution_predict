"""Adapter to save/read metrics, models to mlflow"""

# pylint: disable=E0401, R0913, W1514
import pandas as pd
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.lightgbm import log_model




def send_model_to_ml_flow(x_train_df: pd.DataFrame, y_train_df: pd.DataFrame, model,
                          train_score: float, val_score: float,
                          artifact_path: str,
                          metric_name: str, model_params_settings: dict,
                          optuna_params_settings: dict):
    """
    Send model and metrics to mlflow
    @param x_train_df: X_train dataframe
    @param y_train_df: y_train dataframe
    @param model: Trained model
    @param train_score: Train score
    @param val_score: Validation score
    @param artifact_path: The path to artifact
    @param metric_name: The name of the used metric
    @param model_params_settings: Parameters used to train the model
    @param optuna_params_settings: Parameters used to run optuna
    """
    signature = infer_signature(x_train_df, y_train_df)
    with mlflow.start_run() as _:
        log_model(lgb_model=model, signature=signature, artifact_path=artifact_path)
        mlflow.log_metric(f'train_{metric_name}', train_score)
        mlflow.log_metric(f'val_{metric_name}', val_score)
        params = {"model_params": model_params_settings, "optuna_params": optuna_params_settings}
        mlflow.log_params(params)
