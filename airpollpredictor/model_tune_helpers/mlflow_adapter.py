"""Adapter to save/read metrics, models to mlflow"""

# pylint: disable=E0401, R0913, W1514

from enum import Enum
import mlflow
from settings import env_reader
# from mlflow.models.signature import infer_signature


class MlFlowAdapter:
    class ModelType(Enum):
        PD_ARIMA = 1
        LGBM = 2
        TEMPORARY_FUSION_TRANSFORMER = 3

    def __init__(self, model_tp: ModelType):
        self.model_tp = model_tp

    @staticmethod
    def set_experiment(experiment_name: str, mlflow_env_file: str):
        env_values = env_reader.get_params(mlflow_env_file)
        mlflow.set_tracking_uri(f'{env_values.get("MLFLOW_URI")}:{env_values.get("MLFLOW_PORT")}')
        mlflow.set_experiment(experiment_name)

    def init_autolog(self):
        match self.model_tp:
            case self.ModelType.PD_ARIMA:
                return
            case self.ModelType.LGBM:
                mlflow.lightgbm.autolog()
                return
            case self.ModelType.TEMPORARY_FUSION_TRANSFORMER:
                mlflow.pytorch.autolog(log_every_n_epoch=10, silent=True)

    @staticmethod
    def save_metrics(metric_name, train_score_all, val_score_all, run_id=0):
        run_id = mlflow.last_active_run().info.run_id if run_id == 0 else run_id
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric(f'train_{metric_name}_all', train_score_all)
            mlflow.log_metric(f'val_{metric_name}_all', val_score_all)

    @staticmethod
    def set_run_name(run_name, run_id=0):
        run_id = mlflow.last_active_run().info.run_id if run_id == 0 else run_id
        mlflow.tracking.MlflowClient().set_tag(run_id, "mlflow.runName", run_name)


# def send_model_to_ml_flow(x_train_df: pd.DataFrame, y_train_df: pd.DataFrame, model,
#                           train_score: float, val_score: float,
#                           artifact_path: str,
#                           metric_name: str, model_params_settings: dict,
#                           optuna_params_settings: dict):
#     """
#     Send model and metrics to mlflow
#     @param x_train_df: X_train dataframe
#     @param y_train_df: y_train dataframe
#     @param model: Trained model
#     @param train_score: Train score
#     @param val_score: Validation score
#     @param artifact_path: The path to artifact
#     @param metric_name: The name of the used metric
#     @param model_params_settings: Parameters used to train the model
#     @param optuna_params_settings: Parameters used to run optuna
#     """
#     signature = infer_signature(x_train_df, y_train_df)
#     with mlflow.start_run() as _:
#         log_model(lgb_model=model, signature=signature, artifact_path=artifact_path)
#         mlflow.log_metric(f'train_{metric_name}', train_score)
#         mlflow.log_metric(f'val_{metric_name}', val_score)
#         params = {"model_params": model_params_settings, "optuna_params": optuna_params_settings}
#         mlflow.log_params(params)
