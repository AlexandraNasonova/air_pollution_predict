"""Adapter to save/read metrics, models to mlflow"""

# pylint: disable=E0401, R0913, W1514

from enum import Enum
import mlflow
import pandas as pd
import os
from mlflow.models.signature import infer_signature
from settings import env_reader


class MlFlowAdapter:
    class ModelType(Enum):
        PD_ARIMA = 1
        LGBM = 2
        TEMPORARY_FUSION_TRANSFORMER = 3

    def __init__(self, model_tp: ModelType):
        self._run = 0
        self._model_tp = model_tp

    def get_run(self):
        return self._run

    @staticmethod
    def set_experiment(experiment_name: str, mlflow_env_file: str):
        env_values = env_reader.get_params(mlflow_env_file)
        mlflow.set_tracking_uri(f'{env_values.get("MLFLOW_URI")}:{env_values.get("MLFLOW_PORT")}')
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name):
        self._run = mlflow.start_run()
        mlflow.set_tag("mlflow.runName", run_name)
        # mlflow.tracking.MlflowClient().set_tag(
        #     self._run.info.run_id, "mlflow.runName", run_name)

    @staticmethod
    def end_run():
        mlflow.end_run()

    def init_autolog(self):
        match self._model_tp:
            case self.ModelType.PD_ARIMA:
                return
            case self.ModelType.LGBM:
                mlflow.lightgbm.autolog()
                return
            case self.ModelType.TEMPORARY_FUSION_TRANSFORMER:
                mlflow.pytorch.autolog()
                return

    def __log_model(self, model, signature, artifact_path):
        match self._model_tp:
            case self.ModelType.PD_ARIMA:
                mlflow.pmdarima.log_model(pmdarima_model=model,
                                          signature=signature,
                                          artifact_path=artifact_path
                                          )
                return
            case self.ModelType.LGBM:
                mlflow.lightgbm.log_model(lgb_model=model,
                                          signature=signature,
                                          artifact_path=artifact_path)
                return
            case self.ModelType.TEMPORARY_FUSION_TRANSFORMER:
                mlflow.pytorch.log_model(pytorch_model=model,
                                         signature=signature,
                                         artifact_path=artifact_path
                                         )
                return

    def save_metrics_to_last_run(self, metric_name, train_score_all, val_score_all, run_id=0):
        run_id = mlflow.last_active_run().info.run_id if run_id == 0 else run_id
        with mlflow.start_run(run_id=run_id):
            self.save_metrics(metric_name, train_score_all, val_score_all)

    @staticmethod
    def save_metrics(metric_name, train_score, val_score):
        mlflow.log_metric(f'train_{metric_name}_all', train_score)
        mlflow.log_metric(f'val_{metric_name}_all', val_score)

    @staticmethod
    def set_run_name_to_last_run(run_name, run_id=0):
        run_id = mlflow.last_active_run().info.run_id if run_id == 0 else run_id
        mlflow.tracking.MlflowClient().set_tag(run_id, "mlflow.runName", run_name)

    @staticmethod
    def save_shap(run_id, model_predict, x_train: pd.DataFrame):
        run_id = mlflow.last_active_run().info.run_id if run_id == 0 else run_id
        with mlflow.start_run(run_id=run_id):
            mlflow.shap.log_explanation(model_predict, x_train)

    def save_model(self,
                   x_train_df: pd.DataFrame,
                   y_train_df: pd.DataFrame,
                   model,
                   artifact_path: str,
                   best_model_params: dict,
                   ):
        """
        Send model and metrics to mlflow
        @param x_train_df: X_train dataframe
        @param y_train_df: y_train dataframe
        @param model: Trained model
        @param artifact_path: The path to artifact
        @param best_model_params: Parameters of the best model
        """
        if x_train_df:
            signature = infer_signature(x_train_df, y_train_df)
        else:
            signature = infer_signature(y_train_df, y_train_df)
        self.__log_model(model=model, signature=signature, artifact_path=artifact_path)
        self.save_params(best_model_params)

    @staticmethod
    def save_params(params: dict):
        mlflow.log_params(params)

    @staticmethod
    def save_tag(key: str, value):
        mlflow.set_tag(key, value)

    def save_extra_params_to_last_run(self, params: dict, artifact_file: str,
                                      run_id=0):
        run_id = mlflow.last_active_run().info.run_id if run_id == 0 else run_id
        with mlflow.start_run(run_id=run_id):
            self.save_extra_params(params, artifact_file)

    @staticmethod
    def save_extra_params(params: dict, artifact_file: str):
        mlflow.log_dict(params, artifact_file)

    def save_artifact_to_last_run(self, source_file: str, artifact_path: str,
                                  run_id=0):
        run_id = mlflow.last_active_run().info.run_id if run_id == 0 else run_id
        with mlflow.start_run(run_id=run_id):
            self.save_artifact(source_file, artifact_path)

    @staticmethod
    def save_artifact(source_file: str, artifact_path: str):
        mlflow.log_artifact(source_file, artifact_path)

    def save_artifact_folder_to_last_run(self, source_folder: str,
                                         artifact_path: str,
                                         run_id=0):
        run_id = mlflow.last_active_run().info.run_id if run_id == 0 else run_id
        with mlflow.start_run(run_id=run_id):
            self.save_artifact_folder(source_folder, artifact_path)

    @staticmethod
    def save_artifact_folder(source_folder: str, artifact_path: str):
        mlflow.log_artifacts(source_folder, artifact_path)

    def save_dataframe_to_last_run(self, df: pd.DataFrame, artifact_path: str,
                                   run_id=0):
        run_id = mlflow.last_active_run().info.run_id if run_id == 0 else run_id
        with mlflow.start_run(run_id=run_id):
            self.save_dataframe(df, artifact_path)

    @staticmethod
    def save_dataframe(df: pd.DataFrame, artifact_path: str):
        mlflow.log_table(df, artifact_path)
