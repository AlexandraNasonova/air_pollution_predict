# pylint: disable=E0401

from argparse import ArgumentParser
import pandas as pd
import json
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.lightgbm import log_model
from sklearn.model_selection import TimeSeriesSplit
import yaml
from settings import settings
import lgbm_tuner.columns_filter as col_filter
from ml_tune_helpers.lgbm_optuna.optuna_lgb_search import OptunaLgbSearch
import warnings
warnings.filterwarnings('ignore')


STAGE = "tune_lgbm_model"


def __parse_args():
    parser = ArgumentParser(STAGE)
    parser.add_argument('--input_train_file', required=True, help='Path to input train data')
    parser.add_argument('--input_val_file', required=True, help='Path to input validation data')
    parser.add_argument('--output_metrics_files', required=True, help='Path to metrics file')
    parser.add_argument('--params', required=True, help='Path to params')
    parser.add_argument('--params_section', required=True, help='Section with filter params')
    parser.add_argument('--mlflow_artifact', required=True, help='MLFlow model artifacts path')
    return parser.parse_args()


def __extract_labels(df_ts: pd.DataFrame, target_column: str):
    x_ts = df_ts.drop([target_column], axis=1)
    y_ts = df_ts[target_column]
    return x_ts, y_ts


def __save_metrics(metrics_output_file_path: str, train_score: float, val_score: float,
                   metric_name: str):
    with open(metrics_output_file_path, 'w') as f_stream:
        json.dump({
            f'train_{metric_name}': train_score,
            f'val_{metric_name}': val_score,
        }, f_stream)


def __send_model_to_ml_flow(x_train_df: pd.DataFrame, y_train_df: pd.DataFrame, model,
                            train_score: float, val_score: float,
                            artifact_path: str,
                            metric_name: str, model_params_settings: dict,
                            optuna_params_settings: dict):
    signature = infer_signature(x_train_df, y_train_df)
    with mlflow.start_run() as run:
        log_model(lgb_model=model, signature=signature, artifact_path=artifact_path)
        mlflow.log_metric(f'train_{metric_name}', train_score)
        mlflow.log_metric(f'val_{metric_name}', val_score)
        params = {"model_params": model_params_settings, "optuna_params": optuna_params_settings}
        mlflow.log_params(params)


if __name__ == '__main__':
    stage_args = __parse_args()
    with open(stage_args.params, 'r', encoding='UTF-8') as file_stream:
        yaml_params = yaml.safe_load(file_stream)

    model_params = yaml_params[stage_args.params_section]
    metric = yaml_params["metric"]
    optuna_params = yaml_params["optuna"]
    pol_id = model_params["pol_id"]
    target_column_name = col_filter.get_target_column(
        prediction_value_type=model_params['prediction_value_type'],
        pol_id=pol_id)

    df_train = pd.read_csv(stage_args.input_train_file, parse_dates=True,
                           index_col=settings.DATE_COLUMN_NAME)
    df_val = pd.read_csv(stage_args.input_val_file, parse_dates=True,
                         index_col=settings.DATE_COLUMN_NAME)
    x_train, y_train = __extract_labels(df_train, target_column_name)
    x_val, y_val = __extract_labels(df_val, target_column_name)

    optuna_tuner = OptunaLgbSearch(
        study_name=f'lgbm_{pol_id if pol_id > 0 else "all"}',
        metric=metric,
        objective=optuna_params["objective"],
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        default_params=model_params["default_params"],
        default_category=model_params["default_category"],
        categories_for_optimization=model_params["categories"],
        default_top_features_count=model_params["default_top_features_count"])

    optuna_tuner.run_params_search(
        n_trials=optuna_params["n_trials"],
        n_jobs=optuna_params["n_jobs"],
        save_best_params=True,
        direction=optuna_params["optimization_direction"],
        best_features_only=True,
        search_category=model_params["search_category"],
        with_pruner=True,
        cv_splitter=TimeSeriesSplit(optuna_params["cv_folders"]),
        warm_params=None)

    print(f'----Optuna finished params tuning---')

    train_score_best, val_score_best, model_best = optuna_tuner.run_model_and_eval(
        params=optuna_tuner.study_best_params,
        categorical_features=optuna_tuner.study_best_params['categorical_features'],
        best_features_only=True,
        set_as_best_model=False)

    print(f'---Model trained with best params: '
          f'best_train_score: {train_score_best}, best_val_score: {val_score_best}')

    __save_metrics(metrics_output_file_path=stage_args.output_metrics_files,
                   train_score=train_score_best,
                   val_score=val_score_best,
                   metric_name=metric)

    print(f'---Metrics saved locally---')

    __send_model_to_ml_flow(x_train_df=x_train,
                            y_train_df=y_train,
                            train_score=train_score_best,
                            val_score=val_score_best,
                            artifact_path=stage_args.mlflow_artifact,
                            model=model_best,
                            metric_name=metric,
                            model_params_settings=model_params,
                            optuna_params_settings=optuna_params)

    print(f'---Model saved to MLFlow---')
