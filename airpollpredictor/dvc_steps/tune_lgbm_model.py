"""
DVC Stage tune_lgbm_model - find best LGBM model params using optuna
"""
# pylint: disable=E0401, R0913, W1514


from argparse import ArgumentParser
import datetime
# import json
import mlflow
import warnings
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import yaml
from settings import settings, env_reader
import data_preprocessing.columns_filter as col_filter
from model_tune_helpers import ts_splitter
from model_tune_helpers.lgbm_optuna.optuna_lgb_search import OptunaLgbSearch
from model_tune_helpers import onnx_adapter
from model_tune_helpers import metrics_adapter

warnings.filterwarnings('ignore')

STAGE = "tune_lgbm_model"


# noinspection DuplicatedCode
def __parse_args():
    parser = ArgumentParser(STAGE)
    parser.add_argument('--input_train_file', required=True, help='Path to input train data')
    parser.add_argument('--input_val_file', required=True, help='Path to input validation data')
    parser.add_argument('--output_metrics_file', required=True, help='Path to metrics file')
    parser.add_argument('--output_model_params_file', required=True, help='Path to metrics file')
    parser.add_argument('--output_onnx_file', required=True, help='Path to onnx file')
    parser.add_argument('--params', required=True, help='Path to params')
    parser.add_argument('--params_section', required=True, help='Section with filter params')
    parser.add_argument('--mlflow_env_file', required=True, help='Path to env file MlFlow')
    # parser.add_argument('--mlflow_artifact', required=True, help='MLFlow model artifacts path')
    return parser.parse_args()


def __get_train_val(input_train_file, input_val_file, target_column):
    df_train = pd.read_csv(input_train_file, parse_dates=True,
                           index_col=settings.DATE_COLUMN_NAME)
    df_val = pd.read_csv(input_val_file, parse_dates=True,
                         index_col=settings.DATE_COLUMN_NAME)
    x_tr, y_tr = ts_splitter.extract_labels(df_train, target_column)
    x_vl, y_vl = ts_splitter.extract_labels(df_val, target_column)
    return x_tr, y_tr, x_vl, y_vl


def __run_optuna(metric, optuna_params, model_params,
                 x_train, y_train, x_val, y_val, run_name):
    optuna_tuner = OptunaLgbSearch(
        study_name=run_name,
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
        best_features_only=False,
        search_category=model_params["search_category"],
        with_pruner=True,
        cv_splitter=TimeSeriesSplit(optuna_params["cv_folders"]),
        warm_params=None)
    return optuna_tuner


def _get_best_categories(model_params, optuna_tuner):
    if model_params["search_category"]:
        cat_features = optuna_tuner.study_best_params['categorical_features']
    else:
        cat_features = model_params["default_category"]
    return cat_features


def __mlflow_init_autolog(experiment_name):
    env_values = env_reader.get_params(stage_args.mlflow_env_file)
    mlflow.set_tracking_uri(f'{env_values.get("MLFLOW_URI")}:{env_values.get("MLFLOW_PORT")}')
    timestamp = datetime.datetime.now().strftime("%d_%m_%Y_T_%H_%M_%S")
    # experiment_name = f'{experiment_name_pref}_{timestamp}'
    print(f'experiment_name: {experiment_name}')
    mlflow.set_experiment(experiment_name)
    # with mlflow.start_run(run_name="timestamp") as run:
    mlflow.lightgbm.autolog()
    # return run.info.experiment_id, run.info.run_id


def __mlflow_save_metrics(metric, train_score_all, val_score_all, run_name):
    autolog_run = mlflow.last_active_run()
    with mlflow.start_run(run_id=autolog_run.info.run_id):
        mlflow.log_metric(f'train_{metric}_all', train_score_all)
        mlflow.log_metric(f'val_{metric}_all', val_score_all)

    mlflow.tracking.MlflowClient().set_tag(autolog_run.info.run_id,
                                           "mlflow.runName", run_name)


if __name__ == '__main__':
    print(f'Stage {STAGE} started')
    stage_args = __parse_args()

    with open(stage_args.params, 'r', encoding='UTF-8') as file_stream:
        yaml_params = yaml.safe_load(file_stream)
    model_params = yaml_params[stage_args.params_section]
    metric = yaml_params["metric"]

    target_column_name = col_filter.get_target_column(
        prediction_value_type=model_params['prediction_value_type'],
        pol_id=model_params["pol_id"])

    x_train, y_train, x_val, y_val = __get_train_val(
        stage_args.input_train_file, stage_args.input_val_file,
        target_column_name)

    run_name = f'{model_params["exp_name"]}_{model_params["run_name"]}'
    optuna_tuner = __run_optuna(metric=metric,
                                optuna_params=yaml_params["optuna"],
                                model_params=model_params,
                                x_train=x_train, y_train=y_train,
                                x_val=x_val, y_val=y_val,
                                run_name=model_params["run_name"])

    print('----Optuna finished params tuning---')
    cat_features = _get_best_categories(model_params, optuna_tuner)

    __mlflow_init_autolog(model_params["exp_name"])

    train_score_best, val_score_best, model_best = optuna_tuner.run_model_and_eval(
        params=optuna_tuner.study_best_params,
        categorical_features=cat_features,
        best_features_only=True,
        set_as_best_model=False)

    print(f'---Model trained with best params: '
          f'best_train_score: {train_score_best}, '
          f'best_val_score: {val_score_best}')

    __mlflow_save_metrics(metric, train_score_best, val_score_best, run_name)
    print(f'---Model saved to MLFlow---')

    metrics_adapter.save_metrics_to_json(
        metrics_file_path=stage_args.output_metrics_file,
        train_score=train_score_best,
        val_score=val_score_best,
        metric_name=metric)
    onnx_adapter.save_model(x_train_df=x_train, model=model_best,
                            onnx_file_path=stage_args.output_onnx_file)
    print('---Model saved to ONNX, metrics to json---')
    print(f'Stage {STAGE} finished')


#
# def __save_model_params(model_params_output_file_path: str, model_params_to_save: dict):
#     with open(model_params_output_file_path, 'w') as f_stream:
#         json.dump(model_params_to_save, f_stream)

# def __save_model_locally():
#     metrics_adapter.save_metrics_to_json(
#         metrics_file_path=stage_args.output_metrics_file,
#         train_score=train_score_best,
#         val_score=val_score_best,
#         metric_name=metric)
#
#     print('---Metrics saved locally---')
#
#     tuned_model_params = optuna_tuner.study_best_params
#     tuned_model_params['top_features_count'] = optuna_tuner.best_features_count
#     __save_model_params(model_params_output_file_path=stage_args.output_model_params_file,
#                         model_params_to_save=tuned_model_params)
#
#     print('---Model params saved locally---')