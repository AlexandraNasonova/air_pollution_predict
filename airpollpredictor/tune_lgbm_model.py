"""
DVC Stage tune_lgbm_model - find best LGBM model params using optuna
"""
# pylint: disable=E0401, R0913, W1514


from argparse import ArgumentParser
import json
import warnings
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import yaml
from settings import settings
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
    # parser.add_argument('--mlflow_artifact', required=True, help='MLFlow model artifacts path')
    return parser.parse_args()


def __save_model_params(model_params_output_file_path: str, model_params_to_save: dict):
    with open(model_params_output_file_path, 'w') as f_stream:
        json.dump(model_params_to_save, f_stream)


if __name__ == '__main__':
    print(f'Stage {STAGE} started')

    # noinspection DuplicatedCode
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
    x_train, y_train = ts_splitter.extract_labels(df_train, target_column_name)
    x_val, y_val = ts_splitter.extract_labels(df_val, target_column_name)

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

    print('----Optuna finished params tuning---')

    # noinspection DuplicatedCode
    train_score_best, val_score_best, model_best = optuna_tuner.run_model_and_eval(
        params=optuna_tuner.study_best_params,
        categorical_features=optuna_tuner.study_best_params['categorical_features'],
        best_features_only=True,
        set_as_best_model=False)

    print(f'---Model trained with best params: '
          f'best_train_score: {train_score_best}, best_val_score: {val_score_best}')

    metrics_adapter.save_metrics_to_json(
        metrics_file_path=stage_args.output_metrics_file,
        train_score=train_score_best,
        val_score=val_score_best,
        metric_name=metric)

    print('---Metrics saved locally---')

    tuned_model_params = optuna_tuner.study_best_params
    tuned_model_params['top_features_count'] = optuna_tuner.best_features_count
    __save_model_params(model_params_output_file_path=stage_args.output_model_params_file,
                        model_params_to_save=tuned_model_params)

    print('---Model params saved locally---')

    # __send_model_to_ml_flow(x_train_df=x_train,
    #                         y_train_df=y_train,
    #                         train_score=train_score_best,
    #                         val_score=val_score_best,
    #                         artifact_path=stage_args.mlflow_artifact,
    #                         model=model_best,
    #                         metric_name=metric,
    #                         model_params_settings=model_params,
    #                         optuna_params_settings=optuna_params)

    # print(f'---Model saved to MLFlow---')

    onnx_adapter.save_model(x_train_df=x_train, model=model_best,
                            onnx_file_path=stage_args.output_onnx_file)

    print('---Model saved to ONNX---')

    print(f'Stage {STAGE} finished')
