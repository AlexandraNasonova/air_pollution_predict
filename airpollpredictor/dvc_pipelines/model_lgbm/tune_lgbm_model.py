"""
DVC Stage tune_lgbm_model - find best LGBM model params using optuna
"""
# pylint: disable=E0401, R0913, W1514


from argparse import ArgumentParser
# import json
import warnings
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import yaml
from settings import settings
import data_preprocessing.columns_filter as col_filter
from model_tune_helpers import ts_splitter
from model_tune_helpers.lgbm_optuna.optuna_lgb_search import OptunaLgbSearch
from model_tune_helpers.models_saving import onnx_adapter, json_adapter
from model_tune_helpers.models_saving.mlflow_adapter import MlFlowAdapter

warnings.filterwarnings('ignore')

STAGE = "tune_lgbm_model"


# noinspection DuplicatedCode
def __parse_args():
    parser = ArgumentParser(STAGE)
    parser.add_argument('--input_train_file', required=True, help='Path to input train data')
    parser.add_argument('--input_val_file', required=True, help='Path to input validation data')
    parser.add_argument('--output_metrics_file', required=True, help='Path to metrics file')
    parser.add_argument('--output_onnx_file', required=True, help='Path to onnx file')
    parser.add_argument('--output_pred_file', required=False, help='Path to predicts file')
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


if __name__ == '__main__':
    print(f'Stage {STAGE} started')
    stage_args = __parse_args()

    with open(stage_args.params, 'r', encoding='UTF-8') as file_stream:
        yaml_params = yaml.safe_load(file_stream)
    model_params = yaml_params[stage_args.params_section]
    optuna_params = yaml_params["optuna"]
    metric = yaml_params["metric"]

    target_column_name = col_filter.get_target_column(
        prediction_value_type=model_params['prediction_value_type'],
        pol_id=model_params["pol_id"])

    x_train, y_train, x_val, y_val = __get_train_val(
        stage_args.input_train_file, stage_args.input_val_file,
        target_column_name)

    run_name = f'{model_params["exp_name"]}_{model_params["run_name"]}'

    print('----Optuna started---')
    optuna_tuner = __run_optuna(metric=metric,
                                optuna_params=optuna_params,
                                model_params=model_params,
                                x_train=x_train, y_train=y_train,
                                x_val=x_val, y_val=y_val,
                                run_name=model_params["run_name"])

    print('----Optuna finished params tuning---')
    cat_features = _get_best_categories(model_params, optuna_tuner)

    # init MlFlow experiment and autolog
    mlflow_adapter = MlFlowAdapter(
        model_tp=MlFlowAdapter.ModelType.LGBM)
    mlflow_adapter.set_experiment(experiment_name=model_params["exp_name"],
                                  mlflow_env_file=stage_args.mlflow_env_file)
    mlflow_adapter.init_autolog()

    train_score_best, val_score_best, model_best = optuna_tuner.run_model_and_eval(
        params=optuna_tuner.study_best_params,
        categorical_features=cat_features,
        best_features_only=True,
        set_as_best_model=True)

    onnx_adapter.save_lgbm_model(x_train_df=x_train, model=model_best,
                                 onnx_file_path=stage_args.output_onnx_file)

    mlflow_adapter.set_run_name_to_last_run(run_name=run_name)
    mlflow_adapter.save_extra_params_to_last_run(model_params,
                                                 "pipeline/pipeline_params.yaml")
    mlflow_adapter.save_extra_params_to_last_run(optuna_params,
                                                 "pipeline/optuna_params.yaml")
    mlflow_adapter.save_artifact_to_last_run(stage_args.output_onnx_file,
                                             artifact_path="pkl")

    predictions = optuna_tuner.predict_by_best_model(x_val)
    if stage_args.output_pred_file:
        pd.DataFrame(predictions, columns=['C1'])\
            .to_csv(stage_args.output_pred_file)
        mlflow_adapter.save_artifact_to_last_run(
            stage_args.output_pred_file, artifact_path="predictions")
    print(f'---Model is saved')

    print(f'---Model trained with best params: '
          f'best_train_score: {train_score_best}, '
          f'best_val_score: {val_score_best}')

    mlflow_adapter.save_metrics_to_last_run(
        metric_name=metric, train_score_all=train_score_best,
        val_score_all=val_score_best)
    json_adapter.save_metrics_to_json(
        file_path=stage_args.output_metrics_file,
        train_score=train_score_best,
        val_score=val_score_best,
        metric_name=metric)

    print(f'---Metrics are saved')
    print(f'Stage {STAGE} finished')
