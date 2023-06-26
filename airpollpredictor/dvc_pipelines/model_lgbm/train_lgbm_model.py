"""
DVC Stage train_lgbm_model - trains LGBM models with updated data
"""
# pylint: disable=E0401, R0913, W1514

from argparse import ArgumentParser
import warnings
import pandas as pd
import yaml
from settings import settings
import data_preprocessing.columns_filter as col_filter
from model_tune_helpers import ts_splitter
from model_tune_helpers.lgbm_optuna.optuna_lgb_search import OptunaLgbSearch
from model_tune_helpers.models_saving import onnx_adapter, json_adapter

warnings.filterwarnings('ignore')

STAGE = "train_lgbm_model"


# noinspection DuplicatedCode
def __parse_args():
    parser = ArgumentParser(STAGE)
    parser.add_argument('--input_train_file', required=True, help='Path to input train data')
    parser.add_argument('--input_val_file', required=True, help='Path to input validation data')
    parser.add_argument('--output_metrics_file', required=True, help='Path to metrics file')
    parser.add_argument('--input_model_params_file', required=True, help='Path to metrics file')
    parser.add_argument('--output_onnx_file', required=True, help='Path to onnx file')
    parser.add_argument('--params', required=True, help='Path to params')
    parser.add_argument('--params_section', required=True, help='Section with filter params')
    # parser.add_argument('--mlflow_artifact', required=True, help='MLFlow model artifacts path')
    return parser.parse_args()


if __name__ == '__main__':
    print(f'Stage {STAGE} started')

    # noinspection DuplicatedCode
    stage_args = __parse_args()
    with open(stage_args.params, 'r', encoding='UTF-8') as file_stream:
        yaml_params = yaml.safe_load(file_stream)
    with open(stage_args.input_model_params_file, 'r', encoding='UTF-8') as file_stream:
        train_best_params = yaml.safe_load(file_stream)

    metric = yaml_params["metric"]
    categorical_features = train_best_params['categorical_features']
    top_features_count = train_best_params['top_features_count']
    train_best_params.pop('categorical_features')
    train_best_params.pop('top_features_count')

    model_params = yaml_params[stage_args.params_section]
    target_column_name = col_filter.get_target_column(
        prediction_value_type=model_params['prediction_value_type'],
        pol_id=model_params["pol_id"])
    df_train = pd.read_csv(stage_args.input_train_file, parse_dates=True,
                           index_col=settings.DATE_COLUMN_NAME)
    df_val = pd.read_csv(stage_args.input_val_file, parse_dates=True,
                         index_col=settings.DATE_COLUMN_NAME)
    x_train, y_train = ts_splitter.extract_labels(df_train, target_column_name)
    x_val, y_val = ts_splitter.extract_labels(df_val, target_column_name)

    optuna_tuner = OptunaLgbSearch(
        study_name='',
        metric=metric,
        objective='',
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        default_params=train_best_params,
        default_category=categorical_features,
        categories_for_optimization=None,
        default_top_features_count=top_features_count)

    # noinspection DuplicatedCode
    train_score_best, val_score_best, model_best = optuna_tuner.run_model_and_eval(
        best_features_only=True,
        set_as_best_model=False)

    print(f'---Model trained with best params: '
          f'best_train_score: {train_score_best}, best_val_score: {val_score_best}')

    metrics_adapter.save_metrics_to_json(
        metrics_file_path=stage_args.output_metrics_file,
        train_score=train_score_best,
        val_score=val_score_best,
        metric_name=metric)

    print('---Metrics saved to JSON---')

    onnx_adapter.save_model(x_train_df=x_train, model=model_best,
                            onnx_file_path=stage_args.output_onnx_file)

    print('---Model saved to ONNX---')

    print(f'Stage {STAGE} finished')
