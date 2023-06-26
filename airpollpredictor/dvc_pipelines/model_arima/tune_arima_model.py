"""
DVC Stage tune_lgbm_model - find best ARIMA model params using pmarima
"""
# pylint: disable=E0401, R0913, W1514


from argparse import ArgumentParser
import warnings
import pandas as pd
import pmdarima as pm
from sklearn.metrics import mean_squared_error
import yaml
from settings import settings
import data_preprocessing.columns_filter as col_filter
from model_tune_helpers.models_saving import joblib_adapter, json_adapter
from model_tune_helpers.arima.transformations import TransformHelper
from model_tune_helpers.models_saving.mlflow_adapter import MlFlowAdapter

warnings.filterwarnings('ignore')

STAGE = "tune_arima_model"


# noinspection DuplicatedCode
def __parse_args():
    parser = ArgumentParser(STAGE)
    parser.add_argument('--input_train_file', required=True, help='Path to input train data')
    parser.add_argument('--input_val_file', required=True, help='Path to input validation data')
    parser.add_argument('--output_params_file', required=True, help='Path to params file')
    parser.add_argument('--output_metrics_file', required=True, help='Path to metrics file')
    parser.add_argument('--output_model_file', required=True, help='Path to model file')
    parser.add_argument('--params', required=True, help='Path to params')
    parser.add_argument('--params_section', required=True, help='Section with filter params')
    parser.add_argument('--mlflow_env_file', required=True, help='Path to env file MlFlow')
    # parser.add_argument('--mlflow_artifact', required=True, help='MLFlow model artifacts path')
    return parser.parse_args()


def __get_train_val_fourier_exog(input_train_file: str, input_val_file: str,
                                 target_column: str, fourier_params: {},
                                 exog_params: {}) \
        -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    df_train = pd.read_csv(input_train_file, parse_dates=True,
                           index_col=settings.DATE_COLUMN_NAME)
    df_val = pd.read_csv(input_val_file, parse_dates=True,
                         index_col=settings.DATE_COLUMN_NAME)
    y_train = df_train[target_column]
    y_val = df_val[target_column]

    fourier_train, fourier_val = None, None
    exog_train, exog_val = None, None

    if fourier_params["apply"]:
        fourier_train, fourier_val = TransformHelper.apply_fourier(
            y_train=y_train, y_val=y_val,
            date_column=settings.DATE_COLUMN_NAME,
            m_season_period=fourier_params["period"],
            k_sins=fourier_params["k_sins"])

    if exog_params["apply"]:
        exog_train = df_train[exog_params["exog_columns"]]
        exog_val = df_val[exog_params["exog_columns"]]

    if fourier_params["apply"] and exog_params["apply"]:
        exog_fourier_train = fourier_train.merge(
            exog_train, left_index=True, right_index=True)
        exog_fourier_val = fourier_val.merge(
            exog_val, left_index=True, right_index=True)
        return y_train, y_val, exog_fourier_train, exog_fourier_val

    if fourier_params["apply"]:
        return y_train, y_val, fourier_train, fourier_val

    if exog_params["apply"]:
        return y_train, y_val, exog_train, exog_val

    return y_train, y_val, None, None


def __predict_rmse(model, y_val, exog) -> float:
    predictions = model.predict(n_periods=len(y_val), X=exog)
    return mean_squared_error(y_true=y_val, y_pred=predictions, squared=False)


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

    y_train, y_val, exog_train, exog_val = __get_train_val_fourier_exog(
        input_train_file=stage_args.input_train_file,
        input_val_file=stage_args.input_val_file,
        target_column=target_column_name,
        fourier_params=model_params["fourier_params"],
        exog_params=model_params["exog_params"]
    )

    run_name = f'{model_params["exp_name"]}_{model_params["run_name"]}'
    # init MlFlow experiment and autolog
    mlflow_adapter = MlFlowAdapter(
        model_tp=MlFlowAdapter.ModelType.PD_ARIMA)
    mlflow_adapter.set_experiment(experiment_name=model_params["exp_name"],
                                  mlflow_env_file=stage_args.mlflow_env_file)
    mlflow_adapter.start_run(run_name=run_name)

    try:
        mlflow_adapter.save_extra_params(model_params, "pipeline/pipeline_params.yaml")
        print('----PMARIMA started---')
        model = pm.auto_arima(y=y_train, X=exog_train, **model_params["auto_arima_params"])
        print('----PMARIMA finished params tuning---')
        mlflow_adapter.save_model(model=model, x_train_df=exog_train,
                                  y_train_df=y_train,
                                  artifact_path="model",
                                  best_model_params=model.get_params())
        joblib_adapter.save_model(model_file_path=stage_args.output_model_file,
                                  model=model)
        json_adapter.save_params_to_json(file_path=stage_args.output_params_file,
                                         params=model.get_params())
        mlflow_adapter.save_artifact(stage_args.output_model_file,
                                     artifact_path="pkl")
        print(f'---Model is saved')

        train_score_best = __predict_rmse(model, y_train, exog_train)
        val_score_best = __predict_rmse(model, y_val, exog_val)
        print(f'---Model trained with best params: '
              f'best_train_score: {train_score_best}, '
              f'best_val_score: {val_score_best}')
        mlflow_adapter.save_metrics(train_score=train_score_best,
                                    val_score=val_score_best,
                                    metric_name=metric)
        json_adapter.save_metrics_to_json(file_path=stage_args.output_metrics_file,
                                          train_score=train_score_best,
                                          val_score=val_score_best,
                                          metric_name=metric)
        print(f'---Metrics are saved---')
        print(f'Stage {STAGE} finished')

    finally:
        mlflow_adapter.end_run()
