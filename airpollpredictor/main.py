# pylint: disable=E0401

import datetime
import requests

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


async def __post(url: str, params: dict):
    return requests.post(
        url=url,
        params={
            'save_dir_path': 'datasets_tests/weather-source-data/'
        },
        json=params
    )


async def load_weather_history_from_station():
    params = {'station_id': '06344'}
    response = await __post(url='http://127.0.0.1:8041/download_current_year', params=params)
    if response.status_code != 200:
        print("ERROR")
        print(response)
    print(response.url)


def __extract_labels(df_ts: pd.DataFrame, target_column: str):
    x_ts = df_ts.drop([target_column], axis=1)
    y_ts = df_ts[target_column]
    return x_ts, y_ts


if __name__ == '__main__':
    # save_path = asyncio.run(aqi_loader.pollutants_txt_lists_load())

    # save_path = asyncio.run(
    #     aqi_loader.pollutants_txt_lists_load(
    #         pollutant_codes=[7, 6001, 5, 8], country="NL",
    #         city='Rotterdam', year_from=2015,
    #         station_per_pollutant={7: 'STA-NL00418', 5: 'STA-NL00418',
    #         6001: 'STA-NL00448', 8: 'STA-NL00418'}))
    # save_path =
    # "/home/alexna/work/projects/air_pollution_predict/
    # airpollpredictor/aqreport_loader/data/02_11_2022_12_12_44"

    # asyncio.run(aqi_loader.csv_list_load(save_path))
    #
    # SAVE_DIR_PATH = 'datasets_tests/weather-source-data/'
    # date_from = datetime.strptime("2015-01-01", "%Y-%m-%d").date()
    # date_from = datetime.strptime("2023-04-01", "%Y-%m-%d").date()
    # save_path = asyncio.run(weather_loader.load_weather_history_from_station(
    #     save_dir_path=SAVE_DIR_PATH, station="06344", date_from=date_from))

    # SAVE_DIR_PATH = 'datasets_tests/pollutants-source-data/'
    # pol_service.download_prev_years(SAVE_DIR_PATH)

    # asyncio.run(load_weather_history_from_station())

    # YEAR_FROM = 2023
    # DATE_FROM = str(datetime.date(year=YEAR_FROM, month=1, day=1))
    # DATE_TO = datetime.datetime.now().date()
    # print(DATE_FROM)
    # print(type(DATE_TO))



    file_name = "params.yaml"
    with open(file_name, 'r', encoding='UTF-8') as file_stream:
        yaml_params = yaml.safe_load(file_stream)

    model_params = yaml_params["lgbm_pm25"]
    metric = yaml_params["metric"]
    optuna_params = yaml_params["optuna"]
    pol_id = model_params["pol_id"]
    target_column_name = col_filter.get_target_column(
        prediction_value_type=model_params['prediction_value_type'],
        pol_id=pol_id)

    df_train = pd.read_csv("experiments_results/lgbm/6001/train.csv", parse_dates=True,
                           index_col=settings.DATE_COLUMN_NAME)
    df_val = pd.read_csv("experiments_results/lgbm/6001/val.csv", parse_dates=True,
                         index_col=settings.DATE_COLUMN_NAME)
    x_train, y_train = __extract_labels(df_train, target_column_name)
    x_val, y_val = __extract_labels(df_val, target_column_name)

    default_params = {'n_jobs': -1, 'verbosity': -1, 'metric': 'rmse', 'boosting_type': 'gbdt',
                      'extra_trees': True, 'n_estimators': 1000, 'num_leaves': 150, 'learning_rate': 0.01,
                      'subsample': 0.7, 'subsample_freq': 5, 'subsample_for_bin': 100000, 'min_child_samples': 30,
                      'reg_alpha': 0.1, 'reg_lambda': 0.2, 'max_depth': 10, 'max_bin': 150}

    optuna_tuner = OptunaLgbSearch(
        study_name=f'lgbm_{pol_id if pol_id > 0 else "all"}',
        metric=metric,
        objective=optuna_params["objective"],
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        default_params= default_params, #model_params["default_params"],
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

    train_score_best, val_score_best, model_best = optuna_tuner.run_model_and_eval(
        params=optuna_tuner.study_best_params,
        categorical_features=optuna_tuner.study_best_params['categorical_features'],
        best_features_only=True,
        set_as_best_model=False)
