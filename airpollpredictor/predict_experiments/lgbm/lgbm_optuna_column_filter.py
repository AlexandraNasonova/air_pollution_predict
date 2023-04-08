# pylint: disable=E0401, R0913, R0914, W0703, R0902

"""
Module for filtering columns of the dataframe for different experiments
"""
import datetime
from enum import Enum
import re
import pandas as pd
from ml_models_search import ts_splitter
from ml_models_search.params_searchers.optuna_lgb_search import OptunaLgbSearch

# import aqi_calculations.aqi_calculator as aqc

POL_MEASURES = {7: "µg/m3", 6001: "µg/m3", 5: "µg/m3", 10: "mg/m3", 1: "µg/m3", 8: "µg/m3"}
POL_CODES = [7, 6001, 5, 8]
POL_NAMES = {7: "O3", 6001: "PM25", 5: "PM10", 8: "NO2"}

DATE_USE_COLUMNS = ['weekday', 'day', 'month', 'year', 'season', 'is_weekend', 'is_new_year']
WEATHER_USE_COLUMNS = ['tavg', 'tmin', 'tmax', 'prcp', 'wdir', 'wspd', 'wpgt', 'pres']

METRIC = 'rmse'
OBJECTIVE = 'regression'
OPTIMIZATION_DIRECTION = 'minimize'


class PredictionValueType(Enum):
    """
    Enum with types of the values for predictions
    """
    AQI = 1
    LAG = 2
    C_MEAN = 3
    C_MAX = 4


def __get_value_type_str(prediction_value_type: PredictionValueType):
    if prediction_value_type == PredictionValueType.AQI:
        value_type = 'AQI'
    elif prediction_value_type == PredictionValueType.C_MEAN:
        value_type = 'C_mean'
    elif prediction_value_type == PredictionValueType.C_MAX:
        value_type = 'C_max'
    else:
        value_type = 'AQI'
    return value_type


def __get_required_columns(df_timeseries: pd.DataFrame,
                           pol_id: int,
                           target_column_name: str,
                           use_aqi_cols: bool,
                           use_c_mean_cols: bool,
                           use_c_median_cols: bool,
                           use_c_max_cols: bool,
                           use_c_min_cols: bool,
                           use_lag_cols: bool,
                           use_gen_lags_cols: bool,
                           use_pol_cols: bool,
                           use_weather_cols: bool
                           ) -> list:
    if pol_id > 0:
        cols = [x for x in df_timeseries.columns.values if x.find(POL_NAMES[pol_id]) > 0]
    else:
        cols = [x for x in df_timeseries.columns.values
                if [p for p in POL_CODES if x.find(POL_NAMES[p]) > 0]]

    all_values_columns = [x for x in df_timeseries.columns.values if
                          [p for p in POL_CODES if x.endswith(POL_NAMES[p])]] + [
                             'AQI'] + ['Pollutant']
    cols = [x for x in cols if x not in all_values_columns]

    if not use_gen_lags_cols:
        regular_expr = re.compile(r".*_lag\d+d_.*")
        df_gen_lags = list(filter(regular_expr.match, cols))
        cols = [x for x in cols if x not in df_gen_lags]

    if not use_lag_cols:
        regular_expr = re.compile(r".*_lag\d+$")
        df_lags = list(filter(regular_expr.match, cols))
        cols = [x for x in cols if x not in df_lags]

    if not use_aqi_cols:
        cols = [x for x in cols if not x.startswith('AQI_')]
    if not use_c_mean_cols:
        cols = [x for x in cols if not x.startswith('C_mean')]
    if not use_c_median_cols:
        cols = [x for x in cols if not x.startswith('C_median')]
    if not use_c_max_cols:
        cols = [x for x in cols if not x.startswith('C_max')]
    if not use_c_min_cols:
        cols = [x for x in cols if not x.startswith('C_min')]
    if not use_pol_cols:
        cols = [x for x in cols if not x.startswith('Pollutant')]

    if use_weather_cols:
        cols += WEATHER_USE_COLUMNS

    cols = DATE_USE_COLUMNS + cols
    if target_column_name not in cols:
        cols = [target_column_name] + cols

    return cols


def init_optuna(df_timeseries: pd.DataFrame, pol_id: int,
                prediction_value_type: PredictionValueType,
                train_start_dt: datetime.date, train_end_dt: datetime.date,
                test_start_dt: datetime.date, test_end_dt: datetime.date,
                use_aqi_cols: bool, use_c_mean_cols: bool, use_lag_cols: bool,
                use_gen_lags_cols: bool, use_weather_cols: bool,
                use_c_median_cols=False, use_c_max_cols=False,
                use_c_min_cols=False, use_pol_cols=False,
                default_params=None, default_category=None,
                categories_for_optimization=None,
                default_top_features_count=-1) -> \
        (OptunaLgbSearch, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Splits timeseries for experiments, initializes Optuna LightGbm Wrapper
    @param df_timeseries: The timeseries
    @param pol_id: The standard identificator of the pollutant
    @param prediction_value_type: The type of the prediction value
    (only Aqi and mean concentration were tested)
    @param train_start_dt: The first date of the train dataset
    @param train_end_dt: The last date of the train dataset
    @param test_start_dt: The first date of the test dataset
    @param test_end_dt: The last date of the train dataset
    @param use_aqi_cols: The flag if the AQI columns should be included to
    the features datasets
    @param use_c_mean_cols: The flag if the Mean Concentration columns should be
    included to the features datasets
    @param use_lag_cols: The flag if the Lag columns should be included to
    the features datasets
    @param use_gen_lags_cols: The flag if the Aggregated Lag columns should
    be included to the features datasets
    @param use_weather_cols: The flag if the Weather columns should
    be included to the features datasets
    @param use_c_median_cols: The flag if the Median Concentration columns should be included to
    the features datasets
    @param use_c_max_cols: The flag if the Max Concentration columns should be included to
    the features datasets
    @param use_c_min_cols: The flag if the Min Concentration columns should be included to
    the features datasets
    @param use_pol_cols: The flag if the Pollutant columns should be included to
    the features datasets (!not tested yet)
    @param default_params: The default model params (optional)
    @param default_category: The default set of categories (optional)
    @param categories_for_optimization: The list of sets of categories for
    search the best one (optional)
    @param default_top_features_count: The default quantity of the most important
    features to be used
    @return: The Optuna LightGBM wrapper instance and train/test X/y filtered dataframes
    """
    value_type = __get_value_type_str(prediction_value_type)
    target_column_name = value_type if pol_id <= 0 else f'{value_type}_{POL_NAMES[pol_id]}'

    req_cols = __get_required_columns(df_timeseries=df_timeseries,
                                      pol_id=pol_id,
                                      target_column_name=target_column_name,
                                      use_aqi_cols=use_aqi_cols,
                                      use_c_mean_cols=use_c_mean_cols,
                                      use_c_median_cols=use_c_median_cols,
                                      use_c_max_cols=use_c_max_cols,
                                      use_c_min_cols=use_c_min_cols,
                                      use_lag_cols=use_lag_cols,
                                      use_gen_lags_cols=use_gen_lags_cols,
                                      use_pol_cols=use_pol_cols,
                                      use_weather_cols=use_weather_cols)
    df_use = df_timeseries[req_cols]

    x_train_filt, y_train_filt = \
        ts_splitter.split_x_y_for_period(df_timeseries=df_use, index_cols='DatetimeEnd',
                                         y_value_col=target_column_name,
                                         dt_start=train_start_dt, dt_end=train_end_dt)
    x_val_filt, y_val_filt = \
        ts_splitter.split_x_y_for_period(df_timeseries=df_use, index_cols='DatetimeEnd',
                                         y_value_col=target_column_name, dt_start=test_start_dt,
                                         dt_end=test_end_dt)

    optuna_helper = OptunaLgbSearch(study_name=f'lgbm_{pol_id if pol_id > 0 else "all"}',
                                    metric=METRIC,
                                    objective=OBJECTIVE,
                                    x_train=x_train_filt, y_train=y_train_filt,
                                    x_val=x_val_filt, y_val=y_val_filt,
                                    default_params=default_params,
                                    default_category=default_category,
                                    categories_for_optimization=categories_for_optimization,
                                    default_top_features_count=default_top_features_count)
    return optuna_helper, x_train_filt, y_train_filt, x_val_filt, y_val_filt

# def convert_conc_to_aqi(pol_id: int, target_values):
#     measure = POL_MEASURES[pol_id]
#     df_aqi = aqc.calc_aqi_for_day_pd(pol_id, target_values, measure)
#     return df_aqi

# def filter_data(df_timeseries: pd.DataFrame,
#                 pol_id: int,
#                 target_column_name: str,
#                 use_aqi_cols: bool,
#                 use_c_mean_cols: bool,
#                 use_c_median_cols: bool,
#                 use_c_max_cols: bool,
#                 use_c_min_cols: bool,
#                 use_lag_cols: bool,
#                 use_gen_lags_cols: bool,
#                 use_pol_cols: bool,
#                 use_weather_cols: bool) -> pd.DataFrame:
#     req_cols = __get_required_columns(df_timeseries=df_timeseries,
#                                       pol_id=pol_id,
#                                       target_column_name=target_column_name,
#                                       use_aqi_cols=use_aqi_cols,
#                                       use_c_mean_cols=use_c_mean_cols,
#                                       use_c_median_cols=use_c_median_cols,
#                                       use_c_max_cols=use_c_max_cols,
#                                       use_c_min_cols=use_c_min_cols,
#                                       use_lag_cols=use_lag_cols,
#                                       use_gen_lags_cols=use_gen_lags_cols,
#                                       use_pol_cols=use_pol_cols,
#                                       use_weather_cols=use_weather_cols)
#     return df_timeseries[req_cols]
