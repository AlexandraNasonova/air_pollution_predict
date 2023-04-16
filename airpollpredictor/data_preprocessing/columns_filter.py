# pylint: disable=E0401, R0913, R0914, W0703, R0902

"""
Module for filtering columns of the dataframe for different experiments
"""
import re
import pandas as pd
from settings import settings


def get_target_column(prediction_value_type: str, pol_id: int = -1) -> str:
    '''
    Returns target columns name
    @param prediction_value_type: Prediction value type (AQI / MEAN / MEDIAN / MAX / MIN)
    @param pol_id: The code of the pollutant
    @return: Target columns name
    '''
    return prediction_value_type if pol_id <= 0 \
        else f'{prediction_value_type}_{settings.POL_NAMES[pol_id]}'


def filter_data_frame(df_timeseries: pd.DataFrame,
                      pol_codes: [],
                      weather_columns: [],
                      date_columns: [],
                      target_column_name: str,
                      use_aqi_cols: bool,
                      use_c_mean_cols: bool,
                      use_c_median_cols: bool,
                      use_c_max_cols: bool,
                      use_c_min_cols: bool,
                      use_lag_cols: bool,
                      use_gen_lags_cols: bool,
                      use_pol_cols: bool,
                      use_weather_cols: bool,
                      pol_id: int = -1
                      ) -> pd.DataFrame:
    """
    Returns dataframe with columns filtered by requirements
    @param df_timeseries: The timeseries
    @param pol_codes: The list of the pollutant codes
    @param weather_columns: The list of the required weather columns
    @param date_columns: The list of the required data columns
    @param pol_id: The standard identificator of the pollutant (optional)
    @param target_column_name: The name of the target column (for predictions)
    @param use_aqi_cols: The flag if the AQI columns should be included to
    the features datasets_tests
    @param use_c_mean_cols: The flag if the Mean Concentration columns should be
    included to the features datasets_tests
    @param use_lag_cols: The flag if the Lag columns should be included to
    the features datasets_tests
    @param use_gen_lags_cols: The flag if the Aggregated Lag columns should
    be included to the features datasets_tests
    @param use_weather_cols: The flag if the Weather columns should
    be included to the features datasets_tests
    @param use_c_median_cols: The flag if the Median Concentration columns should be included to
    the features datasets_tests
    @param use_c_max_cols: The flag if the Max Concentration columns should be included to
    the features datasets_tests
    @param use_c_min_cols: The flag if the Min Concentration columns should be included to
    the features datasets_tests
    @param use_pol_cols: The flag if the Pollutant columns should be included to
    the features datasets_tests (!not tested yet)
    @return:
    """
    if pol_id > 0:
        cols = [x for x in df_timeseries.columns.values if x.find(settings.POL_NAMES[pol_id]) > 0]
    else:
        cols = [x for x in df_timeseries.columns.values
                if [p for p in pol_codes if x.find(settings.POL_NAMES[p]) > 0]]

    all_values_columns = [x for x in df_timeseries.columns.values if
                          [p for p in pol_codes if x.endswith(settings.POL_NAMES[p])]] + [
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
        cols += weather_columns

    cols = date_columns + cols
    if target_column_name not in cols:
        cols = [target_column_name] + cols

    df_use = df_timeseries[cols]
    return df_use
