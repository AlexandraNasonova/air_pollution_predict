# pylint: disable=E0401, R0913
"""
Module for enriching pollutants data with data and lag features
"""

import os
import pandas as pd
from settings import settings
from .features_generations import ts_lag_features_generator as lag_gen
from .features_generations import ts_date_features_generator as date_gen
from .aqi_calculations import aqi_calculator as aqc

CONCENTRATION_AGGREGATES = ['mean']
CONCENTRATION_AGGREGATES_FOR_LAGS = ['mean']
NO_FILTER = 'NoFilter'
ID_COLS = []


def generate_features(df_aqi_mean: pd.DataFrame,
                      pollutants_codes: list[int],
                      lags_shift: list[int],
                      filters_aqi: list[str],
                      windows_filters_aqi: dict,
                      methods_agg_aqi: list[str],
                      lags_agg_aqi: list[int],
                      ewm_filters_aqi: dict):
    """
    Generates lag and data features for pollutants and saves the result to one file
    @param df_aqi_mean: Merged dataframe
    @param pollutants_codes: The list of pollutant codes
    @param lags_shift: The list of lags for the shift
    @param filters_aqi: The list of columns for the lags filtering
    (for AQI columns)
    @param windows_filters_aqi: The dictionary of windows for rolling calculations
    per filter (for AQI columns)
    @param methods_agg_aqi: The list of aggregation methods for rolling calculations
    (for AQI columns)
    @param lags_agg_aqi: The list of lags for the shift for calculated aggregates
    (for AQI columns)
    @param ewm_filters_aqi: The dictionary of lags for Exponential Moving Average
    per filter (for AQI columns)
    @return:
    """
    df_gen = date_gen.add_date_info(df_aqi_mean)
    df_gen = __get_lag_data_shift(pollutants_codes=pollutants_codes,
                                  df_gen=df_gen,
                                  lags=lags_shift)
    df_gen[NO_FILTER] = 1
    target_cols = __get_aqi_columns(pollutants_codes, df_gen)
    df_gen = lag_gen \
        .generate_lagged_features(df_gen,
                                  target_cols=target_cols,
                                  id_cols=ID_COLS,
                                  date_col=settings.DATE_COLUMN_NAME,
                                  lags=lags_agg_aqi,
                                  windows=windows_filters_aqi,
                                  preagg_methods=CONCENTRATION_AGGREGATES,
                                  agg_methods=methods_agg_aqi,
                                  dynamic_filters=filters_aqi + [NO_FILTER],
                                  ewm_params=ewm_filters_aqi
                                  )
    return df_gen


def calc_aqi_and_mean_concentration_and_merge(
        source_data_path: str, pollutants_codes: list[int],
        date_from: str, date_end: str) -> pd.DataFrame:
    """
    Calculates mean concentrations and AQI, rollup lines from hours to days
    @param source_data_path: The path to pollutant files
    @param pollutants_codes: The list of pollutant codes
    @param date_from: The first date of the data sources
    @param date_end: The last date of the data sources
    @return: Dataframe with AQI, mean concentrations, rolled up from hours to days
    """
    df_gen = pd.DataFrame(
        index=pd.date_range(start=date_from, end=date_end, freq='D',
                            inclusive="both", name=settings.DATE_COLUMN_NAME))
    df_gen = __merge_pollutants(source_data_path=source_data_path,
                                pollutants_codes=pollutants_codes,
                                df_gen=df_gen)
    df_gen[settings.POLLUTANT_COLUMN_NAME] = df_gen.idxmax(axis=1) \
        .apply(lambda x: settings.POL_NAMES_REVERSE[x[x.index('_') + 1:]])
    return df_gen


def __merge_column_by_index(pollutant_id: int, df_gen: pd.DataFrame, df_to_merge: pd.DataFrame,
                            source_column: str, new_column=None) -> pd.DataFrame:
    if new_column is None:
        new_column = source_column

    df_gen = df_gen.merge(df_to_merge[source_column], left_index=True, right_index=True)
    df_gen = df_gen.rename(
        columns={source_column: f'{new_column}_{settings.POL_NAMES[pollutant_id]}'})
    return df_gen


def __merge_pollutants(
        source_data_path: str, pollutants_codes: list[int], df_gen: pd.DataFrame) -> pd.DataFrame:
    for pollutant_id in pollutants_codes:
        df_pollutant = pd.read_csv(os.path.join(source_data_path, f'{pollutant_id}.csv'),
                                   parse_dates=True, index_col=settings.DATE_COLUMN_NAME)
        df_pollutant = df_pollutant.tz_localize(None)
        df_gen = __merge_column_by_index(pollutant_id, df_gen, df_pollutant,
                                         settings.AQI_COLUMN_NAME)
    return df_gen


def __get_all_concentration_and_aqi_columns(pollutants_codes: list[int], df_gen: pd.DataFrame):
    return [col for col in df_gen.columns.values
            if [pol_code for pol_code in pollutants_codes
                if col.endswith(settings.POL_NAMES[pol_code])]]


def __get_aqi_columns(pollutants_codes: list[int], df_gen: pd.DataFrame):
    return [col for col in df_gen.columns.values
            if col.startswith(settings.AQI_COLUMN_NAME) and
            [pol_code for pol_code in pollutants_codes
             if col.endswith(settings.POL_NAMES[pol_code])]]


def __get_lag_data_shift(pollutants_codes: list[int], df_gen: pd.DataFrame, lags: []) \
        -> pd.DataFrame:
    df_gen_copy = df_gen.copy(deep=True)
    target_cols = __get_all_concentration_and_aqi_columns(pollutants_codes, df_gen_copy)
    for column in target_cols:
        for lag in lags:
            df_gen_copy[f'{column}_lag{lag}'] = df_gen[column].shift(lag)
    return df_gen_copy
