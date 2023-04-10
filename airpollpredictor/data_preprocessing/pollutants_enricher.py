# pylint: disable=E0401, R0913
"""
Module for enriching pollutants data with data and lag features
"""

import os
import pandas as pd
from .features_generations import ts_lag_features_generator as lag_gen
from .features_generations import ts_date_features_generator as date_gen
from .aqi_calculations import aqi_calculator as aqc
from . import settings

CONCENTRATION_AGGREGATES = ['mean']
CONCENTRATION_AGGREGATES_FOR_LAGS = ['mean']
NO_FILTER = 'NoFilter'
ID_COLS = []


def generate_features(source_data_path: str,
                      output_file: str,
                      pollutants_codes: list[int],
                      date_from: str,
                      date_end: str,
                      lags_shift: list[int],
                      filters_aqi: list[str],
                      windows_filters_aqi: dict,
                      methods_agg_aqi: list[str],
                      lags_agg_aqi: list[int],
                      ewm_filters_aqi: dict):
    """
    Generates lag and data features for pollutants and saves the result to one file
    @param source_data_path: The path to pollutant files
    @param output_file: The output path to the resulting .csv file
    @param pollutants_codes: The list of pollutant codes
    @param date_from: The first date of the data sources
    @param date_end: The last date of the data sources
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
    df_aqi_mean = calc_aqi_and_mean_concentration_and_merge(
        source_data_path=source_data_path, pollutants_codes=pollutants_codes,
        date_from=date_from, date_end=date_end)
    df_aqi_mean = date_gen.add_date_info(df_aqi_mean)
    df_aqi_mean_lags = __get_all_lag_data(
        pollutants_codes=pollutants_codes, df_gen=df_aqi_mean, lags_shift=lags_shift,
        filters_aqi=filters_aqi, windows_filters_aqi=windows_filters_aqi,
        methods_agg_aqi=methods_agg_aqi, lags_agg_aqi=lags_agg_aqi,
        ewm_filters_aqi=ewm_filters_aqi)
    __save_calc(df_aqi_mean_lags, output_file)


def __save_calc(df_enriched: pd.DataFrame, output_file: str):
    df_enriched.to_csv(output_file)


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
    df_gen = __calc_aqi_per_pollutant_and_merge_pollutants(
        source_data_path=source_data_path, pollutants_codes=pollutants_codes,
        df_gen=df_gen)
    # g['Pollutant'] = g.idxmax(axis=1).apply(lambda x: x[x.index('_') + 1:])
    df_gen[settings.AQI_COLUMN_NAME] = df_gen.max(axis=1)
    df_gen[settings.POLLUTANT_COLUMN_NAME] = df_gen.idxmax(axis=1) \
        .apply(lambda x: settings.POL_NAMES_REVERSE[x[x.index('_') + 1:]])
    df_gen = __calc_mean_concentration_and_merge_pollutants(
        source_data_path=source_data_path, pollutants_codes=pollutants_codes,
        df_gen=df_gen)
    return df_gen


def __read_dataframe_for_pollutant(source_data_path: str, pollutant_id: int):
    df_pollutant = pd.read_csv(os.path.join(source_data_path, f'{pollutant_id}.csv'),
                               parse_dates=True, index_col=settings.DATE_COLUMN_NAME)
    return df_pollutant


def __merge_column_by_index(pollutant_id: int, df_gen: pd.DataFrame, df_to_merge: pd.DataFrame,
                            source_column: str, new_column=None) -> pd.DataFrame:
    if new_column is None:
        new_column = source_column

    df_gen = df_gen.merge(df_to_merge[source_column], left_index=True, right_index=True)
    df_gen = df_gen.rename(
        columns={source_column: f'{new_column}_{settings.POL_NAMES[pollutant_id]}'})
    return df_gen


def __calc_aqi_per_pollutant_and_merge_pollutants(
        source_data_path: str, pollutants_codes: list[int], df_gen: pd.DataFrame) -> pd.DataFrame:
    for pollutant_id in pollutants_codes:
        df_pollutant = __read_dataframe_for_pollutant(source_data_path, pollutant_id)
        measure = settings.POL_MEASURES[pollutant_id]
        df_pollutant_agg = aqc.calc_aqi_for_day_pd(
            pollutant_id, df_pollutant, measure).tz_localize(None)
        df_gen = __merge_column_by_index(pollutant_id, df_gen, df_pollutant_agg,
                                         settings.AQI_COLUMN_NAME)
    return df_gen


def __calc_mean_concentration_and_merge_pollutants(
        source_data_path: str, pollutants_codes: list[int],
        df_gen: pd.DataFrame) -> pd.DataFrame:
    for pollutant_id in pollutants_codes:
        df_pollutant = __read_dataframe_for_pollutant(source_data_path, pollutant_id)
        for method in CONCENTRATION_AGGREGATES:
            df_pollutant_agg = df_pollutant[settings.CONCENTRATION_COLUMN_NAME] \
                .groupby(pd.Grouper(freq="24H")).agg(
                method).tz_localize(None).to_frame()
            df_gen = __merge_column_by_index(
                pollutant_id, df_gen, df_pollutant_agg, settings.CONCENTRATION_COLUMN_NAME,
                # pylint: disable=E1101,C0123
                f'C_{(method if type(method) is str else method.__name__).upper()}')
    return df_gen


def __get_all_concentration_and_aqi_columns(pollutants_codes: list[int], df_gen: pd.DataFrame):
    return [col for col in df_gen.columns.values
            if [pol_code for pol_code in pollutants_codes
                if col.endswith(settings.POL_NAMES[pol_code])]] \
        + [settings.AQI_COLUMN_NAME] + [settings.POLLUTANT_COLUMN_NAME]


def __get_aqi_columns(pollutants_codes: list[int], df_gen: pd.DataFrame):
    return [col for col in df_gen.columns.values
            if col.startswith(settings.AQI_COLUMN_NAME) and
            [pol_code for pol_code in pollutants_codes
             if col.endswith(settings.POL_NAMES[pol_code])]] \
        + [settings.AQI_COLUMN_NAME]


def __get_concentration_columns_by_method(pollutants_codes: list[int],
                                          df_gen: pd.DataFrame, agg_method: str):
    return [col for col in df_gen.columns.values if col.startswith(f'C_{agg_method.upper()}')
            and [pol_code for pol_code in pollutants_codes
                 if col.endswith(settings.POL_NAMES[pol_code])]]


def __get_lag_data_shift(pollutants_codes: list[int], df_gen: pd.DataFrame, lags: []) \
        -> pd.DataFrame:
    df_gen_copy = df_gen.copy(deep=True)
    target_cols = __get_all_concentration_and_aqi_columns(pollutants_codes, df_gen_copy)
    for column in target_cols:
        for lag in lags:
            df_gen_copy[f'{column}_lag{lag}'] = df_gen[column].shift(lag)
    return df_gen_copy


def __get_lag_data_aqi(pollutants_codes: list[int], df_gen: pd.DataFrame,
                       id_cols: [], filters: [],
                       windows: dict, agg_methods: list, lags: [],
                       ewm_params: dict) -> pd.DataFrame:
    target_cols = __get_aqi_columns(pollutants_codes, df_gen)
    df_lagged_features = lag_gen \
        .generate_lagged_features(df_gen,
                                  target_cols=target_cols,
                                  id_cols=id_cols,
                                  date_col=settings.DATE_COLUMN_NAME,
                                  lags=lags,
                                  windows=windows,
                                  preagg_methods=CONCENTRATION_AGGREGATES,
                                  agg_methods=agg_methods,
                                  dynamic_filters=filters,
                                  ewm_params=ewm_params
                                  )
    df_lagged_features.set_index(settings.DATE_COLUMN_NAME, inplace=True)
    return df_lagged_features


def __get_lag_data_concentration(pollutants_codes: list[int],
                               df_gen: pd.DataFrame,
                               id_cols: [],
                               filters: [],
                               windows: dict,
                               method,
                               lags: [],
                               ewm_params: dict) -> pd.DataFrame:
    target_cols = __get_concentration_columns_by_method(pollutants_codes, df_gen, method)
    agg_methods = [method]
    df_lagged_features = lag_gen \
        .generate_lagged_features(df_gen,
                                  target_cols=target_cols,
                                  id_cols=id_cols,
                                  date_col=settings.DATE_COLUMN_NAME,
                                  lags=lags,
                                  windows=windows,
                                  preagg_methods=CONCENTRATION_AGGREGATES,
                                  agg_methods=agg_methods,
                                  dynamic_filters=filters,
                                  ewm_params=ewm_params
                                  )
    df_lagged_features.set_index(settings.DATE_COLUMN_NAME, inplace=True)
    return df_lagged_features


def __get_all_lag_data(pollutants_codes: list[int],
                     df_gen: pd.DataFrame,
                     lags_shift: list[int],
                     filters_aqi: list[str],
                     windows_filters_aqi: dict,
                     methods_agg_aqi: list[str],
                     lags_agg_aqi: list[int],
                     ewm_filters_aqi: dict) -> pd.DataFrame:
    df_gen_shift = __get_lag_data_shift(pollutants_codes=pollutants_codes,
                                        df_gen=df_gen,
                                        lags=lags_shift)
    df_gen_shift[NO_FILTER] = 1
    df_gen_shift = __get_lag_data_aqi(pollutants_codes=pollutants_codes,
                                      df_gen=df_gen_shift,
                                      id_cols=ID_COLS,
                                      filters=filters_aqi + [NO_FILTER],
                                      windows=windows_filters_aqi,
                                      agg_methods=methods_agg_aqi,
                                      lags=lags_agg_aqi,
                                      ewm_params=ewm_filters_aqi)
    # for method in CONCENTRATION_AGGREGATES_FOR_LAGS:
    #     df_gen_shift = \
    #         get_lag_data_concentration(
    #             df_gen_shift, id_cols=ID_COLS,
    #             filters=FILTERS_FOR_MEAN_CONCENTRATION + [NO_FILTER],
    #             windows=WINDOWS_WITH_FILTERS_FOR_MEAN_CONCENTRATION,
    #             method=method,
    #             lags=LAGS_AGGREGATES_FOR_MEAN_CONCENTRATION,
    #             ewm_params=EWM_PARAMS_FOR_MEAN_CONCENTRATION)
    return df_gen_shift
