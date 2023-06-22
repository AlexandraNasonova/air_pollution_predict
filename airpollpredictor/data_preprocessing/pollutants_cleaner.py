# pylint: disable=E0401
"""
Module for cleaning pollutant data
"""

import warnings
import numpy as np
import pandas as pd
from settings import settings


def drop_sampling_unverified_duplicates(pollutants_codes: [int],
                                        df_list: list[pd.DataFrame]):
    """
    Drop duplicate lines (the ones with the same data except Sampling point, leaves verified data)
    @param pollutants_codes: The list of pollutant codes
    @param df_list: List of dataframes with merged for every pollutant data
    """
    for i in range(len(pollutants_codes)):
        df_list[i] = df_list[i].sort_values('Verification') \
            .drop_duplicates(subset=['AirQualityStation', 'DatetimeEnd'], keep='first')


def convert_negative_values_to_nan(pollutants_codes: list[int],
                                   df_list: list[pd.DataFrame]):
    """
    Converts negative values to Nan
    @param pollutants_codes: The list of pollutant codes
    @param df_list: List of dataframes with merged for every pollutant data
    """
    for i in range(len(pollutants_codes)):
        bad_mask = df_list[i][settings.CONCENTRATION_COLUMN_NAME] < 0
        df_list[i].loc[bad_mask, settings.CONCENTRATION_COLUMN_NAME] = np.NaN


def fix_non_hour_intervals(pollutants_codes: list[int],
                           df_list: list[pd.DataFrame]):
    """
    Fixes non-hour intervals if exist
    (i.e. for the day intervals the method adds line per each hour and fills them with Nan)
    @param pollutants_codes: The list of pollutant codes
    @param df_list: List of dataframes with merged for every pollutant data
    """
    df_df_days = __get_non_hour_intervals(pollutants_codes, df_list)
    for i in range(len(pollutants_codes)):
        if df_df_days[i] is None:
            continue
        df_list[i] = pd.concat([df_list[i], __get_hour_columns(df_df_days[i])],
                               axis=0, ignore_index=True)


def remove_unused_columns(pollutants_codes: list[int],
                          df_list: list[pd.DataFrame]):
    """
    Removes redundant columns
    @param pollutants_codes: The list of pollutant codes
    @param df_list: List of dataframes with merged for every pollutant data
    """
    for i in range(len(pollutants_codes)):
        df_list[i] = df_list[i].drop(
            columns=['AirQualityStation', 'Verification', 'Validity', 'UnitOfMeasurement',
                     'AveragingTime', 'SamplingPoint', 'SamplingProcess', 'Countrycode'])


def __get_hour_columns(df_days) -> pd.DataFrame:
    df_days.assign(AveragingTime='hour')
    df_days.assign(Concentration=np.NaN)
    df_dub = df_days.loc[df_days.index.repeat(23), :]
    df_range = df_days.loc[:, settings.DATE_COLUMN_NAME].apply(
        lambda x: pd.date_range(x - pd.Timedelta(hours=23),
                                x - pd.Timedelta(hours=1), freq='1h'))
    df_dub[settings.DATE_COLUMN_NAME] = df_range.explode()
    return df_dub


def __get_non_hour_intervals(pollutants_codes: [int],
                             df_list: list[pd.DataFrame]) -> list:
    non_hour_lines_count = 0
    df_df_days = []
    for i in range(len(pollutants_codes)):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
            df_list[i][settings.DATE_COLUMN_NAME] = \
                pd.to_datetime(df_list[i][settings.DATE_COLUMN_NAME])
        no_hour_mask = df_list[i]['AveragingTime'] != 'hour'
        df_hour_dif = df_list[i][no_hour_mask]
        non_hour_lines_count += df_hour_dif.shape[0]
        if df_hour_dif is not None and df_hour_dif.shape[0] > 0:
            df_df_days.append(df_hour_dif)
        else:
            df_df_days.append(None)
    return df_df_days
