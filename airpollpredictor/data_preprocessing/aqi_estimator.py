# pylint: disable=E0401, R0913, R0914, W0703

"""
Module estimates AQI for cleaned pollutant data
"""
import os
import pandas as pd
from settings import settings
from .aqi_calculations import aqi_calculator as aqc


def read_and_calculate_aqi(source_data_path: str, pollutants_codes: [int]):
    """
    Reads csv files for pollutants and calculates AQI indices for pollutant files with clean data
    @param source_data_path: The path to pollutant files
    @param pollutants_codes: The list of pollutant codes
    @return: List of dataframes with calculated AQI per pollutant
    """
    df_aqi_list = []
    for pollutant_id in pollutants_codes:
        measure = settings.POL_MEASURES[pollutant_id]
        df_source = pd.read_csv(os.path.join(source_data_path, f'{pollutant_id}.csv'),
                                parse_dates=True, index_col=settings.DATE_COLUMN_NAME)
        df_aqi_list.append(aqc.calc_aqi_for_day_pd(pollutant_id, df_source, measure))
    return df_aqi_list


def calculate_aqi(df_pol_list: list[pd.DataFrame], pollutants_codes: [int]):
    """
    Calculate AQI indices for pollutant files with clean data
    @param df_pol_list: List with concentration datasets per pollutant
    @param pollutants_codes: The list of pollutant codes
    @return: List of dataframes with calculated AQI per pollutant
    """
    df_aqi_list = []
    for i in range(len(pollutants_codes)):
        pollutant_id = pollutants_codes[i]
        measure = settings.POL_MEASURES[pollutant_id]
        df_aqi_list.append(aqc.calc_aqi_for_day_pd(pollutant_id, df_pol_list[i], measure))
    return df_aqi_list
