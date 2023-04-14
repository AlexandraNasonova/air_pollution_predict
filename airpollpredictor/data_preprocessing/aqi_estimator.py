# pylint: disable=E0401, R0913, R0914, W0703

"""
Module estimates AQI for cleaned pollutant data
"""
import os
import pandas as pd
from .aqi_calculations import aqi_calculator as aqc
from settings import settings


def calculate_aqi_indexes(source_data_path: str, pollutants_codes: [int]):
    """
    Calculate AQI indices for pollutant files with clean data
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


def save_aqi_data(pollutants_codes: [int], df_aqi_list: list[pd.DataFrame],
                  output_path: str):
    """
    Saves calculated AQI per pollutants to .csv files
    @param pollutants_codes: The list of pollutant codes
    @param df_aqi_list: List of dataframes with calculated AQI per pollutant
    @param output_path: The output path
    """
    # pylint: disable=C0200
    for i in range(len(pollutants_codes)):
        file_path = os.path.join(output_path, f'{pollutants_codes[i]}.csv')
        df_aqi_list[i].to_csv(file_path)
