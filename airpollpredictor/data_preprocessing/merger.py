# pylint: disable=E0401, R0913, R0914, W0703
"""
Module mergers AQI and weather data
"""

import pandas as pd
from . import settings


def merge(weather_file_path: str, aqi_file_path: str, output_file_path: str):
    """
    Merges AQI and weather data and saves result
    @param weather_file_path: The path to clean weather data
    @param aqi_file_path: The path to clean and enriched AQI data
    @param output_file_path: The output file path
    """
    df_weather = pd.read_csv(weather_file_path, index_col=settings.DATE_WEATHER_COLUMN_NAME,
                             parse_dates=True)
    df_aqi = pd.read_csv(aqi_file_path, index_col=settings.DATE_COLUMN_NAME,
                         parse_dates=True)
    df_all = df_aqi.merge(df_weather, how='left', left_index=True, right_index=True)
    df_all.to_csv(output_file_path)
