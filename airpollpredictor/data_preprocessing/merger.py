# pylint: disable=E0401, R0913, R0914, W0703
"""
Module mergers AQI and weather data
"""

import pandas as pd
from settings import settings


def merge(weather_prev_years_file_path: str,
          weather_cur_year_file_path: str,
          aqi_prev_years_file_path: str,
          aqi_cur_year_file_path: str,
          output_file_path: str):
    """
    Merges AQI and weather data and saves result
    @param weather_prev_years_file_path: The path to clean weather data with prev_years
    @param weather_cur_year_file_path: The path to clean weather data with cur_year
    @param aqi_prev_years_file_path: The path to clean and enriched AQI data with prev_years
    @param aqi_cur_year_file_path: The path to clean and enriched AQI data with cur_year
    @param output_file_path: The output file path
    """
    df_weather_prev_years = pd.read_csv(weather_prev_years_file_path,
                                        index_col=settings.DATE_WEATHER_COLUMN_NAME,
                                        parse_dates=True)
    df_weather_cur_year = pd.read_csv(weather_cur_year_file_path,
                                      index_col=settings.DATE_WEATHER_COLUMN_NAME,
                                      parse_dates=True)
    df_aqi_prev_years = pd.read_csv(aqi_prev_years_file_path,
                                    index_col=settings.DATE_COLUMN_NAME,
                                    parse_dates=True)
    df_aqi_cur_year = pd.read_csv(aqi_cur_year_file_path,
                                  index_col=settings.DATE_COLUMN_NAME,
                                  parse_dates=True)
    df_weather = pd.concat([df_weather_prev_years, df_weather_cur_year])
    df_aqi = pd.concat([df_aqi_prev_years, df_aqi_cur_year])
    df_all = df_aqi.merge(df_weather, how='left', left_index=True, right_index=True)
    df_all.to_csv(output_file_path)
