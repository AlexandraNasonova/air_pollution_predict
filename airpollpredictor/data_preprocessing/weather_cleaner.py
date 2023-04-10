# pylint: disable=E0401

"""
Module for cleaning weather data
"""

import pandas as pd
from . import settings


def clean(source_file_path: str, columns: list[str], output_file_path: str):
    """
    Cleans weather data and saves the result
    @param source_file_path: The path to source weather file
    @param columns: Required weather columns
    @param output_file_path: The output path to weather file with clean data
    """
    df_weather = pd.read_csv(source_file_path,
                             usecols=columns + [settings.DATE_WEATHER_COLUMN_NAME],
                             index_col=settings.DATE_WEATHER_COLUMN_NAME,
                             parse_dates=True)
    df_weather.to_csv(output_file_path)
