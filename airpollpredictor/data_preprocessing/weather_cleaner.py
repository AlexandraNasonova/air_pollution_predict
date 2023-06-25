# pylint: disable=E0401

"""
Module for cleaning weather data
"""

import numpy as np
import pandas as pd
from settings import settings


def read_data(source_file_path: str, columns: list[str]):
    """
    Cleans weather data and saves the result
    @param source_file_path: The path to source weather file
    @param columns: Required weather columns
    """
    df_weather = pd.read_csv(source_file_path,
                             usecols=columns + [settings.DATE_WEATHER_COLUMN_NAME],
                             index_col=settings.DATE_WEATHER_COLUMN_NAME,
                             parse_dates=True)

    df_weather.reset_index(inplace=True)
    df_weather['date'] = pd.DatetimeIndex(df_weather['date'])
    df_weather.set_index('date', inplace=True, drop=True)
    return df_weather


def save_clean_data(df: pd.DataFrame, output_file_path: str):
    """
    Saves the clean dataset.
    @param df: Dataset to save
    @param output_file_path: The output path to weather file with clean data
    """
    df.to_csv(output_file_path)


def convert_wind(df: pd.DataFrame):
    """
    Converts wind spead and direction to projections
    @param df: Weather dataset
    """
    wv = df.pop("wspd")
    wd_rad = df.pop('wdir') * np.pi / 180
    df['Wx'] = wv * np.cos(wd_rad)
    df['Wy'] = wv * np.sin(wd_rad)
    max_wv = df['wpgt']
    df['max_Wx'] = max_wv*np.cos(wd_rad)
    df['max_Wy'] = max_wv*np.sin(wd_rad)
    df['wpgt'].mask(df['wpgt'] > 0, 1, inplace=True)


def fill_nan(df: pd.DataFrame):
    """
    Fills nans (required for DL)
    @param df: Weather dataset
    """
    df['prcp'].fillna(0, inplace=True)
    df['tmin'] = df['tmin'].interpolate(method='time')
    df['tmin'].fillna(0, inplace=True)
    df['Wx'].fillna(0, inplace=True)
    df['Wy'].fillna(0, inplace=True)
    df['pres'].fillna(0, inplace=True)
