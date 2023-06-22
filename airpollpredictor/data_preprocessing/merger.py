# pylint: disable=E0401, R0913, R0914, W0703
"""
Module mergers AQI and weather data
"""

import glob
import os
import pandas as pd
from settings import settings


def read_and_merge_prev_and_cur(pollutants_codes: [int],
                                source_data_path_prev: str,
                                source_data_path_cur: str = None):
    """
    Reads files in folders for prev years and current year.
    One .csv file for each pollutant in every folder
    @param pollutants_codes: The list of pollutant codes
    @param source_data_path_prev: Path to folder with datasets
    for the previous years
    @param source_data_path_cur: Path to folder with datasets
    for the current year
    @return: List of merged datasets per pollutant
    """

    # read previous years - .csv per pollutant
    if pollutants_codes is None:
        pollutants_codes = list()
    df_list_prev = []
    for i in range(len(pollutants_codes)):
        path = os.path.join(source_data_path_prev, f'{str(pollutants_codes[i])}.csv')
        df_list_prev.append(pd.read_csv(
            path, parse_dates=True, index_col=settings.DATE_COLUMN_NAME))

    if not source_data_path_cur:
        return df_list_prev

    # read current years and merger with previous - optional
    df_list_all = []
    for i in range(len(pollutants_codes)):
        path = os.path.join(source_data_path_cur, f'{str(pollutants_codes[i])}.csv')
        df_cur = pd.read_csv(path,
                             parse_dates=True,
                             index_col=settings.DATE_COLUMN_NAME)
        df_f = pd.concat([df_list_prev[i], df_cur])
        df_f.reset_index(inplace=True)
        df_f.set_index(settings.DATE_COLUMN_NAME, inplace=True)
        df_f.sort_index(inplace=True)
        df_list_all.append(df_f)

    return df_list_all


def cut_dataset_from_date(pollutants_codes: [int],
                          df_list: list[pd.DataFrame],
                          from_date):
    """
    Cuts datasets in the list from the required date
    @param pollutants_codes:
    @param df_list: List of the datasets per pollutant
    @param from_date: Start date to cut from
    @return: List of the cut datasets per pollutant
    """
    for i in range(len(pollutants_codes)):
        df_list[i] = df_list[i][from_date:]
        df_list[i].sort_index(inplace=True)


def merge_dataframes_per_pollutant(source_data_path: str, pollutants_codes: [int]):
    """
    Merges files for pollutants (mergers all years to one file for every pollutant)
    @param source_data_path: The path to the source pollutant data
    @param pollutants_codes: The list of pollutant codes
    @return: List of dataframes with merged for every pollutant data
    """
    df_list = []
    for pol_id in pollutants_codes:
        df_pol = pd.concat(map(lambda p: pd.read_csv(p, usecols=settings.POL_USE_COLUMNS),
                               glob.glob(os.path.join(source_data_path, str(pol_id), "*.csv"))))
        # print(f'Pollutant: {settings.POL_NAMES[pol_id] :10} Lines count: {df.shape[0]}')
        df_list.append(df_pol)
    return df_list


def set_index_per_pollutant(pollutants_codes: list[int], df_list: list[pd.DataFrame]):
    """
    Reset date index
    @param pollutants_codes: The list of pollutant codes
    @param df_list: List of dataframes with merged for every pollutant data
    """
    for i in range(len(pollutants_codes)):
        df_list[i].set_index(settings.DATE_COLUMN_NAME, inplace=True)
        df_list[i].sort_index(inplace=True)


def save_datasets_per_pollutant(pollutants_codes: list[int],
                                df_list: list[pd.DataFrame],
                                output_path: str):
    """
    Saves datasets in .csv per pollutant
    @param pollutants_codes: The list of pollutant codes
    @param df_list: List of dataframes with merged for every pollutant data
    @param output_path: The output path
    """
    # pylint: disable=C0200
    for i in range(len(pollutants_codes)):
        file_path = os.path.join(output_path, f'{pollutants_codes[i]}.csv')
        df_list[i].to_csv(file_path)


def __read_and_merge_aqi_files(aqi_prev_years_file_path: str,
                               aqi_cur_year_file_path: str):
    """
    Reads and merges AQI api previous and current years datasets
    @param aqi_prev_years_file_path: The path to clean and enriched AQI data with prev_years
    @param aqi_cur_year_file_path: The path to clean and enriched AQI data with cur_year
    """
    df_aqi_prev_years = pd.read_csv(aqi_prev_years_file_path,
                                    index_col=settings.DATE_COLUMN_NAME,
                                    parse_dates=True)
    df_aqi_cur_year = pd.read_csv(aqi_cur_year_file_path,
                                  index_col=settings.DATE_COLUMN_NAME,
                                  parse_dates=True)
    df_aqi = pd.concat([df_aqi_prev_years, df_aqi_cur_year])
    return df_aqi


def __read_and_merge_weather_files(weather_prev_years_file_path: str,
                                   weather_cur_year_file_path: str):
    """
    Reads and merges weather api previous and current years datasets
    @param weather_prev_years_file_path: The path to clean weather data with prev_years
    @param weather_cur_year_file_path: The path to clean weather data with cur_year
    """
    df_weather_prev_years = pd.read_csv(weather_prev_years_file_path,
                                        index_col=settings.DATE_WEATHER_COLUMN_NAME,
                                        parse_dates=True)
    df_weather_cur_year = pd.read_csv(weather_cur_year_file_path,
                                      index_col=settings.DATE_WEATHER_COLUMN_NAME,
                                      parse_dates=True)
    df_weather = pd.concat([df_weather_prev_years, df_weather_cur_year])
    return df_weather


def merge_and_save_aqi_and_weather(aqi_prev_years_file_path: str,
                                   aqi_cur_year_file_path: str,
                                   weather_prev_years_file_path: str,
                                   weather_cur_year_file_path: str,
                                   output_file_path: str):
    """
    Merges AQI and weather data and saves result
    @param aqi_prev_years_file_path: The path to clean and enriched AQI data with prev_years
    @param aqi_cur_year_file_path: The path to clean and enriched AQI data with cur_year
    @param weather_prev_years_file_path: The path to clean weather data with prev_years
    @param weather_cur_year_file_path: The path to clean weather data with cur_year
    @param output_file_path: The output file path
    """
    df_aqi = __read_and_merge_aqi_files(aqi_prev_years_file_path,
                                        aqi_cur_year_file_path)
    df_weather = __read_and_merge_weather_files(weather_prev_years_file_path,
                                                weather_cur_year_file_path)
    df_all = df_aqi.merge(df_weather, how='left', left_index=True,
                          right_index=True)
    df_all.to_csv(output_file_path)
