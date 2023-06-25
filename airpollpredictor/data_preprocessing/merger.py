# pylint: disable=E0401, R0913, R0914, W0703
"""
Module mergers AQI and weather data
"""

import glob
import os
import pandas as pd
from settings import settings
from data_preprocessing.features_generations import ts_date_features_generator as date_gen


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
        print("FFFFF")
        print(settings.POL_USE_COLUMNS)
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


def __merge_column_by_index(pollutant_id: int, df_gen: pd.DataFrame, df_to_merge: pd.DataFrame,
                            source_column: str, new_column=None) -> pd.DataFrame:
    if new_column is None:
        new_column = source_column

    df_gen = df_gen.merge(df_to_merge[source_column], left_index=True, right_index=True)
    df_gen = df_gen.rename(
        columns={source_column: f'{new_column}_{settings.POL_NAMES[pollutant_id]}'})
    return df_gen


def __read_and_merge_pollutants(
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

    for pollutant_id in pollutants_codes:
        df_pollutant = pd.read_csv(os.path.join(source_data_path, f'{pollutant_id}.csv'),
                                   parse_dates=True, index_col=settings.DATE_COLUMN_NAME)
        df_pollutant = df_pollutant.tz_localize(None)
        df_gen = __merge_column_by_index(pollutant_id, df_gen, df_pollutant,
                                         settings.AQI_COLUMN_NAME)

    df_gen[settings.POLLUTANT_COLUMN_NAME] = df_gen.idxmax(axis=1) \
        .apply(lambda x: settings.POL_NAMES_REVERSE[x[x.index('_') + 1:]])
    # df_gen[settings.AQI_COLUMN_NAME] = df_gen.max(axis=1)
    return df_gen


def merge_and_save_aqi_per_pollutants_and_weather(
        aqi_prev_years_path: str,
        aqi_cur_year_path: str,
        weather_prev_years_file_path: str,
        weather_cur_year_file_path: str,
        pollutants_codes: list[int],
        date_prev_from: str, date_prev_end: str,
        date_cur_from: str, date_cur_end: str,
        output_file_path: str):
    """
    Merges AQI and weather data and saves result
    @param aqi_prev_years_path: The path to clean AQI data with prev_years
    @param aqi_cur_year_path: The path to clean AQI data with cur_year
    @param weather_prev_years_file_path: The path to clean weather data with prev_years
    @param weather_cur_year_file_path: The path to clean weather data with cur_year
    @param pollutants_codes: The list of pollutant codes
    @param date_prev_from: The first date of the data sources for previous years
    @param date_prev_end: The last date of the data sources for a current year
    @param date_cur_from: The first date of the data sources for previous years
    @param date_cur_end: The last date of the data sources for a current year
    @param output_file_path: The output file path
    """
    df_aqi_prev = __read_and_merge_pollutants(aqi_prev_years_path, pollutants_codes,
                                              date_prev_from, date_prev_end)
    df_aqi_cur = __read_and_merge_pollutants(aqi_cur_year_path, pollutants_codes,
                                             date_cur_from, date_cur_end)

    df_aqi = pd.concat([df_aqi_prev, df_aqi_cur])
    df_weather_prev_years = pd.read_csv(weather_prev_years_file_path,
                                        index_col=settings.DATE_WEATHER_COLUMN_NAME,
                                        parse_dates=True)
    df_weather_cur_year = pd.read_csv(weather_cur_year_file_path,
                                      index_col=settings.DATE_WEATHER_COLUMN_NAME,
                                      parse_dates=True)
    df_weather = pd.concat([df_weather_prev_years, df_weather_cur_year])
    df_all = df_aqi.merge(df_weather, how='left', left_index=True,
                          right_index=True)
    df_all.to_csv(output_file_path)


def merge_and_save_aqi_enriched_and_weather(
        aqi_prev_years_file_path: str,
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
    df_aqi_prev_years = pd.read_csv(aqi_prev_years_file_path,
                                    index_col=settings.DATE_COLUMN_NAME,
                                    parse_dates=True)
    df_aqi_cur_year = pd.read_csv(aqi_cur_year_file_path,
                                  index_col=settings.DATE_COLUMN_NAME,
                                  parse_dates=True)
    df_aqi = pd.concat([df_aqi_prev_years, df_aqi_cur_year])
    df_weather_prev_years = pd.read_csv(weather_prev_years_file_path,
                                        index_col=settings.DATE_WEATHER_COLUMN_NAME,
                                        parse_dates=True)
    df_weather_cur_year = pd.read_csv(weather_cur_year_file_path,
                                      index_col=settings.DATE_WEATHER_COLUMN_NAME,
                                      parse_dates=True)
    df_weather = pd.concat([df_weather_prev_years, df_weather_cur_year])
    df_all = df_weather.merge(df_aqi, how='left', left_index=True,
                              right_index=True)
    # df_all = date_gen.add_date_info(df_all)
    df_all.index.name = settings.DATE_COLUMN_NAME
    df_all.to_csv(output_file_path)
