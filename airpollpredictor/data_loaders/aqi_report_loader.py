# pylint: disable=E0401,  E0611, R0913, R0914, W0703

"""
Async loader of .csv files with concentrations of pollutants from fme.discomap.eea.europa.eu
Supports logging and re-downloading
"""

import datetime
import os
import pathlib
import time
import requests
from . import logger

URL_POLLUTANT_LIST = "https://fme.discomap.eea.europa.eu/fmedatastreaming/AirQualityDownload/" \
                     "AQData_Extract.fmw?CountryCode=$country$&CityName=$city$&Pollutant=" \
                     "$pollutant_id$&Year_from=$year_from$&Year_to=$year_to$&Station=" \
                     "$station$&Samplingpoint=&Source=All&Output=TEXT&UpdateDate=&TimeCoverage=Year"
RELOAD_COUNTER = 2
RELOAD_REPEATING_COUNTER = 5
POLLUTANTS_CODES_ALL = [7, 6001, 5, 10, 1, 8]


async def pollutants_txt_lists_load(save_dir_path: str,
                                    pollutant_codes: list = None,
                                    country=None,
                                    city=None,
                                    station=None,
                                    year_from=datetime.datetime.now().year,
                                    station_per_pollutant: dict = None,
                                    year_to=datetime.datetime.now().year) -> str:
    """
    Downloads and saves .txt files with urls to .csv files with concentrations of the pollutants
    @param save_dir_path: Path to directory for txt files with urls
    @param pollutant_codes: The list of pollutant identificator
    (8 - for NO2 etc.) (optional)
    @param country: Country 2-letters code (optional)
    @param city: The name of the city (optional)
    @param station: The code of the pollution control station (optional)
    @param year_from: The first year for data downloading
    @param station_per_pollutant: Dictionary with different stations lists
    for different pollutants (optional)
    @param year_to: The last year for data downloading
    @return: The path to saved .csv files with concentrations of the pollutants
    """

    save_path = __create_sub_dir(save_dir_path, "urls")
    if pollutant_codes is None:
        pollutant_codes = POLLUTANTS_CODES_ALL

    for pol_id in pollutant_codes:
        url = URL_POLLUTANT_LIST.replace("$pollutant_id$", str(pol_id))
        url = url.replace("$country$", ('' if country is None else country))
        if station_per_pollutant is not None:
            url = url.replace("$station$", ('' if station_per_pollutant[pol_id] is None
                                            else station_per_pollutant[pol_id]))
        elif station is not None and station != '':
            url = url.replace("$station$", station)
        url = url.replace("$city$", ('' if city is None else city))
        url = url.replace("$year_from$", str(year_from))
        url = url.replace("$year_to$", str(year_to))
        await __txt_file_load_and_save(f'{str(pol_id)}.txt', save_path, url)
    return save_path


async def __txt_file_load_and_save(file_name: str, save_path: str, url: str) -> None:
    print(f'Request TXT {url} sent')
    response = requests.get(url)
    print(f'Response TXT status_code: {response.status_code}')
    txt_file = str(response.content, encoding='UTF8')
    if response.status_code != 200:
        logger.log_error(save_path=save_path, text=txt_file, url=url)
        raise ConnectionError()
    file_path = os.path.join(save_path, f'{file_name}')
    with open(file_path, 'w', encoding='UTF8') as file_stream:
        file_stream.write(txt_file)
    print(f'File {file_name} saved')


async def csv_list_load(save_path: str, url_path: str) -> None:
    """
    Downloads .csv files by urls saved in .txt
    @param url_path: Path to .txt  with urls for .csv files
    @param save_path: Path to save .csv files
    @return:
    """
    txt_lists = list(pathlib.Path(url_path).glob('*.txt'))
    print(f'Loaded {len(txt_lists)} txt lists')
    repeated_reloads_for_dif_files = 0
    for txt_list in txt_lists:
        with open(txt_list, "r", encoding='utf-8-sig') as file_stream:
            txt_content = file_stream.read()
        csv_list = txt_content.split()
        print(f'Loaded {len(csv_list)} csv urls for file {txt_list}')
        pollutant_id = os.path.splitext(os.path.basename(txt_list))[0]
        sub_path = __create_sub_dir(save_path=save_path, sub_dir=pollutant_id)
        for csv_url in csv_list:
            file_name = csv_url.split('/')[-1:][0]
            for i in range(RELOAD_COUNTER + 1):
                if i == RELOAD_COUNTER:
                    repeated_reloads_for_dif_files += 1
                    logger.log_error(save_path=sub_path,
                              text=f"Reload attempt {repeated_reloads_for_dif_files}",
                              url=csv_url)
                    break
                try:
                    csv_file = await __load_csv_file(csv_url)
                    await __save_csv_file_from_str(
                        save_path=sub_path, file_name=file_name, csv_file=csv_file)
                    break
                except Exception:
                    print("SHORT SLEEP")
                    time.sleep(30)

            if repeated_reloads_for_dif_files == RELOAD_REPEATING_COUNTER:
                time.sleep(60 * 60)
                print("LONG LONG SLEEP")
                repeated_reloads_for_dif_files = 0


async def __load_csv_file(url: str) -> str:
    print(f'Request CSV {url} sent')
    response = requests.get(url, timeout=120)
    csv_file = str(response.content, encoding='UTF8')
    print(f'Response CSV status_code: {response.status_code}')
    return csv_file


async def __save_csv_file_from_str(save_path: str, file_name: str, csv_file: str) -> None:
    """
    Save string with csv file data to .csv file
    @param save_path: Path to the directory
    @param file_name: The required .csv file name
    @param csv_file: Dataframe for saving
    """
    file_path = os.path.join(save_path, f'{file_name}')
    with open(file_path, 'w', encoding='UTF8') as file_stream:
        file_stream.write(csv_file)
    print(f'File {file_name} saved')


def __create_sub_dir(save_path: str, sub_dir: str) -> str:
    """
    Create directory by the required path
    @param save_path: The path to the directory
    @param sub_dir: The name of the subdirectory
    @return: The full path to the created subdirectory
    """
    path = os.path.join(save_path, sub_dir)
    pathlib.Path(path).mkdir(exist_ok=True)
    return path
