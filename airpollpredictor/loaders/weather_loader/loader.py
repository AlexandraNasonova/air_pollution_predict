# pylint: disable=E0401, R0913, R0914, W0703

"""
Async loader of historical data about weather in .json format via meteostat API
"""

import datetime
import requests
import pandas as pd
from loaders.logger import log_error
from loaders.file_saver import create_local_data_dir, save_csv_file_from_dataframe

URL_DAILY_HIST = "https://meteostat.p.rapidapi.com/stations/daily"
RAPID_API_KEY = '636601abdamsheec399674665a87p1878bfjsnf16635dcc483'
RAPID_API_HOST = 'meteostat.p.rapidapi.com'


async def __get(station: str, date_from, date_end):
    return requests.get(
        URL_DAILY_HIST,
        params={
            'station': station,
            'start': date_from,
            'end': date_end
        },
        headers={
            'X-RapidAPI-Key': RAPID_API_KEY,
            'X-RapidAPI-Host': RAPID_API_HOST
        }
    )


async def load_weather_history_from_station(station: str, date_from: datetime.date,
                                            date_end=datetime.datetime.now().date()) -> str:
    """
    Downloads historical weather data via Meteostat API
    @param station: The code of the station
    @param date_from: The first date of the historical period
    @param date_end: The last date of the historical period
    @return: The path to saved .csv file with historical weather data
    """
    save_path = create_local_data_dir("data_weather")
    print(f"Directory {save_path} created")
    response = await __get(station, date_from, date_end)
    if response.status_code != 200:
        print(response.url)
        log_error(save_path, text=response.text, error_reason=response.reason,
                  station=station, date_from=date_from, date_end=date_end, url=response.url)
        raise ConnectionError()

    df_result = __convert_json_to_pandas(response.json())
    await save_csv_file_from_dataframe(save_path, file_name=f'{station}.csv', df_to_save=df_result)
    return save_path


def __convert_json_to_pandas(response_json) -> pd.DataFrame:
    if 'data' not in response_json:
        raise AttributeError()
    df_weather = pd.DataFrame(data=response_json['data'])
    df_weather['date'] = pd.to_datetime(df_weather['date'], format='%Y-%m-%d')
    df_weather.set_index('date', inplace=True)
    return df_weather
