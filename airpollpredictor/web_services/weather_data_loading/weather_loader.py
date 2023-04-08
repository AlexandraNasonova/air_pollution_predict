# pylint: disable=E0401, R0913, R0914, W0703

"""
Async loader of historical data about weather in .json format via meteostat API
"""

import datetime
import os

import requests
import pandas as pd
from web_services.logger import log_error

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


async def load_weather_history_from_station(save_file_path: str,
                                            station: str,
                                            date_from: datetime.date,
                                            date_end=datetime.datetime.now().date()):
    """
    Downloads historical weather data via Meteostat API
    @param save_file_path: Path to file to save weather data
    @param station: The code of the station
    @param date_from: The first date of the historical period
    @param date_end: The last date of the historical period
    """
    response = await __get(station, date_from, date_end)
    if response.status_code != 200:
        print(response.url)
        log_error(os.path.dirname(save_file_path), text=response.text, error_reason=response.reason,
                  station=station, date_from=date_from, date_end=date_end, url=response.url)
        raise ConnectionError()

    df_result = __convert_json_to_pandas(response.json())
    df_result.to_csv(save_file_path)


def __convert_json_to_pandas(response_json) -> pd.DataFrame:
    if 'data' not in response_json:
        raise AttributeError()
    df_weather = pd.DataFrame(data=response_json['data'])
    df_weather['date'] = pd.to_datetime(df_weather['date'], format='%Y-%m-%d')
    df_weather.set_index('date', inplace=True)
    return df_weather
