# pylint: disable=E0401

import datetime
import requests


async def __post(url: str, params: dict):
    return requests.post(
        url=url,
        params={
            'save_dir_path': 'datasets_tests/weather-source-data/'
        },
        json=params
    )


async def load_weather_history_from_station():
    params = {'station_id': '06344'}
    response = await __post(url='http://127.0.0.1:8041/download_current_year', params=params)
    if response.status_code != 200:
        print("ERROR")
        print(response)
    print(response.url)


if __name__ == '__main__':
    # save_path = asyncio.run(aqi_loader.pollutants_txt_lists_load())

    # save_path = asyncio.run(
    #     aqi_loader.pollutants_txt_lists_load(
    #         pollutant_codes=[7, 6001, 5, 8], country="NL",
    #         city='Rotterdam', year_from=2015,
    #         station_per_pollutant={7: 'STA-NL00418', 5: 'STA-NL00418',
    #         6001: 'STA-NL00448', 8: 'STA-NL00418'}))
    # save_path =
    # "/home/alexna/work/projects/air_pollution_predict/
    # airpollpredictor/aqreport_loader/data/02_11_2022_12_12_44"

    # asyncio.run(aqi_loader.csv_list_load(save_path))
    #
    # SAVE_DIR_PATH = 'datasets_tests/weather-source-data/'
    # date_from = datetime.strptime("2015-01-01", "%Y-%m-%d").date()
    # date_from = datetime.strptime("2023-04-01", "%Y-%m-%d").date()
    # save_path = asyncio.run(weather_loader.load_weather_history_from_station(
    #     save_dir_path=SAVE_DIR_PATH, station="06344", date_from=date_from))

    # SAVE_DIR_PATH = 'datasets_tests/pollutants-source-data/'
    # pol_service.download_prev_years(SAVE_DIR_PATH)

    # asyncio.run(load_weather_history_from_station())

    YEAR_FROM = 2023
    DATE_FROM = str(datetime.date(year=YEAR_FROM, month=1, day=1))
    DATE_TO = datetime.datetime.now().date()
    print(DATE_FROM)
    print(type(DATE_TO))
