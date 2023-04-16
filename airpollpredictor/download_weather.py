"""
DVC Stage download-pollutants - downloads weather data
"""

# pylint: disable=E0401, C0414

from argparse import ArgumentParser
import asyncio
import datetime
import yaml
import data_loaders.weather_loader as weather_loader

STAGE = "download-weather"


def __parse_args():
    parser = ArgumentParser(STAGE)
    parser.add_argument('--output_file', required=True, help='Path to save file')
    parser.add_argument('--params', required=True, help='Path to params')
    parser.add_argument('--params_section', required=True, help='Section with params')
    return parser.parse_args()


if __name__ == '__main__':
    # noinspection DuplicatedCode
    print(f'Stage {STAGE} started')
    stage_args = __parse_args()
    with open(stage_args.params, 'r', encoding='UTF-8') as file_stream:
        params_yaml = yaml.safe_load(file_stream)
    params = params_yaml[stage_args.params_section]

    date_from = datetime.datetime.strptime(params["date_from"], "%Y-%m-%d").date()
    date_to = datetime.datetime.strptime(params["date_to"], "%Y-%m-%d").date()

    asyncio.run(weather_loader.load_weather_history_from_station(
        save_file_path=stage_args.output_file, station=params["station_id"],
        date_from=date_from, date_end=date_to))
    print(f'Stage {STAGE} finished')
