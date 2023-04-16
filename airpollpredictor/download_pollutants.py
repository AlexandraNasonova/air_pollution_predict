"""
DVC Stage download-pollutants - downloads pollutants data
"""

# pylint: disable=E0401

from argparse import ArgumentParser
import asyncio
import datetime
import yaml
import data_loaders.aqi_report_loader as pol_loader


STAGE = "download-pollutants"


def __parse_args():
    parser = ArgumentParser(STAGE)
    parser.add_argument('--output_folder', required=True, help='Path to save files file')
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

    urls_path = asyncio.run(
        pol_loader.pollutants_txt_lists_load(
            save_dir_path=stage_args.output_folder,
            year_from=date_from.year,
            year_to=date_to.year,
            pollutant_codes=list(params["stations_per_pollutants"].keys()),
            country=params["country_code"],
            city=params["city"],
            station_per_pollutant=params["stations_per_pollutants"]))
    asyncio.run(pol_loader.csv_list_load(stage_args.output_folder, urls_path))
    print(f'Stage {STAGE} finished')
