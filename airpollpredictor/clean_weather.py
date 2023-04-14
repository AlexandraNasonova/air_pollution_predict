# pylint: disable=E0401

from argparse import ArgumentParser
import yaml
import data_preprocessing.weather_cleaner as w_cleaner

STAGE = "clean-weather"


def __parse_args():
    parser = ArgumentParser(STAGE)
    parser.add_argument('--input_file', required=True, help='Path to source data')
    parser.add_argument('--output_file', required=True, help='Path to clean data')
    parser.add_argument('--params', required=True, help='Path to params')
    return parser.parse_args()


def __process_data(source_file_path: str, output_file_path: str, features: list[str]):
    w_cleaner.clean(source_file_path=source_file_path, output_file_path=output_file_path,
                    columns=features)


if __name__ == '__main__':
    stage_args = __parse_args()
    with open(stage_args.params, 'r', encoding='UTF-8') as file_stream:
        weather_features = yaml.safe_load(file_stream)["weather-features"]
    __process_data(source_file_path=stage_args.input_file,
                   output_file_path=stage_args.output_file,
                   features=weather_features)
