"""
DVC Stage clean-weather - cleans loaded weather data
"""

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


if __name__ == '__main__':
    print(f'Stage {STAGE} started')
    stage_args = __parse_args()
    with open(stage_args.params, 'r', encoding='UTF-8') as file_stream:
        weather_features = yaml.safe_load(file_stream)["weather-features"]

    df_weather = w_cleaner.read_data(source_file_path=stage_args.input_file, columns=weather_features)
    w_cleaner.convert_wind(df_weather)
    w_cleaner.fill_nan(df_weather)
    w_cleaner.save_clean_data(df_weather, output_file_path=stage_args.output_file)
    w_cleaner.save_clean_data(df_weather, output_file_path=stage_args.output_file)
    print(f'Stage {STAGE} finished')
