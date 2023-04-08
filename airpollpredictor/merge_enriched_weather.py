import yaml
from argparse import ArgumentParser
import data_preprocessing.merger as merger


def parse_args(args_parser_name: str):
    parser = ArgumentParser(args_parser_name)
    parser.add_argument('--input_weather_file', required=True, help='Path to clean weather data')
    parser.add_argument('--input_aqi_file', required=True, help='Path to AQI file')
    parser.add_argument('--output_file', required=True, help='Path to save clean file')
    parser.add_argument('--params', required=True, help='Path to params')
    return parser.parse_args()


def process_data(weather_file_path: str, aqi_file_path: str, output_file_path: str):
    merger.merge(weather_file_path=weather_file_path, aqi_file_path=aqi_file_path,
                 output_file_path=output_file_path)


if __name__ == '__main__':
    section = "merge_enriched_weather"
    stage_args = parse_args(section)
    with open(stage_args.params, 'r') as fp:
        columns = yaml.safe_load(fp)["weather-features"]
    process_data(weather_file_path=stage_args.input_weather_file,
                 aqi_file_path=stage_args.input_aqi_file,
                 output_file_path=stage_args.output_file)
