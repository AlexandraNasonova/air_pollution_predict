# pylint: disable=E0401

from argparse import ArgumentParser
import yaml
import data_preprocessing.weather_cleaner as w_cleaner

SECTION = "clean-weather"


def parse_args(args_parser_name: str):
    """
    Parses command line args
    @param args_parser_name: Parser name
    @return: Parsed arguments
    """
    parser = ArgumentParser(args_parser_name)
    parser.add_argument('--input_file', required=True, help='Path to source data')
    parser.add_argument('--output_file', required=True, help='Path to save clean file')
    parser.add_argument('--params', required=True, help='Path to params')
    return parser.parse_args()


def process_data(source_file_path: str, output_file_path: str, features: list[str]):
    w_cleaner.clean(source_file_path=source_file_path, output_file_path=output_file_path,
                    columns=features)


if __name__ == '__main__':
    stage_args = parse_args(SECTION)
    with open(stage_args.params, 'r', encoding='UTF-8') as file_stream:
        weather_features = yaml.safe_load(file_stream)["weather-features"]
    process_data(source_file_path=stage_args.input_file,
                 output_file_path=stage_args.output_file,
                 features=weather_features)
