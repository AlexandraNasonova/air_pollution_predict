import yaml
from argparse import ArgumentParser
import data_preprocessing.weather_cleaner as w_cleaner


def parse_args(args_parser_name: str):
    parser = ArgumentParser(args_parser_name)
    parser.add_argument('--input_file', required=True, help='Path to source data')
    parser.add_argument('--output_file', required=True, help='Path to save clean file')
    parser.add_argument('--params', required=True, help='Path to params')
    return parser.parse_args()


def process_data(source_file_path: str, output_file_path: str, columns: list[str]):
    w_cleaner.clean(source_file_path=source_file_path, output_file_path=output_file_path,
                    columns=columns)


if __name__ == '__main__':
    section = "clean-weather"
    stage_args = parse_args(section)
    with open(stage_args.params, 'r') as fp:
        columns = yaml.safe_load(fp)["weather-features"]
    process_data(source_file_path=stage_args.input_file,
                 output_file_path=stage_args.output_file,
                 columns=columns)
