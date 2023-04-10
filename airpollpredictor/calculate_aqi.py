# pylint: disable=E0401

from argparse import ArgumentParser
import yaml
import data_preprocessing.aqi_estimator as aqi_calc

SECTION = "calculate-aqi"


def parse_args(args_parser_name: str):
    """
    Parses command line args
    @param args_parser_name: Parser name
    @return: Parsed arguments
    """
    parser = ArgumentParser(args_parser_name)
    parser.add_argument('--input_folder', required=True, help='Path to clean data')
    parser.add_argument('--output_folder', required=True, help='Path to aqi data')
    parser.add_argument('--params', required=True, help='Path to params')
    return parser.parse_args()


def process_data(source_data_path: str, output_data_path: str, pollutants_codes: list[int]):
    df_aqi_list = aqi_calc.calculate_aqi_indexes(source_data_path, pollutants_codes)
    aqi_calc.save_aqi_data(pollutants_codes, df_aqi_list, output_data_path)


if __name__ == '__main__':
    stage_args = parse_args(SECTION)
    with open(stage_args.params, 'r', encoding='UTF-8') as file_stream:
        pollutant_codes = yaml.safe_load(file_stream)["pollutants-codes"]
    process_data(stage_args.input_folder, stage_args.output_folder, pollutant_codes)
