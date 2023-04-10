# pylint: disable=E0401

from argparse import ArgumentParser
import yaml
import data_preprocessing.pollutants_cleaner as pol_cleaner

SECTION = "clean-pollutants"


def parse_args(args_parser_name: str):
    """
    Parses command line args
    @param args_parser_name: Parser name
    @return: Parsed arguments
    """
    parser = ArgumentParser(args_parser_name)
    parser.add_argument('--input_folder', required=True, help='Path to source data')
    parser.add_argument('--output_folder', required=True, help='Path to clean data')
    parser.add_argument('--params', required=True, help='Path to params')
    return parser.parse_args()


def process_data(source_data_path: str, output_data_path: str, pollutants_codes: list[int]):
    df_list = pol_cleaner.get_merged_dataframes_per_pollutant(
        source_data_path, pollutants_codes)
    pol_cleaner.drop_sampling_unverified_duplicates(pollutants_codes, df_list)
    pol_cleaner.convert_negative_values_to_nan(pollutants_codes, df_list)
    pol_cleaner.fix_non_hour_intervals(pollutants_codes, df_list)
    pol_cleaner.remove_unused_columns(pollutants_codes, df_list)
    pol_cleaner.set_index(pollutants_codes, df_list)
    pol_cleaner.save_clean_data(pollutants_codes, df_list, output_data_path)


if __name__ == '__main__':
    stage_args = parse_args(SECTION)
    with open(stage_args.params, 'r', encoding='UTF-8') as file_stream:
        pollutant_codes = yaml.safe_load(file_stream)["pollutants-codes"]
    process_data(source_data_path=stage_args.input_folder,
                 output_data_path=stage_args.output_folder,
                 pollutants_codes=pollutant_codes)