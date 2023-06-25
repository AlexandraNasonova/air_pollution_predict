"""
DVC Stage clean-pollutants - cleans loaded pollutants data
"""

# pylint: disable=E0401

from argparse import ArgumentParser
import yaml
import data_preprocessing.pollutants_cleaner as pol_cleaner
import data_preprocessing.merger as merger

STAGE = "clean-pollutants"


def __parse_args():
    parser = ArgumentParser(STAGE)
    parser.add_argument('--input_folder', required=True, help='Path to source data')
    parser.add_argument('--output_folder', required=True, help='Path to clean data')
    parser.add_argument('--params', required=True, help='Path to params')
    return parser.parse_args()


def __process_data(source_data_path: str, output_data_path: str, pol_codes: list[int]):
    df_list = merger.merge_dataframes_per_pollutant(
        source_data_path, pol_codes)
    pol_cleaner.drop_sampling_unverified_duplicates(pol_codes, df_list)
    pol_cleaner.convert_negative_values_to_nan(pol_codes, df_list)
    pol_cleaner.fix_non_hour_intervals(pol_codes, df_list)
    pol_cleaner.remove_unused_columns(pol_codes, df_list)
    merger.set_index_per_pollutant(pol_codes, df_list)
    merger.save_datasets_per_pollutant(pol_codes, df_list, output_data_path)


if __name__ == '__main__':
    print(f'Stage {STAGE} started')
    stage_args = __parse_args()
    with open(stage_args.params, 'r', encoding='UTF-8') as file_stream:
        pollutants_codes = yaml.safe_load(file_stream)["pollutants-codes"]
    __process_data(source_data_path=stage_args.input_folder,
                   output_data_path=stage_args.output_folder,
                   pol_codes=pollutants_codes)
    print(f'Stage {STAGE} finished')
