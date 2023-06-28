"""
DVC Stage calculate-aqi - calculates AQI fro loaded and cleaned data
"""
# pylint: disable=E0401

from argparse import ArgumentParser
import yaml
import data_preprocessing.aqi_estimator as aqi_calc
import data_preprocessing.merger as merger

STAGE = "calculate-aqi"


def __parse_args():
    parser = ArgumentParser(STAGE)
    parser.add_argument('--input_folder_prev', required=True,
                        help='Path to AQI file with prev_years')
    parser.add_argument('--input_folder_cur', required=False,
                        help='Path to AQI file with cur_year')
    parser.add_argument('--output_folder', required=True, help='Path to save files with AQI')
    parser.add_argument('--params', required=True, help='Path to params')
    return parser.parse_args()


if __name__ == '__main__':
    print(f'Stage {STAGE} started')
    stage_args = __parse_args()
    pollutant_codes: list
    with open(stage_args.params, 'r', encoding='UTF-8') as file_stream:
        params = yaml.safe_load(file_stream)
        pollutants_codes = params["pollutants-codes"]
        date_current_year_start = params["period-settings"]["date_current_year_start"]
    df_conc_list = merger.read_and_merge_prev_and_cur(
        pollutants_codes, stage_args.input_folder_prev,
        stage_args.input_folder_cur)

    df_aqi_list = aqi_calc.calculate_aqi(df_conc_list, pollutants_codes)
    if stage_args.input_folder_cur:
        merger.cut_dataset_from_date(pollutants_codes, df_aqi_list,
                                     date_current_year_start)
    merger.save_datasets_per_pollutant(pollutants_codes, df_aqi_list,
                                       stage_args.output_folder)
    print(f'Stage {STAGE} finished')
