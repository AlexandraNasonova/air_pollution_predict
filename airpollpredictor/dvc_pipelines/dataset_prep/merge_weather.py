"""
DVC Stage merge_enriched_weather - merges enriched pollutants data and weather data
"""
# pylint: disable=E0401

from argparse import ArgumentParser
import yaml
from data_preprocessing import merger

STAGE = "merge_weather"


def __parse_args():
    parser = ArgumentParser(STAGE)
    parser.add_argument('--input_weather_prev_years_file', required=True,
                        help='Path to clean weather data with prev_years')
    parser.add_argument('--input_weather_cur_year_file', required=True,
                        help='Path to clean weather data with cur_year')
    parser.add_argument('--input_aqi_prev_years_folder', required=True,
                        help='Path to AQI file with prev_years')
    parser.add_argument('--input_aqi_cur_year_folder', required=True,
                        help='Path to AQI file with cur_year')
    parser.add_argument('--output_file', required=True, help='Path to save clean file')
    parser.add_argument('--params', required=True, help='Path to params')
    return parser.parse_args()


if __name__ == '__main__':
    print(f'Stage {STAGE} started')
    stage_args = __parse_args()
    with open(stage_args.params, 'r', encoding='UTF-8') as file_stream:
        params_yaml = yaml.safe_load(file_stream)
    period_params = params_yaml["period-settings"]

    df_aqi = merger.merge_and_save_aqi_per_pollutants_and_weather(
        aqi_prev_years_path=stage_args.input_aqi_prev_years_folder,
        aqi_cur_year_path=stage_args.input_aqi_cur_year_folder,
        weather_prev_years_file_path=stage_args.input_weather_prev_years_file,
        weather_cur_year_file_path=stage_args.input_weather_cur_year_file,
        pollutants_codes=params_yaml["pollutants-codes"],
        date_prev_from=period_params["date_start_train"],
        date_prev_end=period_params["date_prev_years_end"],
        date_cur_from=period_params["date_current_year_start"],
        date_cur_end=period_params["current_date"],
        output_file_path=stage_args.output_file
    )
    print(f'Stage {STAGE} finished')
