# pylint: disable=E0401

from argparse import ArgumentParser
from data_preprocessing import merger

STAGE = "merge_enriched_weather"


def __parse_args():
    parser = ArgumentParser(STAGE)
    parser.add_argument('--input_weather_prev_years_file', required=True,
                        help='Path to clean weather data with prev_years')
    parser.add_argument('--input_weather_cur_year_file', required=True,
                        help='Path to clean weather data with cur_year')
    parser.add_argument('--input_aqi_prev_years_file', required=True,
                        help='Path to AQI file with prev_years')
    parser.add_argument('--input_aqi_cur_year_file', required=True,
                        help='Path to AQI file with cur_year')
    parser.add_argument('--output_file', required=True, help='Path to save clean file')
    parser.add_argument('--params', required=True, help='Path to params')
    return parser.parse_args()


if __name__ == '__main__':
    stage_args = __parse_args()
    merger.merge(weather_prev_years_file_path=stage_args.input_weather_prev_years_file,
                 weather_cur_year_file_path=stage_args.input_weather_cur_year_file,
                 aqi_prev_years_file_path=stage_args.input_aqi_prev_years_file,
                 aqi_cur_year_file_path=stage_args.input_aqi_cur_year_file,
                 output_file_path=stage_args.output_file)

