import yaml
from argparse import ArgumentParser
import data_preprocessing.aqi_estimator as aqi_calc


# sys.path.insert(0, '/home/alexandra/work/projects/air_pol/air_pollution_predict/airpollpredictor/data_preprocessing')
# sys.path.insert(0, '/home/alexandra/work/projects/air_pol/air_pollution_predict/airpollpredictor')

def parse_args(args_parser_name: str):
    parser = ArgumentParser(args_parser_name)
    parser.add_argument('--input_folder', required=True, help='Path to clean data')
    parser.add_argument('--output_folder', required=True, help='Path to aqi data')
    parser.add_argument('--params', required=True, help='Path to params')
    return parser.parse_args()


def process_data(source_data_path: str, output_data_path: str, pollutants_codes: list[int]):
    df_aqi_list = aqi_calc.calculate_aqi_indexes(source_data_path, pollutants_codes)
    aqi_calc.save_aqi_data(pollutants_codes, df_aqi_list, output_data_path)


if __name__ == '__main__':
    section = "calculate-aqi"
    stage_args = parse_args(section)
    with open(stage_args.params, 'r') as fp:
        pollutant_codes = yaml.safe_load(fp)["pollutants-codes"]
    process_data(stage_args.input_folder, stage_args.output_folder, pollutant_codes)
