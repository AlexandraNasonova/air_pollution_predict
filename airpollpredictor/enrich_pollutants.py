# pylint: disable=E0401

from argparse import ArgumentParser
import datetime
import yaml
import data_preprocessing.pollutants_enricher as pol_enrich

STAGE = "enrich-pollutants"


def __parse_args():
    parser = ArgumentParser(STAGE)
    parser.add_argument('--input_folder', required=True, help='Path to clean data')
    parser.add_argument('--output_file', required=True, help='Path to enriched file')
    parser.add_argument('--params', required=True, help='Path to params')
    return parser.parse_args()


if __name__ == '__main__':
    stage_args = __parse_args()
    with open(stage_args.params, 'r', encoding='UTF-8') as file_stream:
        params_yaml = yaml.safe_load(file_stream)

    pollutants_codes = params_yaml["pollutants-codes"]
    enrich_params = params_yaml["enrich-pollutants"]
    period_settings = params_yaml["period-settings"]

    lags_shift = enrich_params["lags-shifts"]
    filters = enrich_params["filters"]
    windows_filters_aqi = enrich_params["windows_filters_aqi"]
    lag_agg_aqi = enrich_params["lag_agg_aqi"]
    methods_agg_aqi = enrich_params["methods_agg_aqi"]
    ewm_filters_aqi = enrich_params["ewm_filters_aqi"]
    year_from = period_settings["year_start_train"]

    DATE_FROM = str(datetime.date(year=year_from, month=1, day=1))
    DATE_TO = str(datetime.datetime.now().date())

    pol_enrich.generate_features(source_data_path=stage_args.input_folder,
                                 output_file=stage_args.output_file,
                                 pollutants_codes=pollutants_codes,
                                 date_from=DATE_FROM,
                                 date_end=DATE_TO,
                                 lags_shift=lags_shift,
                                 filters_aqi=filters,
                                 windows_filters_aqi=windows_filters_aqi,
                                 methods_agg_aqi=methods_agg_aqi,
                                 lags_agg_aqi=lag_agg_aqi,
                                 ewm_filters_aqi=ewm_filters_aqi
                                 )
