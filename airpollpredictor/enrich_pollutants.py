# pylint: disable=E0401

from argparse import ArgumentParser
import datetime
import yaml
import data_preprocessing.pollutants_enricher as pol_enrich

SECTION = "enrich-pollutants"


def parse_args(args_parser_name: str):
    """
    Parses command line args
    @param args_parser_name: Parser name
    @return: Parsed arguments
    """
    parser = ArgumentParser(args_parser_name)
    parser.add_argument('--input_folder', required=True, help='Path to clean data')
    parser.add_argument('--output_file', required=True, help='Path to enriched file')
    parser.add_argument('--params', required=True, help='Path to params')
    return parser.parse_args()


if __name__ == '__main__':
    stage_args = parse_args(SECTION)
    with open(stage_args.params, 'r', encoding='UTF-8') as file_stream:
        params_yaml = yaml.safe_load(file_stream)
        _pollutants_codes = params_yaml["pollutants-codes"]
        enrich_params = params_yaml["enrich-pollutants"]
        _lags_shift = enrich_params["lags-shifts"]
        _filters = enrich_params["filters"]
        _windows_filters_aqi = enrich_params["windows_filters_aqi"]
        _lag_agg_aqi = enrich_params["lag_agg_aqi"]
        _methods_agg_aqi = enrich_params["methods_agg_aqi"]
        _ewm_filters_aqi = enrich_params["ewm_filters_aqi"]
        _year_from = params_yaml["period-settings"]["year_start_train"]

    DATE_FROM = str(datetime.date(year=_year_from, month=1, day=1))
    DATE_TO = str(datetime.datetime.now().date())

    pol_enrich.generate_features(source_data_path=stage_args.input_folder,
                                 output_file=stage_args.output_file,
                                 pollutants_codes=_pollutants_codes,
                                 date_from=DATE_FROM,
                                 date_end=DATE_TO,
                                 lags_shift=_lags_shift,
                                 filters_aqi=_filters,
                                 windows_filters_aqi=_windows_filters_aqi,
                                 methods_agg_aqi=_methods_agg_aqi,
                                 lags_agg_aqi=_lag_agg_aqi,
                                 ewm_filters_aqi=_ewm_filters_aqi
                                 )
