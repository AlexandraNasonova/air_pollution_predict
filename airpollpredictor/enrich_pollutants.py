import datetime
import yaml
from argparse import ArgumentParser
import data_preprocessing.pollutants_enricher as pol_enrich


# sys.path.insert(0, '/home/alexandra/work/projects/air_pol/air_pollution_predict/airpollpredictor/data_preprocessing')
# sys.path.insert(0, '/home/alexandra/work/projects/air_pol/air_pollution_predict/airpollpredictor')


def parse_args(args_parser_name: str):
    parser = ArgumentParser(args_parser_name)
    parser.add_argument('--input_folder', required=True, help='Path to clean data')
    parser.add_argument('--output_file', required=True, help='Path to enriched file')
    parser.add_argument('--params', required=True, help='Path to params')
    return parser.parse_args()


if __name__ == '__main__':
    section = "enrich-pollutants"
    stage_args = parse_args(section)
    with open(stage_args.params, 'r') as fp:
        params_yaml = yaml.safe_load(fp)
        _pollutants_codes = params_yaml["pollutants-codes"]
        enrich_params = params_yaml["enrich-pollutants"]
        _lags_shift = enrich_params["lags-shifts"]
        _filters = enrich_params["filters"]
        _windows_filters_aqi = enrich_params["windows_filters_aqi"]
        _lag_agg_aqi = enrich_params["lag_agg_aqi"]
        _methods_agg_aqi = enrich_params["methods_agg_aqi"]
        _ewm_filters_aqi = enrich_params["ewm_filters_aqi"]
        _year_from = params_yaml["period-settings"]["year_start_train"]

    _date_from = str(datetime.date(year=_year_from, month=1, day=1))
    _date_to = str(datetime.datetime.now().date())

    pol_enrich.generate_features(source_data_path=stage_args.input_folder,
                                 output_file=stage_args.output_file,
                                 pollutants_codes=_pollutants_codes,
                                 date_from=_date_from,
                                 date_end=_date_to,
                                 lags_shift=_lags_shift,
                                 filters_aqi=_filters,
                                 windows_filters_aqi=_windows_filters_aqi,
                                 methods_agg_aqi=_methods_agg_aqi,
                                 lags_agg_aqi=_lag_agg_aqi,
                                 ewm_filters_aqi=_ewm_filters_aqi
                                 )
