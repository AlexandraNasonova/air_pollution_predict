"""
DVC Stage enrich-pollutants - add date and lag features to cleaned pollutants datas
"""

# pylint: disable=E0401

from argparse import ArgumentParser
import datetime

import pandas as pd
import yaml
import data_preprocessing.pollutants_enricher as pol_enrich

STAGE = "enrich-pollutants"


def __parse_args():
    parser = ArgumentParser(STAGE)
    parser.add_argument('--input_prev_years_folder', required=True,
                        help='Path to clean data prev_years')
    parser.add_argument('--input_cur_year_folder', required=False,
                        help='Path to clean data cur_year')
    parser.add_argument('--output_file', required=True, help='Path to enriched file')
    parser.add_argument('--params', required=True, help='Path to params')
    parser.add_argument('--params_section', required=True, help='Section with params')

    return parser.parse_args()


if __name__ == '__main__':
    print(f'Stage {STAGE} started')

    stage_args = __parse_args()
    with open(stage_args.params, 'r', encoding='UTF-8') as file_stream:
        params_yaml = yaml.safe_load(file_stream)

    pollutants_codes = params_yaml["pollutants-codes"]
    params = params_yaml[stage_args.params_section]
    enrich_params = params['params']
    date_prev_from = params['date_prev_from']
    date_prev_to = params['date_prev_to']

    df_aqi: pd.DataFrame
    date_save_from = datetime.datetime.strptime(date_prev_from, "%Y-%m-%d").date()

    # only previous data

    df_aqi = pol_enrich.calc_aqi_and_mean_concentration_and_merge(
        source_data_path=stage_args.input_prev_years_folder, pollutants_codes=pollutants_codes,
        date_from=date_prev_from, date_end=date_prev_to)

    # current data + 100 days of previous data to calculate lags
    if stage_args.input_cur_year_folder is not None:
        date_cur_from = params['date_cur_from']
        date_cur_to = params['date_cur_to']
        df_aqi_cur = pol_enrich.calc_aqi_and_mean_concentration_and_merge(
            source_data_path=stage_args.input_cur_year_folder, pollutants_codes=pollutants_codes,
            date_from=date_cur_from, date_end=date_cur_to)
        df_aqi = pd.concat([df_aqi, df_aqi_cur])
        date_cur_from_d = datetime.datetime.strptime(date_cur_from, "%Y-%m-%d").date()
        date_cut = (date_cur_from_d - datetime.timedelta(days=100)).strftime("%Y-%m-%d")
        df_aqi = df_aqi[date_cut:]
        date_save_from = date_cur_from_d

    df_aqi = pol_enrich.generate_features(df_aqi_mean=df_aqi,
                                          pollutants_codes=pollutants_codes,
                                          lags_shift=enrich_params["lags-shifts"],
                                          filters_aqi=enrich_params["filters"],
                                          windows_filters_aqi=enrich_params["windows_filters_aqi"],
                                          methods_agg_aqi=enrich_params["methods_agg_aqi"],
                                          lags_agg_aqi=enrich_params["lag_agg_aqi"],
                                          ewm_filters_aqi=enrich_params["ewm_filters_aqi"]
                                          )
    df_aqi = df_aqi[date_save_from:]
    df_aqi.to_csv(stage_args.output_file)

    print(f'Stage {STAGE} finished')
