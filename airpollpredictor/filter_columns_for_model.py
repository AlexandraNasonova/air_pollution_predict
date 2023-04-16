"""
DVC Stage filter_columns_for_model - filters columns of the full datasets \
and get only the ones required to the stage settings
"""
# pylint: disable=E0401

from argparse import ArgumentParser
import yaml
import pandas as pd
import data_preprocessing.columns_filter as col_filter
from settings import settings

STAGE = "filter_columns_for_model"


def __parse_args():
    parser = ArgumentParser(STAGE)
    parser.add_argument('--input_file', required=True, help='Path to input data')
    parser.add_argument('--output_file', required=True, help='Path to filtered data')
    parser.add_argument('--params', required=True, help='Path to params')
    parser.add_argument('--params_section', required=True, help='Section with filter params')
    return parser.parse_args()


if __name__ == '__main__':
    print(f'Stage {STAGE} started')

    stage_args = __parse_args()
    with open(stage_args.params, 'r', encoding='UTF-8') as file_stream:
        model_params = yaml.safe_load(file_stream)[stage_args.params_section]
    pol_id = model_params["pol_id"]
    columns_filters = model_params["columns_filters"]

    df_timeseries = pd.read_csv(stage_args.input_file, parse_dates=True,
                                index_col=settings.DATE_COLUMN_NAME)
    target_column_name = col_filter.get_target_column(
        prediction_value_type=model_params['prediction_value_type'], pol_id=pol_id)

    df_filtered = col_filter.filter_data_frame(
        df_timeseries=df_timeseries,
        pol_id=pol_id,
        pol_codes=columns_filters['pollutants_codes'],
        target_column_name=target_column_name,
        date_columns=columns_filters['date_columns'],
        weather_columns=columns_filters['weather_columns'],
        use_aqi_cols=columns_filters['use_aqi_cols'],
        use_c_mean_cols=columns_filters['use_c_mean_cols'],
        use_c_max_cols=columns_filters['use_c_max_cols'],
        use_c_median_cols=columns_filters['use_c_median_cols'],
        use_c_min_cols=columns_filters['use_c_min_cols'],
        use_pol_cols=columns_filters['use_pol_cols'],
        use_gen_lags_cols=columns_filters['use_gen_lags_cols'],
        use_lag_cols=columns_filters['use_lag_cols'],
        use_weather_cols=columns_filters['use_weather_cols']
    )

    df_filtered.to_csv(stage_args.output_file)
    print(f'Stage {STAGE} finished')
