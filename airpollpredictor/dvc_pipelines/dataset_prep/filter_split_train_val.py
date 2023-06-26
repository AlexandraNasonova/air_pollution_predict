"""
DVC Stage filter_split_train_val - filter columns and splits data to train and val, then saves
"""
# pylint: disable=E0401

from argparse import ArgumentParser
import datetime
import pandas as pd
import yaml
from settings import settings
import data_preprocessing.columns_filter as col_filter

STAGE = "filter_split_train_val"


def __parse_args():
    parser = ArgumentParser(STAGE)
    parser.add_argument('--input_file', required=True, help='Path to input data')
    parser.add_argument('--output_train_file', required=True, help='Path to train data')
    parser.add_argument('--output_val_file', required=True, help='Path to validation data')
    parser.add_argument('--params', required=True, help='Path to params')
    parser.add_argument('--params_section', required=True, help='Section with filter params')
    return parser.parse_args()


def __filter_columns(df_timeseries: pd.DataFrame, model_params: dict) -> pd.DataFrame:
    pol_id = model_params["pol_id"]
    columns_filters = model_params["columns_filters"]
    columns_selected = model_params["columns_selected"]
    target_column_name = col_filter.get_target_column(
        prediction_value_type=model_params['prediction_value_type'], pol_id=pol_id)

    if not columns_selected and columns_filters:
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
    else:
        df_filtered = df_timeseries[columns_selected]
    return df_filtered


def __split_train_val(df_timeseries: pd.DataFrame, split_params: {}) \
        -> (pd.DataFrame, pd.DataFrame):
    val_date_to = split_params["val_date_to"]
    forecast_period = split_params["forecast_period"]
    remove_days_at_start = split_params["remove_days_at_start"]
    val_date_to_d = datetime.datetime.strptime(val_date_to, "%Y-%m-%d").date()
    val_date_from = (val_date_to_d - datetime.timedelta(days=forecast_period - 1))\
        .strftime("%Y-%m-%d")
    train_date_to = (val_date_to_d - datetime.timedelta(days=forecast_period))\
        .strftime("%Y-%m-%d")
    date_first = df_timeseries.index.min()
    train_date_from = (date_first + datetime.timedelta(days=remove_days_at_start))\
        .strftime("%Y-%m-%d")
    df_train = df_timeseries.loc[train_date_from:train_date_to]
    df_val = df_timeseries.loc[val_date_from:val_date_to]
    return df_train, df_val


if __name__ == '__main__':
    print(f'Stage {STAGE} started')
    stage_args = __parse_args()
    with open(stage_args.params, 'r', encoding='UTF-8') as file_stream:
        params = yaml.safe_load(file_stream)
    split_params = params["split_periods"]
    model_params = params[stage_args.params_section]

    df_timeseries = pd.read_csv(stage_args.input_file, parse_dates=True,
                                index_col=settings.DATE_COLUMN_NAME)

    df_timeseries = __filter_columns(df_timeseries, model_params)
    df_train, df_val = __split_train_val(df_timeseries, split_params)

    df_train.to_csv(stage_args.output_train_file)
    df_val.to_csv(stage_args.output_val_file)

    print(f'Stage {STAGE} finished')
