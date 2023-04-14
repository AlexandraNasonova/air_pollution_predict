# pylint: disable=E0401

from argparse import ArgumentParser
import pandas as pd
import yaml
from settings import settings

STAGE = "split_train_val"


def __parse_args():
    parser = ArgumentParser(STAGE)
    parser.add_argument('--input_file', required=True, help='Path to input data')
    parser.add_argument('--output_train_file', required=True, help='Path to train data')
    parser.add_argument('--output_val_file', required=True, help='Path to validation data')
    parser.add_argument('--params', required=True, help='Path to params')
    return parser.parse_args()


if __name__ == '__main__':
    stage_args = __parse_args()
    with open(stage_args.params, 'r', encoding='UTF-8') as file_stream:
        split_periods = yaml.safe_load(file_stream)["split_periods"]

    train_date_from = split_periods["train_date_from"]
    train_date_to = split_periods["train_date_to"]
    val_date_from = split_periods["val_date_from"]
    val_date_to = split_periods["val_date_to"]

    df_timeseries = pd.read_csv(stage_args.input_file, parse_dates=True,
                                index_col=settings.DATE_COLUMN_NAME)

    df_train = df_timeseries.loc[train_date_from:train_date_to]
    df_val = df_timeseries.loc[val_date_from:val_date_to]
    df_train.to_csv(stage_args.output_train_file)
    df_val.to_csv(stage_args.output_val_file)
