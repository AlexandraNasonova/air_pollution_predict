# pylint: disable=E0401, R0913, R0914, W0703
"""Timeseries splitter for tran/test X/y sub-dataframes"""

import datetime
import pandas as pd


def extract_labels(df_timeseries: pd.DataFrame, target_column: str):
    """
    Extract labels (target column values) from dataframe
    @param df_timeseries: Source dataframe
    @param target_column: The name of the target columns
    @return:
    """
    x_ts = df_timeseries.drop([target_column], axis=1)
    y_ts = df_timeseries[target_column]
    return x_ts, y_ts


def split_x_y(df_timeseries: pd.DataFrame, index_cols: [], y_value_col: str) \
        -> (pd.DataFrame, pd.DataFrame):
    """
    Splits the timeseries to X (data for training) and y (prediction for training) dataframes
    @param df_timeseries: The timeseries
    @param index_cols: The list index columns
    @param y_value_col: The name of the value column (contains predictions)
    @return: X, y dataframes
    """
    df_copied = df_timeseries.copy(deep=True)
    df_copied.reset_index(inplace=True)
    df_copied.set_index(index_cols, inplace=True)
    y_df = df_copied.loc[:, y_value_col].copy(deep=True)
    df_copied.drop(columns=[y_value_col], inplace=True)
    return df_copied, y_df


def split_x_y_for_period(df_timeseries: pd.DataFrame, index_cols: [], y_value_col: str,
                         dt_start: datetime.date, dt_end: datetime.date) \
        -> (pd.DataFrame, pd.DataFrame):
    """
    Extracts the required period from the timeseries and
    then slits the result to X (data for training) and y (prediction for training) dataframes
    @param df_timeseries: The timeseries
    @param index_cols: The list index columns
    @param y_value_col: The name of the value column (contains predictions)
    @param dt_end: The first date of the required period
    @param dt_start: The last date of the required period
    @return: X, y dataframes
    """
    df_copied = df_timeseries.copy(deep=True)
    df_copied.reset_index(inplace=True)
    df_copied.set_index(index_cols, inplace=True)
    return split_x_y(df_timeseries.loc[dt_start:dt_end, ], index_cols, y_value_col)


def split_ts_time_part(df_timeseries: pd.DataFrame, index_cols: [], y_value_col: str,
                       train_start_dt: datetime.date, train_end_dt: datetime.date,
                       test_start_dt: datetime.date, test_end_dt: datetime.date) \
        -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Splits the dataframe to train and test and X and y
    @param df_timeseries: The timeseries
    @param index_cols: The list index columns
    @param y_value_col: The name of the value column (contains predictions)
    @param train_start_dt: The first date of the required period for the train dataframe
    @param train_end_dt: The last date of the required period for the train dataframe
    @param test_start_dt: The first date of the required period for the test dataframe
    @param test_end_dt: The last date of the required period for the test dataframe
    @return: X_train, y_train, X_test, y_test
    """
    x_train, y_train = split_x_y_for_period(df_timeseries, index_cols,
                                            y_value_col, train_start_dt, train_end_dt)
    x_test, y_test = split_x_y_for_period(df_timeseries, index_cols,
                                          y_value_col, test_start_dt, test_end_dt)
    return x_train, y_train, x_test, y_test
