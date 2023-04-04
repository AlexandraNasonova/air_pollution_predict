import pandas as pd


def split_x_y(df: pd.DataFrame, index_cols, y_value_col):
    x = df.copy(deep=True)
    x.reset_index(inplace=True)
    x.set_index(index_cols, inplace=True)
    y = x.loc[:,y_value_col].copy(deep=True)
    x.drop(columns=[y_value_col], inplace=True)
    return x, y


def split_x_y_for_period(df: pd.DataFrame, index_cols, y_value_col, dt_start, dt_end):
    x = df.copy(deep=True)
    x.reset_index(inplace=True)
    x.set_index(index_cols, inplace=True)
    return split_x_y(df.loc[dt_start:dt_end,], index_cols, y_value_col)


def split_ts_time_part(df: pd.DataFrame, index_cols, y_value_col, train_start_dt, train_end_dt, test_start_dt, test_end_dt):
    x_train, y_train = split_x_y_for_period(df, index_cols, y_value_col, train_start_dt, train_end_dt)
    x_test, y_test = split_x_y_for_period(df, index_cols, y_value_col, test_start_dt, test_end_dt)
    return x_train, y_train, x_test, y_test

