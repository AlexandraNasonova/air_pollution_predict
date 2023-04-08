# pylint: disable=E0401, R0913, R0914
"""Module for features calculation using aggregation by
rolling windows and Exponential Moving Average"""

from copy import deepcopy
import warnings
import numpy as np
import pandas as pd
import datetime
import re
from progress.bar import Bar

# from IPython.core.display_functions import display
# from ipywidgets import IntProgress

warnings.filterwarnings('ignore')


def percentile(n_percent):
    """
    Calculate n - percentile of data"
    @param n_percent: Percentile percent for calculation
    @return: Percentile value
    """

    def percentile_(values):
        return np.percentile(values, n_percent)

    percentile_.__name__ = f'pctl{n_percent}'
    return percentile_


# add missing dates to GroupBy.Core object
def fill_missing_dates(df_source, date_col):
    """
    Add missing dates to GroupBy.Core object
    @param df_source: Source dataframe
    @param date_col: The name of the columns with dates
    @return: Processed dataframe with fixed dates
    """
    min_date, max_date = df_source[date_col].min(), df_source[date_col].max()
    groupby_day = df_source.groupby(pd.PeriodIndex(df_source[date_col], freq='D'))
    results = groupby_day.sum(min_count=1)

    idx = pd.period_range(min_date, max_date)
    results = results.reindex(idx, fill_value=np.nan)
    if date_col in results.columns.values:
        results.drop([date_col], axis=1, inplace=True)

    results.index.rename(date_col, inplace=True)

    return results


def calc_preag_fill(df_source, group_col, date_col, target_cols, preagg_method):
    """
    Calculate aggregation functions for the columns from group_col list, except the first one.
    @param df_source:  Source dataframe
    @param group_col: Names of the columns for grouping
    @param date_col: Name of the date column
    @param target_cols: The names of the target columns
    @param preagg_method: Function for preaggregation
    @return: Dataframe with pre-aggregation
    """
    # calc preaggregation
    data_preag = df_source.groupby(group_col).agg(
        preagg_method)[target_cols].reset_index()

    # fill missing dates
    data_preag_filled = data_preag.groupby(group_col[:-1]).apply(
        fill_missing_dates, date_col=date_col).drop(group_col[:-1],
                                                    axis=1).reset_index()

    # return DataFrame with calculated preaggregation and filled missing dates
    return data_preag_filled


def calc_rolling(data_preag_filled: pd.DataFrame, group_col: [], date_col: str,
                 method, method_param, rolling_window: int):
    """
    Calculates Aggregate functions for rolling windows
    @param data_preag_filled: Dataframe with aggregates
    @param group_col: Names of the columns for grouping
    @param date_col: Name of the date column
    @param method: Aggregate function to be calculated
    @param rolling_window: Rolling window
    @return: Dataframe with aggregates calculated on rolling window
    """
    # calc rolling stats
    if method_param is None:
        lf_df_filled = data_preag_filled.groupby(group_col[:-1]). \
            apply(lambda x: x.set_index(date_col).rolling(window=rolling_window, min_periods=1)
                  .agg(method)).drop(group_col[:-1], axis=1)
    else:
        lf_df_filled = data_preag_filled.groupby(group_col[:-1]). \
            apply(lambda x: x.set_index(date_col).rolling(window=rolling_window, min_periods=1)
                  .agg(method(method_param))).drop(group_col[:-1], axis=1)
    # return DataFrame with rolled columns from target_vars
    return lf_df_filled


def calc_ewm(data_preag_filled: pd.DataFrame, group_col: [], date_col: str, span: float):
    """
    Calculates Exponential Moving Average for the data frame
    @param data_preag_filled: Dataframe with aggregates
    @param group_col: Names of the columns for grouping
    @param date_col: Name of the date column
    @param span: Span value for Exponential Moving Average calculation
    @return: Dataframe with calculated Exponential Moving Average
    """
    lf_df_filled = data_preag_filled.groupby(group_col[:-1]). \
        apply(lambda x: x.set_index(date_col).ewm(span=span).mean()).drop(group_col[:-1], axis=1)

    # return DataFrame with rolled columns from target_vars
    return lf_df_filled


def shift(lf_df_filled: pd.DataFrame, group_col: [], date_col: str, lag: int):
    """
    Shifts the dataframe by lag days
    @param lf_df_filled:
    @param group_col: Names of the columns for grouping
    @param date_col: Name of the date column
    @param lag: Value of the lag to shift back
    @return: Shifted by lag days dataframe
    """
    # lf_df = lf_df_filled.groupby(
    #     level=group_col[:-1]).apply(lambda x: x.shift(lag)).reset_index()
    lf_df = lf_df_filled.apply(lambda x: x.shift(lag)).reset_index()
    # lf_df[date_col] = pd.to_datetime(lf_df[date_col].astype(str))
    lf_df[date_col] = pd.to_datetime(lf_df[date_col].astype(str)).map(datetime.datetime.date)

    # return DataFrame with following columns: filter_col, id_cols, date_col and shifted stats
    return lf_df


def __adjust_datetime_indices(data: pd.DataFrame, date_col: str) -> pd.DataFrame:
    data = data.sort_values(date_col)
    data_cl = deepcopy(data)
    data_cl.reset_index(inplace=True)
    data_cl[date_col] = data_cl[date_col].map(datetime.datetime.date)
    data_cl.set_index(date_col, inplace=True)
    data_cl.sort_values(date_col, inplace=True)
    return data_cl


def get_agg_function(method: str):
    if method in ('mean', 'median'):
        return method, None
    value_params = re.findall(r'\d+', method)
    if method.startswith("percentile"):
        return percentile, int(value_params[0])


def generate_lagged_features(
        data: pd.DataFrame,
        target_cols: list,
        id_cols: list,
        date_col: str,
        lags: list,
        windows: dict,
        preagg_methods: list,
        agg_methods: list,
        dynamic_filters: list,
        ewm_params: dict) -> pd.DataFrame:
    """
    data - dataframe with default index
    target_cols - column names for lags calculation
    id_cols - key columns to identify unique values
    date_col - column with datetime format values
    lags - lag values(days)
    windows - windows(days/weeks/months/etc.),
        calculation is performed within time range length of window
    preagg_methods - applied methods before rolling to make
        every value unique for given id_cols
    agg_methods - method of aggregation('mean', 'median', percentile, etc.)
    dynamic_filters - column names to use as filter
    ewm_params - span values(days) for each dynamic_filter
    """

    data_adj = __adjust_datetime_indices(data, date_col)
    data_gen = deepcopy(data_adj)

    total = len(dynamic_filters) * len(lags) * len(preagg_methods) \
            * (len(ewm_params) + len(windows) * len(agg_methods))
    # progress = IntProgress(min=0, max=total)
    # display(progress)

    preagg_methods_count = len(preagg_methods)
    filter_count = len(dynamic_filters)
    key_str = f'key{"|".join(id_cols)}_' if len(id_cols) > 1 else ''

    with Bar(f'Lags features for [{", ".join(target_cols)}] generation ...', max=total) as bar:
        for filter_col in dynamic_filters:
            group_col = [filter_col] + id_cols + [date_col]
            for lag in lags:
                for preagg in preagg_methods:
                    preagg_str = f'preag{preagg}_' if preagg_methods_count > 1 else ''
                    filter_col_str = f'filt{filter_col}' if filter_count > 1 else ''

                    data_preag_filled = calc_preag_fill(data_adj, group_col, date_col, target_cols, preagg)

                    # add ewm features
                    for alpha in ewm_params.get(filter_col, []):
                        ewm_filled = calc_ewm(data_preag_filled, group_col, date_col, alpha)
                        ewm = shift(ewm_filled, group_col, date_col, lag)
                        new_names = \
                            {x: f"{x}_lag{lag}d_ewm{alpha}_{key_str}{preagg_str}{filter_col_str}"
                             for x in target_cols}
                        data_gen = pd.merge(data_gen, ewm.rename(columns=new_names),
                                            how='left', on=group_col)
                        bar.next()
                        # progress.value += 1

                    # add rolling features
                    for window in windows.get(filter_col, []):
                        for method in agg_methods:
                            method_func, method_param = get_agg_function(method)
                            rolling_filled = calc_rolling(data_preag_filled, group_col,
                                                          date_col, method_func, method_param, window)

                            rolling = shift(rolling_filled, group_col, date_col, lag)
                            method_name = method.__name__ if type(method) != str else method
                            # method_name = method.__name__ if method.isinstance(str) else method

                            new_name = f"lag{lag}d_win{window}_{key_str}"
                            new_name += f"{preagg_str}ag{method_name}_{filter_col_str}"
                            new_names = {x: f"{x}_{new_name}" for x in target_cols}

                            data_gen = pd.merge(data_gen, rolling.rename(columns=new_names),
                                                how='left', on=group_col)
                            bar.next()
                            # progress.value += 1

    return data_gen
