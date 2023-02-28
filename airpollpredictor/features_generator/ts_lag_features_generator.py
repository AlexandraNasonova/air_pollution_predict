import numpy as np
import pandas as pd
from copy import deepcopy
from IPython.core.display_functions import display
from ipywidgets import IntProgress
import warnings

warnings.filterwarnings('ignore')


def percentile(n):
    """Calculate n - percentile of data"""

    def percentile_(x):
        return np.percentile(x, n)

    percentile_.__name__ = 'pctl%s' % n
    return percentile_


# add missing dates to GroupBy.Core object
def fill_missing_dates(x, date_col):
    min_date, max_date = x[date_col].min(), x[date_col].max()
    groupby_day = x.groupby(pd.PeriodIndex(x[date_col], freq='D'))
    results = groupby_day.sum(min_count=1)

    idx = pd.period_range(min_date, max_date)
    results = results.reindex(idx, fill_value=np.nan)

    results.index.rename(date_col, inplace=True)

    return results


def calc_preag_fill(data, group_col, date_col, target_cols, preagg_method):
    # calc preaggregation
    data_preag = data.groupby(group_col).agg(
        preagg_method)[target_cols].reset_index()

    # fill missing dates
    data_preag_filled = data_preag.groupby(group_col[:-1]).apply(
        fill_missing_dates, date_col=date_col).drop(group_col[:-1],
                                                    axis=1).reset_index()

    # return DataFrame with calculated preaggregation and filled missing dates
    return data_preag_filled


def calc_rolling(data_preag_filled, group_col, date_col, method, w):
    # calc rolling stats
    lf_df_filled = data_preag_filled.groupby(group_col[:-1]). \
        apply(lambda x: x.set_index(date_col).rolling(window=w, min_periods=1).agg(method)).drop(group_col[:-1], axis=1)

    # return DataFrame with rolled columns from target_vars
    return lf_df_filled


def calc_ewm(data_preag_filled, group_col, date_col, span):
    # calc ewm stats
    lf_df_filled = data_preag_filled.groupby(group_col[:-1]). \
        apply(lambda x: x.set_index(date_col).ewm(span=span).mean()).drop(group_col[:-1], axis=1)

    # return DataFrame with rolled columns from target_vars
    return lf_df_filled


def shift(lf_df_filled, group_col, date_col, lag):
    lf_df = lf_df_filled.groupby(
        level=group_col[:-1]).apply(lambda x: x.shift(lag)).reset_index()
    lf_df[date_col] = pd.to_datetime(lf_df[date_col].astype(str))

    # return DataFrame with following columns: filter_col, id_cols, date_col and shifted stats
    return lf_df


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

    data = data.sort_values(date_col)
    out_df = deepcopy(data)
    # dates = [min(data[date_col]), max(data[date_col])]
    total = len(dynamic_filters) * len(lags) * len(preagg_methods) * (len(ewm_params) + len(windows) * len(agg_methods))
    progress = IntProgress(min=0, max=total)
    display(progress)

    preagg_methods_count = len(preagg_methods)
    filter_count = len(dynamic_filters)
    key_str = f'key{"|".join(id_cols)}_' if len(id_cols) > 1 else ''
    for filter_col in dynamic_filters:
        group_col = [filter_col] + id_cols + [date_col]
        for lag in lags:
            for preagg in preagg_methods:
                preagg_str = f'preag{preagg}_' if preagg_methods_count > 1 else ''
                filter_col_str = f'filt{filter_col}' if filter_count > 1 else ''

                data_preag_filled = calc_preag_fill(data, group_col, date_col, target_cols, preagg)

                # add ewm features
                for alpha in ewm_params.get(filter_col, []):
                    ewm_filled = calc_ewm(data_preag_filled, group_col, date_col, alpha)
                    ewm = shift(ewm_filled, group_col, date_col, lag)
                    new_names = {x: "{0}_lag{1}d_ewm{2}_{3}{4}{5}".format(x, lag, alpha, key_str,
                                                                          preagg_str, filter_col_str)
                                 for x in target_cols}
                    out_df = pd.merge(out_df, ewm.rename(columns=new_names), how='left', on=group_col)
                    progress.value += 1

                # add rolling features
                for w in windows.get(filter_col, []):
                    for method in agg_methods:
                        rolling_filled = calc_rolling(data_preag_filled, group_col, date_col, method, w)

                        rolling = shift(rolling_filled, group_col, date_col, lag)
                        method_name = method.__name__ if type(method) != str else method

                        new_names = {x: "{0}_lag{1}d_win{2}_{3}{4}ag{5}_{6}".format(x, lag, w, key_str, preagg_str,
                                                                                    method_name, filter_col_str)
                                     for x in target_cols}

                        out_df = pd.merge(out_df, rolling.rename(columns=new_names), how='left', on=group_col)
                        progress.value += 1

    return out_df
