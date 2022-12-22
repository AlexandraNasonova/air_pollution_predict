from copy import deepcopy
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from scipy.special import boxcox1p
from predict.plot_helper import PlotHelper

pt = PowerTransformer(method='yeo-johnson')


class TransformHelper:

    @staticmethod
    def apply_best_box_cox(ts: pd.DataFrame, plots_needed=False):
        ts_boxcox = deepcopy(ts)
        pt = PowerTransformer(method='yeo-johnson')
        pt.fit(ts_boxcox['Concentration'].array.reshape(-1, 1))
        print(f'Best lambda: {pt.lambdas_}')
        ts_boxcox['Concentration'] = boxcox1p(ts_boxcox['Concentration'], pt.lambdas_[0])
        if plots_needed:
            PlotHelper.plot_ts(ts_boxcox)
        return ts_boxcox

    @staticmethod
    def apply_differencing(ts: pd.DataFrame, shift_type: str, shift_value: int, source_plots_needed=False,
                           dif_plot_needed=False) -> pd.DataFrame:
        offset, p = TransformHelper.__get_offset_by_period(shift_type, shift_value)
        ts_dif = (ts - ts.shift(fill_value=0, freq=offset))[p:]
        if source_plots_needed:
            PlotHelper.plot_ts(ts, title=f'SOURCE TS')
        if dif_plot_needed:
            PlotHelper.plot_ts(ts_dif, title=f'TS DIF BY {shift_value} {shift_type}')
        ts_dif = ts_dif.dropna()
        return ts_dif

    @staticmethod
    def __get_offset_by_period(shift_type: str, shift_value: int) -> (pd.DateOffset, int):
        # ts_interval = 24
        ts_interval = 1
        if shift_type == 'Y':
            offset = pd.DateOffset(years=shift_value)
            p = 365 * ts_interval
        elif shift_type == 'M':
            offset = pd.DateOffset(months=shift_value)
            p = 30 * ts_interval
        elif shift_type == 'W':
            offset = pd.DateOffset(weeks=shift_value)
            p = 7 * ts_interval
        elif shift_type == 'D':
            offset = pd.DateOffset(days=shift_value)
            p = 1 * ts_interval
        else:
            offset = pd.DateOffset(hours=shift_value)
            p = 1
        return offset, p
