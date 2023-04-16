# pylint: disable=E0401, R0913, R0914, W0703

"""
Timeseries transformations helper: Lag-differentiation, Fourier and Box-Cox transformations
"""
from copy import deepcopy
import pandas as pd
from pmdarima.preprocessing import FourierFeaturizer
from sklearn.preprocessing import PowerTransformer
from scipy.special import boxcox1p
import ml_models_search.ts_plotter as ts_plt

pt = PowerTransformer(method='yeo-johnson')


class TransformHelper:
    """
    Timeseries transformations helper: Fourier, Box-Cox
    """

    @staticmethod
    def apply_fourier(df_timeseries: pd.DataFrame, m_season_period=365, k_sins=1) -> pd.DataFrame:
        """
        Applies Fourier transformation to the time series
        @param df_timeseries: The timeseries
        @param m_season_period: The seasonal periodicity of the endogenous vector, y.
        @param k_sins: The number of sine and cosine terms (each) to include
        @return: Dataframe after Fourier-transformation without date column
        """
        four_terms = FourierFeaturizer(m=m_season_period, k=k_sins)
        y_prime, exog = four_terms.fit_transform(df_timeseries)
        exog['date'] = y_prime.index
        exog = exog.set_index(exog['date'])
        exog.index.freq = 'D'
        exog = exog.drop(columns=['date'])
        return exog

    @staticmethod
    def apply_best_box_cox(df_timeseries: pd.DataFrame,
                           column_name: str, plots_needed=False) -> pd.DataFrame:
        """
        Applies Box-Cox transformation to the time series
        @param df_timeseries: The timeseries
        @param column_name: Value column name
        @param plots_needed: Flag if plotting of the timeseries after
        Box-Cox transformation is needed
        @return: Dataframe after dox-Cox transformation
        """
        ts_boxcox = deepcopy(df_timeseries)
        power_transformer = PowerTransformer(method='yeo-johnson')
        power_transformer.fit(ts_boxcox[column_name].array.reshape(-1, 1))
        print(f'Best lambda: {power_transformer.lambdas_}')
        ts_boxcox[column_name] = boxcox1p(ts_boxcox[column_name], power_transformer.lambdas_[0])
        if plots_needed:
            ts_plt.plot_ts(ts_boxcox)
        return ts_boxcox

    @staticmethod
    def apply_differencing(df_timeseries: pd.DataFrame,
                           shift_type: str,
                           shift_value: int,
                           source_plots_needed=False,
                           dif_plot_needed=False) -> pd.DataFrame:
        """
        Applies lag-differentiation to the time series
        @param df_timeseries: The timeseries
        @param shift_type: The measure of the shift interval:
        Y - year, M - month, W - week, D - day
        @param shift_value: The value of the shift interval
        @param source_plots_needed: Flag if plotting of the timeseries is needed
        @param dif_plot_needed: Flag if plotting of the timeseries after
        differentiation is needed
        @return: Dataframe after lag-differentiation
        """
        offset, p_shift = TransformHelper.__get_offset_by_period(shift_type, shift_value)
        ts_dif = (df_timeseries - df_timeseries.shift(fill_value=0, freq=offset))[p_shift:]
        if source_plots_needed:
            ts_plt.plot_ts(df_timeseries, title='SOURCE TS')
        if dif_plot_needed:
            ts_plt.plot_ts(ts_dif, title=f'TS DIF BY {shift_value} {shift_type}')
        ts_dif = ts_dif.dropna()
        return ts_dif

    @staticmethod
    def __get_offset_by_period(shift_type: str, shift_value: int) -> (pd.DateOffset, int):
        # ts_interval = 24
        ts_interval = 1
        if shift_type == 'Y':
            offset = pd.DateOffset(years=shift_value)
            p_shift = 365 * ts_interval
        elif shift_type == 'M':
            offset = pd.DateOffset(months=shift_value)
            p_shift = 30 * ts_interval
        elif shift_type == 'W':
            offset = pd.DateOffset(weeks=shift_value)
            p_shift = 7 * ts_interval
        elif shift_type == 'D':
            offset = pd.DateOffset(days=shift_value)
            p_shift = 1 * ts_interval
        else:
            offset = pd.DateOffset(hours=shift_value)
            p_shift = 1
        return offset, p_shift
