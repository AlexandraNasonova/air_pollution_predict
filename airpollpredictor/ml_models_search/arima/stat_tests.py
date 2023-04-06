# pylint: disable=E0401, R0913, R0914, W0703

"""
Wrapper for different statistical tests for ARIMA models
"""
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
import statsmodels.api as sm
from ml_models_search.arima.transformations import TransformHelper


class StatTestWrapper:
    """
    Wrapper for different statistical tests for ARIMA models
    """

    @staticmethod
    def get_residuals_analysis(model) -> None:
        """
        Runs all tests required for the residuals analysis
        @param model: Fitted SARIMAX model
        """
        model.plot_diagnostics(figsize=(21, 7))
        plt.show()
        StatTestWrapper.check_stationarity_kpss(model.resid)
        StatTestWrapper.check_resid_ljung_box(model.resid)
        StatTestWrapper.check_resid_durbin_watson(model.resid)
        StatTestWrapper.check_resid_wilcoxon(model.resid)

    @staticmethod
    def pre_analise(df_timeseries: pd.DataFrame,
                    column_name: str,
                    diffs_needed: [] = None,
                    is_box_cox_needed=False,
                    source_plots_needed=False,
                    dif_plot_needed=False,
                    box_cox_plot_needed=False,
                    lags=None):
        """

        @param df_timeseries: The timeseries dataframe
        @param column_name: Values column name
        @param diffs_needed: The list of lags for differentiations.
        Empty or None if not needed
        @param is_box_cox_needed: Flag if Box-Cox transformation is needed
        @param source_plots_needed: Flag if plotting of the timeseries is needed
        @param dif_plot_needed: Flag if plotting of the timeseries after
        differentiation is needed
        @param box_cox_plot_needed: Flag if plotting of the timeseries after
        Box-Cox transformation is needed
        @param lags:
        @return: Transformed and differentiated timeseries
        """
        try:
            ts_n = deepcopy(df_timeseries)
            if is_box_cox_needed:
                ts_n = TransformHelper \
                    .apply_best_box_cox(ts_n, column_name, plots_needed=box_cox_plot_needed)
            source_plots_shown = False
            if diffs_needed is not None and len(diffs_needed) > 0:
                for dif in diffs_needed:
                    ts_n = TransformHelper \
                        .apply_differencing(
                        df_timeseries=ts_n, shift_type=dif[0], shift_value=dif[1],
                        source_plots_needed=(
                                source_plots_needed and not source_plots_shown),
                        dif_plot_needed=dif_plot_needed)
                    source_plots_shown = True
            StatTestWrapper.check_stationarity_dick_fuller(ts_n)
            StatTestWrapper.check_stationarity_kpss(ts_n)
            if lags is None:
                StatTestWrapper.plot_acf_pacf(ts_n)
            else:
                StatTestWrapper.plot_acf_pacf(ts_n, lags)
            return ts_n
        except Exception as ex:
            print(f'\x1b[31mERROR: {ex}\x1b[0m')
        return None

    @staticmethod
    def plot_acf_pacf(df_timeseries: pd.DataFrame, lags=200):
        """
        Plot ACF and PACF for timeseries
        @param lags: The number of lags for ACF/PACF calculations
        @param df_timeseries: The timeseries
        """
        fig = plt.figure(figsize=(21, 7))
        ax1 = fig.add_subplot(211)
        sm.graphics.tsa.plot_acf(df_timeseries, auto_ylims=True, lags=lags, ax=ax1)
        ax2 = fig.add_subplot(212)
        sm.graphics.tsa.plot_pacf(df_timeseries, auto_ylims=True, lags=lags, ax=ax2, method='ols')
        plt.show()

    @staticmethod
    def check_stationarity_dick_fuller(df_timeseries: pd.DataFrame):
        """
        Runs Dick-Fuller test for timeseries
        @param df_timeseries: The timeseries
        """
        result = adfuller(df_timeseries.values)
        StatTestWrapper.__print_test_results("ADF", result)

    @staticmethod
    def check_stationarity_kpss(df_timeseries: pd.DataFrame) -> None:
        """
        Runs KPSS test for timeseries
        @param df_timeseries: The timeseries
        """
        result = kpss(df_timeseries.values)
        StatTestWrapper.__print_test_results("KPSS", result)

    @staticmethod
    def check_resid_kpss(resid) -> None:
        """
        Runs KPSS test for residuals
        @param resid: The residuals
        """
        result = kpss(resid)
        StatTestWrapper.__print_test_results("KPSS", result)

    @staticmethod
    def check_resid_ljung_box(resid, lags_max=30) -> None:
        """
        Runs Ljung-Box test for residuals
        @param lags_max: Maximum lags numbers
        @param resid: The residuals
        """
        # H0: lags are not correlated
        pd.DataFrame({'lags': range(lags_max),
                      'value': sm.stats.diagnostic.acorr_ljungbox(resid, lags=lags_max).iloc[:, 1],
                      'critical': np.array([0.05] * lags_max)}) \
            .set_index('lags').plot(title="Ljung-Box: residuals autocorrection test")
        plt.show()

    @staticmethod
    def check_resid_durbin_watson(resid) -> None:
        """
        Runs Durbin-Watson test for residuals
        @param resid: The residuals
        """
        # if there is no correlation in a sample then Stat ~= 2
        result = sm.stats.durbin_watson(resid)
        print(f'The Durbin-Watson residuals statistics {result:f}')
        if (result >= 1.8) & (result <= 2.2):
            print("\u001b[32mDurbin-Watson: residuals are not correlated\u001b[0m")
        else:
            print("\x1b[31mDurbin-Watson: residuals are correlated\x1b[0m")

    @staticmethod
    def check_resid_wilcoxon(resid) -> None:
        """
        Runs Wilcoxon test for residuals
        @param resid: The residuals
        """
        # H0: observations are not biased
        stat, pvalue, _ = stats.wilcoxon(resid)
        print(f'The Wilcoxon residuals statistic {stat:f}, pvalue: {pvalue:.2f}')
        if pvalue > 0.5:
            print("\u001b[32mWilcoxon: residuals are not biased\u001b[0m")
        else:
            print("\x1b[31mWilcoxon: residuals are biased\x1b[0m")

    @staticmethod
    def check_resid_normality(resid) -> None:
        """
        Runs normality-test for residuals
        @param resid: The residuals
        """
        # H0: observations are not biased
        result = stats.normaltest(resid)
        print(f'The residuals normality statistic: {result}')
        plt.figure(figsize=(17, 5))
        stats.probplot(resid, dist="norm", plot=plt)
        plt.show()

    @staticmethod
    def __print_test_results(test_name: str, result: []) -> None:
        tab_vals = ''
        tab_vals_ind = 0
        if test_name == 'KPSS':
            tab_vals = 'table distribution'
            tab_vals_ind = 3
        elif test_name == 'ADF':
            tab_vals = 'critical values'
            tab_vals_ind = 4
        print(
            f'The {test_name} test statistic {result[0]:f}    '
            f'pvalue: {result[1]:.2f}   maximal Lag: {result[2]:d}   '
            f'{tab_vals}: {result[tab_vals_ind]}')
        if (((test_name == 'ADF') & (result[1] <= 0.05) & (result[tab_vals_ind]['5%'] > result[0]))
                | ((test_name == 'KPSS') & (result[1] > 0.05)
                )):
            print(f"\u001b[32m{test_name}: stationary\u001b[0m")
        else:
            print(f"\x1b[31m{test_name}: non-stationary\x1b[0m")
