import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
import statsmodels.api as sm
from ml_models_search.arima.transformations import TransformHelper
from copy import deepcopy


class StatTestWrapper:

    @staticmethod
    def get_residuals_analysis(model):
        model.plot_diagnostics(figsize=(21, 7))
        plt.show()
        StatTestWrapper.check_stationarity_kpss(model.resid)
        StatTestWrapper.check_resid_ljung_box(model.resid)
        StatTestWrapper.check_resid_durbin_watson(model.resid)
        StatTestWrapper.check_resid_wilcoxon(model.resid)

    @staticmethod
    def pre_analise(ts: pd.DataFrame, column_name: str, diffs_needed: [] = None, is_box_cox_needed=False,
                    source_plots_needed=False, dif_plot_needed=False, box_cox_plot_needed=False, lags=None):
        try:
            ts_n = deepcopy(ts)
            if is_box_cox_needed:
                ts_n = TransformHelper.apply_best_box_cox(ts_n, column_name, plots_needed=box_cox_plot_needed)
            source_plots_shown = False
            if diffs_needed is not None and len(diffs_needed) > 0:
                for dif in diffs_needed:
                    ts_n = TransformHelper.apply_differencing(ts=ts_n, shift_type=dif[0], shift_value=dif[1],
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
    def plot_acf_pacf(ts: pd.DataFrame, lags=200):
        fig = plt.figure(figsize=(21, 7))
        ax1 = fig.add_subplot(211)
        sm.graphics.tsa.plot_acf(ts, auto_ylims=True, lags=lags, ax=ax1)
        ax2 = fig.add_subplot(212)
        sm.graphics.tsa.plot_pacf(ts, auto_ylims=True, lags=lags, ax=ax2, method='ols')
        plt.show()

    @staticmethod
    def check_stationarity_dick_fuller(ts: pd.DataFrame):
        result = adfuller(ts.values)
        StatTestWrapper.__print_test_results("ADF", result)

    @staticmethod
    def check_stationarity_kpss(ts: pd.DataFrame):
        result = kpss(ts.values)
        StatTestWrapper.__print_test_results("KPSS", result)

    @staticmethod
    def check_resid_kpss(resid):
        result = kpss(resid)
        StatTestWrapper.__print_test_results("KPSS", result)

    @staticmethod
    def check_resid_ljung_box(resid, lags_max=30):
        #fig = plt.figure(figsize=(17, 5))
        #ax = fig.add_subplot(111)
        # H0: lags are not correlated
        pd.DataFrame({'lags': range(lags_max),
                      'pvalue': sm.stats.diagnostic.acorr_ljungbox(resid, lags=lags_max).iloc[:, 1],
                      'critical': np.array([0.05] * lags_max)}) \
            .set_index('lags').plot(title="Ljung-Box: residuals autocorrelation test")
        plt.show()

    @staticmethod
    def check_resid_durbin_watson(resid):
        # if there is no correlation in a sample then Stat ~= 2
        result = sm.stats.durbin_watson(resid)
        print('The Durbin-Watson residuals statistics %f' % result)
        if (result >= 1.8) & (result <= 2.2):
            print("\u001b[32mDurbin-Watson: residuals are not correlated\u001b[0m")
        else:
            print("\x1b[31mDurbin-Watson: residuals are correlated\x1b[0m")

    @staticmethod
    def check_resid_wilcoxon(resid):
        # H0: obervations are not biased
        stat, p = stats.wilcoxon(resid)
        print('The Wilcoxon residuals statistic %f, pvalue: %.2f' % (stat, p))
        if p > 0.5:
            print("\u001b[32mWilcoxon: residuals are not biased\u001b[0m")
        else:
            print("\x1b[31mWilcoxon: residuals are biased\x1b[0m")

    @staticmethod
    def check_resid_normality(resid):
        # H0: obervations are not biased
        result = stats.normaltest(resid)
        print(f'The residuals normality statistic: {result}')
        fig = plt.figure(figsize=(17, 5))
        stats.probplot(resid, dist="norm", plot=plt)
        plt.show()

    @staticmethod
    def __print_test_results(test_name: str, result: []):
        tab_vals = ''
        tab_vals_ind = 0
        if test_name == 'KPSS':
            tab_vals = 'table distribution'
            tab_vals_ind = 3
        elif test_name == 'ADF':
            tab_vals = 'critical values'
            tab_vals_ind = 4
        print('The %s test statistic %f    pvalue: %.2f   maximal Lag: %i   %s: %s' %
              (test_name, result[0], result[1], result[2], tab_vals, result[tab_vals_ind]))
        if (((test_name == 'ADF') & (result[1] <= 0.05) & (result[tab_vals_ind]['5%'] > result[0]))
            | ((test_name == 'KPSS') & (result[1] > 0.05)
               #& (result[tab_vals_ind]['5%'] <= result[0])
                )):
            print("\u001b[32m%s: stationary\u001b[0m" % test_name)
        else:
            print("\x1b[31m%s: non-stationary\x1b[0m" % test_name)
