import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from tqdm import tqdm
import statsmodels.api as sm
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from predict.stat_tests import StatTestWrapper


class ArimaHelper:

    __result_table: pd.DataFrame
    __ts: pd.DataFrame
    __col_name_pred: str
    __best_model = None

    def __init__(self, ts: pd.DataFrame, col_name_pred: str):
        self.__ts = deepcopy(ts)
        self.__col_name_pred = col_name_pred

    def optimize_sarima(self, parameters_list: list):
        """
            Return dataframe with parameters and corresponding AIC

            parameters_list - list with (p, q, P, Q) tuples
            d - integration order in ARIMA model
            D - seasonal integration order
            s - length of season
        """

        results = []
        best_aic = float("inf")

        for param in tqdm(parameters_list):
            # we need try-except because on some combinations model fails to converge
            try:
                model = sm.tsa.statespace.SARIMAX(
                    self.__ts[self.__col_name_pred],
                    order=(param[0], param[1], param[2]),
                    seasonal_order=(param[3], param[4], param[5], param[6]),
                ).fit(disp=-1)
            except:
                continue
            aic = model.aic
            if aic < best_aic:
                best_aic = aic
            results.append([param, model.aic])

        result_table = pd.DataFrame(results)
        result_table.columns = ["parameters", "aic"]
        # sorting in ascending order, the lower AIC is - the better
        result_table = result_table.sort_values(by="aic", ascending=True).reset_index(
            drop=True
        )

        self.__result_table = result_table

    def get_best_sarima_params(self) -> (int, int, int, int):
        return self.__result_table.parameters[0]

    def predict_best_sarima(self):
        p, d, q, P, D, Q, s = self.__result_table.parameters[0]
        self.__best_model = sm.tsa.statespace.SARIMAX(self.__ts['AQI'], order=(p, d, q), seasonal_order=(P, D, Q, s))\
            .fit(disp=-1)
        return self.__best_model.summary()

    def plot_sarima(self, n_steps: int):
        """
            Plots model vs predicted values

            series - dataset with timeseries
            model - fitted SARIMA model
            n_steps - number of steps to predict in the future

        """
        # adding model values
        data = deepcopy(self.__ts)
        data.columns = ["actual"]
        data["arima_model"] = self.__best_model.fittedvalues
        # making a shift on s+d steps, because these values were unobserved by the model
        # due to the differentiating
        params = self.get_best_sarima_params()
        d = params[1]
        s = params[6]
        data["arima_model"][: s + d] = np.NaN

        # forecasting on n_steps forward
        forecast = self.__best_model.predict(start=data.shape[0], end=data.shape[0] + n_steps)
        forecast = data.arima_model.append(forecast)
        # calculate error, again having shifted on s+d steps from the beginning
        error = mean_absolute_percentage_error(
            data["actual"][s + d:], data["arima_model"][s + d:]
        )

        plt.figure(figsize=(15, 7))
        plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
        plt.plot(data.actual, label="actual")
        plt.plot(forecast, color="r", label="model")
        plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color="lightgrey")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_residuals_diagnostics(self):
        '''
        fig = plt.figure(figsize=(21, 6))
        ax = fig.add_subplot(111)
        self.__best_model.resid.plot(ax=ax)
        plt.grid(axis='x', color='green', linestyle='--', linewidth=0.5)
        plt.show()
        '''

        self.__best_model['STA-NL00701'].get_best_model().plot_diagnostics(figsize=(21, 7))
        plt.show()

    def get_best_model(self):
        return self.__best_model

    def get_residuals_analysis(self):
        self.plot_residuals_diagnostics()
        StatTestWrapper.check_stationarity_kpss(self.__best_model.resid)
        StatTestWrapper.check_resid_ljung_box(self.__best_model.resid)
        StatTestWrapper.check_resid_durbin_watson(self.__best_model.resid)
        StatTestWrapper.check_resid_wilcoxon(self.__best_model.resid)
        # StatTestWrapper.check_resid_normality(self.__best_model.resid)


