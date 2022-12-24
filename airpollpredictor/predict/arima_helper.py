import numpy as np
from tqdm import tqdm
import statsmodels.api as sm
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


class ArimaHelper:

    __result_table: pd.DataFrame
    __ts: pd.DataFrame
    __col_name_pred: str
    __best_model = None
    __cross_val_predictions: pd.DataFrame
    __rmse_mean: float
    __exog_data: pd.DataFrame

    def __init__(self, ts: pd.DataFrame, col_name_pred: str, exog_data: pd.DataFrame = None):
        self.__ts = deepcopy(ts)
        self.__col_name_pred = col_name_pred
        self.__exog_data = exog_data

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
                    exog=self.__exog_data
                ).fit(disp=-1)
            except:
                continue
            aic = model.aic
            if aic < best_aic:
                best_aic = aic
            results.append([param, model.aic])

        result_table = pd.DataFrame(results)
        result_table.columns = ["parameters", "aic"]
        result_table = result_table.sort_values(by="aic", ascending=True).reset_index(
            drop=True
        )
        self.__result_table = result_table

    def cross_validate_best_arima(self, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        rmse = []
        all_predictions = None
        for train_index, test_index in tscv.split(self.__ts):
            cv_train, cv_test = self.__ts.iloc[train_index], self.__ts.iloc[test_index]
            exog_train, exog_test = self.__exog_data.iloc[train_index], self.__exog_data.iloc[test_index]
            p, d, q, P, D, Q, s = self.get_best_sarima_params()
            model = sm.tsa.statespace.SARIMAX(cv_train, order=(p, d, q),
                                              seasonal_order=(P, D, Q, s), exog=exog_train).fit()
            predictions = model.predict(start=cv_train.shape[0], end=cv_train.shape[0] + cv_test.shape[0] - 1,
                                        exog=exog_test)
            if all_predictions is None:
                all_predictions = predictions
            else:
                all_predictions = pd.concat([all_predictions, predictions], axis=0)
            true_values = cv_test.values
            rmse.append(np.sqrt(mean_squared_error(true_values, predictions)))
        self.__rmse_mean = np.mean(rmse)
        self.__cross_val_predictions = all_predictions

    def plot_cross_val_predictions(self):
        plt.plot(self.__ts, label='Actual data')
        plt.plot(self.__cross_val_predictions, label='Forecast')
        plt.title("RMSE: {0:.2f}%".format(self.__rmse_mean))
        plt.legend()

    def get_best_sarima_params(self) -> (int, int, int, int):
        return self.__result_table.parameters[0]

    def predict_best_sarima(self):
        p, d, q, P, D, Q, s = self.__result_table.parameters[0]
        self.__best_model = sm.tsa.statespace.SARIMAX(self.__ts[self.__col_name_pred], order=(p, d, q),
                                                      seasonal_order=(P, D, Q, s), exog=self.__exog_data).fit()
        return self.__best_model.summary()

    def get_best_model(self):
        return self.__best_model


