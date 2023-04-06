# pylint: disable=E0401, R0913, R0914, W0703

"""
Contains methods for cross-validate and fit/predict SARIMAX models on dataframes
time series with time series
"""
from typing import Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import numpy as np
import pandas as pd


def cross_val_sarimax_calc_rmse(ts_train: pd.DataFrame,
                                order: tuple[int, int, int],
                                seasonal_order: Optional[tuple[int, int, int, int]],
                                exog_train: Optional[pd.DataFrame],
                                n_splits=16, ) -> (float, []):
    """
    Fit SARIMAX model by cross-validation on time series and calculate RMSE mean metric
    @param ts_train: Dataframe for training
    @param order: (p, d, q) ARIMA params
    @param seasonal_order: (P, D, Q, S) ARIMA params
    @param exog_train: Dataframe with exogenous factors for training
    @param n_splits: Number of splits for cross-validation
    @return: Mean RMSE value and the list of all predictions from cross-validation
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse = []
    all_predictions = None
    for train_index, test_index in tscv.split(ts_train):
        cv_train, cv_test = ts_train.iloc[train_index], ts_train.iloc[test_index]
        cv_exog_train, cv_exog_test = None, None
        if exog_train is not None:
            cv_exog_train, cv_exog_test = exog_train.iloc[train_index], exog_train.iloc[test_index]
        cv_model = sm.tsa.statespace.SARIMAX(cv_train, order=order,
                                             seasonal_order=seasonal_order,
                                             exog=cv_exog_train).fit(disp=0)
        predictions = cv_model.predict(start=cv_train.shape[0],
                                       end=cv_train.shape[0] + cv_test.shape[0] - 1,
                                       exog=cv_exog_test)
        if all_predictions is None:
            all_predictions = predictions
        else:
            all_predictions = pd.concat([all_predictions, predictions], axis=0)
        rmse.append(np.sqrt(mean_squared_error(cv_test.values, predictions)))
    rmse_mean = np.mean(rmse)
    return rmse_mean, all_predictions


def predict_sarimax_calc_rmse(ts_train: pd.DataFrame,
                              ts_val: pd.DataFrame,
                              order: tuple[int, int, int],
                              seasonal_order: Optional[tuple[int, int, int, int]],
                              exog_train: Optional[pd.DataFrame],
                              exog_val: Optional[pd.DataFrame]) -> (float, []):
    """
    Fit SARIMAX model, predict on validation dataframe and calculate RMSE mean metric
    @param ts_train: Dataframe for training
    @param ts_val: Dataframe for validation
    @param order: (p, d, q) ARIMA params
    @param seasonal_order: (P, D, Q, S) ARIMA params
    @param exog_train: Dataframe with exogenous factors for training
    @param exog_val: Dataframe with exogenous factors for validation
    @return: Mean RMSE value and the list of predictions from validation
    """
    model = sm.tsa.statespace.SARIMAX(ts_train, order=order,
                                      seasonal_order=seasonal_order,
                                      exog=exog_train).fit(disp=0)
    predictions = model.predict(start=ts_train.shape[0],
                                end=ts_train.shape[0] + ts_val.shape[0] - 1,
                                exog=exog_val)
    rmse = np.sqrt(mean_squared_error(ts_val.values, predictions))
    return rmse, predictions
