from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import numpy as np
import pandas as pd


def cross_val_sarimax_calc_rmse(ts_train, order, seasonal_order, exog_train, n_splits=16, ):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse = []
    all_predictions = None
    for train_index, test_index in tscv.split(ts_train):
        cv_train, cv_test = ts_train.iloc[train_index], ts_train.iloc[test_index]
        cv_exog_train, cv_exog_test = None, None
        if exog_train is not None:
            cv_exog_train, cv_exog_test = exog_train.iloc[train_index], exog_train.iloc[test_index]
        cv_model = sm.tsa.statespace.SARIMAX(cv_train, order=order, seasonal_order=seasonal_order,
                                             exog=cv_exog_train).fit(disp=0)
        predictions = cv_model.predict(start=cv_train.shape[0], end=cv_train.shape[0] + cv_test.shape[0] - 1,
                                       exog=cv_exog_test)
        if all_predictions is None:
            all_predictions = predictions
        else:
            all_predictions = pd.concat([all_predictions, predictions], axis=0)
        rmse.append(np.sqrt(mean_squared_error(cv_test.values, predictions)))
    rmse_mean = np.mean(rmse)
    return rmse_mean, all_predictions


def predict_sarimax_calc_rmse(ts_train, ts_val, order, seasonal_order, exog_train, exog_val):
    model = sm.tsa.statespace.SARIMAX(ts_train, order=order, seasonal_order=seasonal_order, exog=exog_train).fit(disp=0)
    predictions = model.predict(start=ts_train.shape[0], end=ts_train.shape[0] + ts_val.shape[0] - 1, exog=exog_val)
    rmse = np.sqrt(mean_squared_error(ts_val.values, predictions))
    return rmse, predictions
