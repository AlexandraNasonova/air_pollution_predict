"""Adapter to save/read metrics from joblib"""

# pylint: disable=W1514

import joblib


def save_model(model_file_path: str, model):
    """
    Save model to joblib
    @param model_file_path: Model pkl file path
    @param model: Model
    """
    joblib.dump(model, model_file_path)


def predict_model(model_file_path: str, forecast_period: int):
    """
    Reads metrics from json file
    @param model_file_path: Metrics file name
    @param forecast_period: Forecast period in days
    @return: Dictionary with metrics
    """
    predictions = joblib.load(model_file_path).predict(n_periods=forecast_period)
    return predictions