"""FastApi service, which returns predictions and the last evaluation metrics. \
Currently it predicts only PM2.5 pollutant in Rotterdam for the last 3 days.
Later it will return predictions made by different models for different \
pollutants for the next 3 days"""

# pylint: disable=E0401, E0611, W1514:

import json
from fastapi import FastAPI
import pandas as pd
from model_tune_helpers.models_saving import onnx_adapter, json_adapter
from settings import settings
# import model_tune_helpers.onnx_wrapper as onnx_wrapper
# import settings.settings as settings

app = FastAPI()

X_VAL_DATASET_PATH = "experiments_results/lgbm/6001/val.csv"
METRICS_PATH = "experiments_results/lgbm/6001/metrics.json"
MODEL_PATH = "experiments_results/lgbm/6001/model.onnx"


@app.get("/")
def root():
    """
    Function to ping service. Returns Ok
    @return: Ok
    """
    return 'Ok'


@app.get("/predict")
def predict():
    """
    Return predictions for PM2.5 for the last 3 days.
    Source data are loaded and proceeded by scheduler.
    The model is automatically retrained afterward.
    @return: Array with 3 float values
    """
    df_val = pd.read_csv(X_VAL_DATASET_PATH, parse_dates=True,
                         index_col=settings.DATE_COLUMN_NAME)
    df_val.drop(columns=['AQI_PM25'], axis=0, inplace=True)
    y_pred = onnx_adapter.predict_model(x_df=df_val, onnx_file_path=MODEL_PATH)
    result = json.dumps(list(map(float, y_pred.reshape(3))))
    return result


@app.get("/evaluate")
def evaluate():
    """
    Return the metrics of the last evaluation for PM2.5 for the last 3 days.
    Source data are loaded and proceeded by scheduler.
    The model is automatically retrained afterward.
    @return: Dictionary with keys train_[metric_name] and val__[metric_name]
    """
    return metrics_adapter.read_metrics_from_json(METRICS_PATH)
