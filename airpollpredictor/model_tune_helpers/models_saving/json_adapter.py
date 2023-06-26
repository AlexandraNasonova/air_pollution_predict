"""Adapter to save/read metrics from json"""

# pylint: disable=W1514

import json


def save_params_to_json(file_path: str, params: dict):
    """
    Save metrics to json
    @param file_path: Params file name
    @param params: Parameters
    """
    with open(file_path, 'w') as f_stream:
        json.dump(params, f_stream)


def save_metrics_to_json(file_path: str, train_score: float,
                         val_score: float, metric_name: str):
    """
    Save metrics to json
    @param file_path: Metrics file name
    @param train_score: Train score value
    @param val_score: Val score value
    @param metric_name: The name of the metric
    """
    with open(file_path, 'w') as f_stream:
        json.dump({
            f'train_{metric_name}': train_score,
            f'val_{metric_name}': val_score,
        }, f_stream)


def read_from_json(file_path: str) -> dict:
    """
    Reads from json file
    @param file_path: Path to file
    @return: Dictionary with metrics
    """
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data
