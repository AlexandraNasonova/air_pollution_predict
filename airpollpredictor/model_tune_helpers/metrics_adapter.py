"""Adapter to save/read metrics from json"""

# pylint: disable=W1514

import json


def save_metrics_to_json(metrics_file_path: str, train_score: float, val_score: float,
                         metric_name: str):
    """
    Save metrics to json
    @param metrics_file_path: Metrics file name
    @param train_score: Train score value
    @param val_score: Val score value
    @param metric_name: The name of the metric
    """
    with open(metrics_file_path, 'w') as f_stream:
        json.dump({
            f'train_{metric_name}': train_score,
            f'val_{metric_name}': val_score,
        }, f_stream)


def read_metrics_from_json(metrics_file_path: str) -> dict:
    """
    Reads metrics from json file
    @param metrics_file_path: Metrics file name
    @return: Dictionary with metrics
    """
    with open(metrics_file_path) as json_file:
        metrics = json.load(json_file)
    return metrics
