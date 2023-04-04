import os
import pathlib
from datetime import datetime

import pandas as pd


def create_local_data_dir(data_folder) -> str:
    path = os.path.join(os.path.dirname(__file__), data_folder)
    pathlib.Path(path).mkdir(exist_ok=True)

    date_time_now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    path = os.path.join(path, date_time_now)
    pathlib.Path(path).mkdir()
    return path


def create_sub_dir(save_path, sub_dir) -> str:
    path = os.path.join(save_path, sub_dir)
    pathlib.Path(path).mkdir(exist_ok=True)
    return path


async def save_csv_file_from_str(save_path: str, file_name: str, csv_file: str):
    file_path = os.path.join(save_path, f'{file_name}')
    with open(file_path, 'w', encoding='UTF8') as f:
        f.write(csv_file)
    print(f'File {file_name} saved')


async def save_csv_file_from_dataframe(save_path: str, file_name: str, df: pd.DataFrame):
    file_path = os.path.join(save_path, f'{file_name}')
    df.to_csv(file_path)
    print(f'File {file_name} saved')