# pylint: disable=E0401, R0913, R0914, W0703

"""
Helper for .csv file saving and directories creation
"""
import os
import pathlib
from datetime import datetime
import pandas as pd


def create_local_data_dir(data_folder: str) -> str:
    """
    Creates the directory in the same folder where the current file is
    @param data_folder: The name of the required directory
    @return: he full path to the created directory
    """
    path = os.path.join(os.path.dirname(__file__), data_folder)
    pathlib.Path(path).mkdir(exist_ok=True)

    date_time_now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    path = os.path.join(path, date_time_now)
    pathlib.Path(path).mkdir()
    return path


def create_sub_dir(save_path: str, sub_dir: str) -> str:
    """
    Create directory by the required path
    @param save_path: The path to the directory
    @param sub_dir: The name of the subdirectory
    @return: The full path to the created subdirectory
    """
    path = os.path.join(save_path, sub_dir)
    pathlib.Path(path).mkdir(exist_ok=True)
    return path


async def save_csv_file_from_str(save_path: str, file_name: str, csv_file: str) -> None:
    """
    Save string with csv file data to .csv file
    @param save_path: Path to the directory
    @param file_name: The required .csv file name
    @param csv_file: Dataframe for saving
    """
    file_path = os.path.join(save_path, f'{file_name}')
    with open(file_path, 'w', encoding='UTF8') as file_stream:
        file_stream.write(csv_file)
    print(f'File {file_name} saved')


async def save_csv_file_from_dataframe(save_path: str, file_name: str,
                                       df_to_save: pd.DataFrame) -> None:
    """
    Save dataframe to csv file
    @param save_path: Path to the directory
    @param file_name: The required .csv file name
    @param df_to_save: Dataframe for saving
    """
    file_path = os.path.join(save_path, f'{file_name}')
    df_to_save.to_csv(file_path)
    print(f'File {file_name} saved')
