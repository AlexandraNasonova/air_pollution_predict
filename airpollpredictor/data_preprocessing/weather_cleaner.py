import pandas as pd
from . import settings


def clean(source_file_path: str, columns: list[str], output_file_path: str):
    df_weather = pd.read_csv(source_file_path, usecols=columns + [settings.DATE_WEATHER_COLUMN_NAME],
                             index_col=settings.DATE_WEATHER_COLUMN_NAME, parse_dates=True)
    df_weather.to_csv(output_file_path)
