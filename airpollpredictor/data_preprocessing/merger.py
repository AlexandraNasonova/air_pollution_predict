import pandas as pd
from . import settings


def merge(weather_file_path: str, aqi_file_path: str, output_file_path: str):
    df_weather = pd.read_csv(weather_file_path, index_col=settings.DATE_WEATHER_COLUMN_NAME,
                             parse_dates=True)
    df_aqi = pd.read_csv(aqi_file_path, index_col=settings.DATE_COLUMN_NAME,
                         parse_dates=True)
    df_all = df_aqi.merge(df_weather, how='left', left_index=True, right_index=True)
    df_all.to_csv(output_file_path)
