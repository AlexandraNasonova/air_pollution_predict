import os
import pandas as pd
from .aqi_calculations import aqi_calculator as aqc
# import aqi_calculations.aqi_calculator as aqc
from . import settings


def calculate_aqi_indexes(source_data_path: str, pollutants_codes: list[int]):
    df_aqi_list = list()
    for i in range(len(pollutants_codes)):
        pollutant_id = pollutants_codes[i]
        measure = settings.POL_MEASURES[pollutant_id]
        df_source = pd.read_csv(os.path.join(source_data_path, f'{pollutant_id}.csv'),
                                parse_dates=True, index_col=settings.DATE_COLUMN_NAME)
        df_aqi_list.append(aqc.calc_aqi_for_day_pd(pollutant_id, df_source, measure))
    return df_aqi_list


def save_aqi_data(pollutants_codes: list[int], df_aqi_list: list[pd.DataFrame],
                  output_path: str):
    for i in range(len(pollutants_codes)):
        file_path = os.path.join(output_path, f'{pollutants_codes[i]}.csv')
        df_aqi_list[i].to_csv(file_path)
