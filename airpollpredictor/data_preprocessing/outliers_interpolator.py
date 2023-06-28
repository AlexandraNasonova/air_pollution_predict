"""
DVC Stage interpolate-outliers - interpolates outliers
"""

# pylint: disable=E0401

import numpy as np
import pandas as pd
import settings.settings as settings
import data_preprocessing.columns_filter as col_filters


def __get_quantiles(df: pd.DataFrame, column: str, iqr_coef_lower=1.5
                    , iqr_coef_upper=1.5):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_coef_lower * iqr
    upper = q3 + iqr_coef_upper * iqr
    return lower, upper


def __get_outlier_indices(df: pd.DataFrame, column: str, lower: float,
                          upper: float):
    upper_array = df[df[column] >= upper].index
    lower_array = df[df[column] <= lower].index
    return lower_array, upper_array


def interpolate_outliers_for_pollutant(df: pd.DataFrame, column: str,
                                       iqr_coef_lower=1.5,
                                       iqr_coef_upper=1.5
                                       ):
    """
    Interpolates outliers in the dataframe for the required column
    @param iqr_coef_lower: Coefficient for lower iqr interval
    @param iqr_coef_upper: Coefficient for upper iqr interval
    @param df: Dataframe for interpolation outliers
    @param column: Column name
    """
    lower, upper = __get_quantiles(df, column, iqr_coef_lower, iqr_coef_upper)
    lower_array, upper_array = __get_outlier_indices(df, column, lower, upper)
    df.loc[upper_array, column] = np.nan
    df.loc[lower_array, column] = np.nan
    print(f"Outliers quantity: {df[column].isna().sum()}")
    df[column] = df[column].interpolate(method='time')
    df[column].fillna(0, inplace=True)
    print(f'DEBUG fillna, column: {column}, sum_na {df[column].isna().sum()}')


def interpolate_outliers(pollutants_codes: [int],
                         df_list: list[pd.DataFrame],
                         iqr_borders: dict):
    """
    Interpolates outliers in target AQI columns.
    @param pollutants_codes: The list of pollutant codes
    @param df_list: List of dataframes per pollutant
    @param iqr_borders: Dictionary per pollutant with 2-value arrays for IQR coefficients
    """
    for i in range(len(pollutants_codes)):
        pollutant_id = pollutants_codes[i]
        for pred_value_type in settings.PREDICTION_VALUE_TYPES:
            # column_name = col_filters.get_target_column(
            #     pred_value_type, pollutant_id)
            iqrs = iqr_borders[pollutant_id]
            interpolate_outliers_for_pollutant(df_list[i],
                                               pred_value_type,
                                               iqrs[0], iqrs[1])
