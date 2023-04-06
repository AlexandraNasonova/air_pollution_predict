"""Module for date features extraction"""
# pylint: disable=E0401

import pandas as pd


def add_date_info(df_concentrations: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the date from the dataframe and
    adds date features (weekday, day, month, year, season, is_weekend, is_new_year).
    @param df_concentrations: Dataframe with concentrations and dates
    @return:
    """
    df_concentrations["weekday"] = df_concentrations.index.weekday
    df_concentrations["day"] = df_concentrations.index.day
    df_concentrations['month'] = df_concentrations.index.month
    df_concentrations['year'] = df_concentrations.index.year
    df_concentrations['season'] = df_concentrations.index.to_series().apply(
        lambda x: 0 if x.month < 3 else 1 if x.month < 6 else 2 if x.month < 9 else 3)
    df_concentrations['is_weekend'] = [df_concentrations['weekday'].isin([5, 6])][0] * 1
    # df['holiday'] = [df.index.isin(pd.to_datetime(holidays).date)][0]*1

    # the feature to handle new year fireworks
    df_concentrations['is_new_year'] = \
        [(df_concentrations['month'] == 1) & (df_concentrations['day'] == 1)][0] * 1
    return df_concentrations
