import pandas as pd


def add_date_info(df: pd.DataFrame) -> pd.DataFrame:
    df["weekday"] = df.index.weekday
    df["day"] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['season'] = df.index.to_series().apply(
        lambda x: 0 if x.month < 3 else 1 if x.month < 6 else 2 if x.month < 9 else 3)
    df['is_weekend'] = [df['weekday'].isin([5, 6])][0] * 1
    # df['holiday'] = [df.index.isin(pd.to_datetime(holidays).date)][0]*1

    # nieuwe jaar vuurwerken
    df['is_new_year'] = [(df['month'] == 1) & (df['day'] == 1)][0] * 1
    return df
