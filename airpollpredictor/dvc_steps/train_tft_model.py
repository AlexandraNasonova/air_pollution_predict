import pandas as pd

def index_adjust(df: pd.DataFrame, date_column: str, date_index_column: str):
    df.reset_index(inplace=True)
    df[date_index_column] = (df[date_column] - df[date_column].min()).dt.days

