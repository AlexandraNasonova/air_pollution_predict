# pylint: disable=E0401, R0913, R0914, W0703
"""Timeseries plotter"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, WeekdayLocator
from matplotlib.dates import MO, SA


def plot_ts(df_timeseries: pd.DataFrame, value_column: str, predict: pd.DataFrame = None,
            date_from=None, date_to=None, title="", legend="", xticks="") -> None:
    """
    EXAMPLE 1: plot_ts(df_station_sample, title=f'ALL O3 - {"STA-NL00301"}')
    EXAMPLE 2:plot_ts(df_station_sample, date_from='2021-01-01',
    date_to='2022-01-01', title=f'YEAR O3 - {"STA-NL00301"}')
    :param value_column: The value column name
    :param df_timeseries: Time series DataFrame
    :param predict: Prediction series (optional)
    :param date_from: The first date for the plot
    :param date_to: The last date for the plot
    :param title: The title of the plot
    :param legend: The legend text for the plot
    :param xticks: Period type for the ticks on the x-axis (m-month, w-week)
    """
    fig = plt.figure(figsize=(21, 6))
    axs = fig.add_subplot(111)
    if date_from is not None and date_to is not None:
        df_timeseries.loc[date_from:date_to].plot(xlabel="Date", ylabel="Concentration",
                                                  title=title, y=value_column,
                                                  c='tab:blue', ax=axs)
        if predict is not None:
            predict.loc[date_from:date_to].plot(c='tab:red', ax=axs)
    else:
        df_timeseries.plot(xlabel="Date", ylabel="Concentration", title=title,
                           y=value_column, c='tab:blue', ax=axs)
        if predict is not None:
            predict.plot(c='tab:red', ax=axs)
    plt.legend(legend)
    if xticks == 'm':
        mloc = MonthLocator()
        axs.xaxis.set_major_locator(mloc)
    elif xticks == 'w':
        wloc = WeekdayLocator(byweekday=(MO, SA))
        axs.xaxis.set_major_locator(wloc)
    plt.grid(axis='x', color='green', linestyle='--', linewidth=0.5)
    plt.show()
