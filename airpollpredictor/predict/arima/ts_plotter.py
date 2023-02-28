import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, WeekdayLocator
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU


class PlotHelper:

    @staticmethod
    def plot_ts(ts: pd.DataFrame, value_column: str = 'AQI', predict: pd.DataFrame = None,
                date_from=None, date_to=None, title="", legend="", xticks=None):
        '''
        EXAMPLE 1: plot_ts(df_station_sample, title=f'ALL O3 - {"STA-NL00301"}')
        EXAMPLE 2:plot_ts(df_station_sample, date_from='2021-01-01', date_to='2022-01-01', title=f'YEAR O3 - {"STA-NL00301"}')
        :param value_column:
        :param ts: Time series DataFrame
        :param predict: Prediction series (optional)
        :param date_from:
        :param date_to:
        :param title:
        :param legend:
        :return:
        '''
        fig = plt.figure(figsize=(21, 6))
        ax = fig.add_subplot(111)
        if date_from is not None and date_to is not None:
            ts.loc[date_from:date_to].plot(xlabel="Date", ylabel="Concentration", title=title, y=value_column,
                                           c='tab:blue', ax=ax)
            if predict is not None:
                predict.loc[date_from:date_to].plot(c='tab:red', ax=ax)
        else:
            ts.plot(xlabel="Date", ylabel="Concentration", title=title, y=value_column, c='tab:blue', ax=ax)
            if predict is not None:
                predict.plot(c='tab:red', ax=ax)
        plt.legend(legend)
        if xticks == 'm':
            mloc = MonthLocator()
            ax.xaxis.set_major_locator(mloc)
        elif xticks == 'w':
            wloc = WeekdayLocator(byweekday=(MO, SA))
            ax.xaxis.set_major_locator(wloc)
        plt.grid(axis='x', color='green', linestyle='--', linewidth=0.5)
        plt.show()
