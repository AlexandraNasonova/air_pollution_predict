import os
from functools import reduce
import pandas as pd
import subindex_calc as sc

### constants
pol_codes = [1, 5, 7, 8, 10, 6001]
pol_dict = {7: "O3", 6001: "PM2.5", 5: "PM10", 10: "CO", 1: "SO2", 8: "NO2"}
pol_dict_rev ={'SO2': 1,'PM10': 5,'O3': 7,'NO2': 8,'CO': 10,'PM2.5': 6001}
pol_units = {1:'µg/m3',5:'µg/m3',7:'µg/m3',8:'µg/m3',10:'mg/m3',6001:'µg/m3'}

### Code to convert data into hourly timeseries format
def create_timeseries(df):
    df_mean_list = list()
    for i in range(len(pol_codes)):
        df_mean_list.append(df[df['AirPollutant']==pol_codes[i]].groupby('DatetimeEnd', as_index=False)['Concentration'].mean())
        df_mean_list[i].rename(columns={'Concentration': pol_dict[pol_codes[i]]}, inplace=True)
        timeseries = reduce(lambda df1, df2: df1.merge(df2, how='outer', left_on=['DatetimeEnd'], right_on=['DatetimeEnd']), df_mean_list)
        timeseries = timeseries.set_index('DatetimeEnd')
        timeseries = timeseries.asfreq('H')
    return timeseries

### Measurement converter (converts them into units used to calculate subindices)
def converter(df_temp):
    df = df_temp
    df.iloc[:, 0:1] = df.iloc[:, 0:1]/2.664  # SO2 = 1000*0.001*(ug/m3)/2.664 = (ug/m3)/2.664
    df.iloc[:, 2:3] = df.iloc[:, 2:3]/1996   # O3 = (ug/m3)/(1.996*1000)
    df.iloc[:, 3:4] = df.iloc[:, 3:4]/1.913  # NO2 = 1000*0.001*(ug/m3)/1.913 = (ug/m3)/1.913
    df.iloc[:, 4:5] = df.iloc[:, 4:5]/1.165  # CO = (mg/m3)/1.165
    return df

### Function which "rolls" measurements in a specific intervals as required by AQI-document
def roller(df_temp):
    df = df_temp
    df['1_1H'] = df.iloc[:, 0:1]                                              # SO2    1H
    df['5_24H'] = df.iloc[:, 1:2].rolling(window=24, min_periods=1).mean()    # PM10  24H
    df['7_8H'] = df.iloc[:, 2:3].rolling(window=8, min_periods=1).mean()      # O3     8H
    df['7_1H'] = df.iloc[:, 2:3]                                              # O3     1H
    df['8_1H'] = df.iloc[:, 3:4]                                              # NO2    1H
    df['10_8H'] = df.iloc[:, 4:5].rolling(window=24, min_periods=1).mean()    # CO     8H
    df['6001_24H'] = df.iloc[:, 5:6].rolling(window=24, min_periods=1).mean() # PM2.5 24H
    return df

### Function which calculates subindices for each pollutant
def calculator(df_temp):
    df = df_temp
    df['aqi_1'] = sc.aqi_easy_1(df, 0)
    df['aqi_5'] = sc.aqi_easy_5(df, 1)
    df['aqi_7_8H'] = sc.aqi_easy_7_8H(df, 2)
    df['aqi_7_1H'] = sc.aqi_easy_7_1H(df, 2)
    df['aqi_8'] = sc.aqi_easy_8(df, 3)
    df['aqi_10'] = sc.aqi_easy_10(df, 4)
    df['aqi_6001'] = sc.aqi_easy_6001(df, 5)
    return df

### Function which calculates final AQI from previously calculated subindices
def combiner(df_temp):
    df = df_temp
    df = df.iloc[:,13:20]
    df['aqi'] = df.iloc[:,:7].max(axis=1)
    df = df.iloc[:,7:8]
    return df

### Function which creates daily (24H) AQI values using hourly (1H) AQI
def grouper(df_temp):
    df = df_temp
    df_index = pd.DataFrame(index=df.groupby(pd.Grouper(freq="24H")).count().index)
    df = df_index.merge(df.groupby(pd.Grouper(freq="24H")).mean(), left_index=True, right_index=True)
    return df

### Function to fill gaps in timeseries by median
def filler(df_temp):
    df = df_temp
    df = df.fillna(df.median())
    return df

### Function to round calculated AQI
def rounder(df_temp):
    df = df_temp
    df['aqi'] = round(df['aqi'], 0)
    return df

### Combined super-function creating timeseries with final 24H AQI values from initial timeseries
def aqi(df):
    df = rounder(filler(grouper(combiner(calculator(roller(converter(df)))))))
    return df


def main() -> None:
    df = pd.read_csv('denmark_v5.csv', low_memory=False)
    df = df.drop_duplicates(['AirQualityStation','AirPollutant','DatetimeEnd'], keep='last')
    station = df.groupby('AirQualityStation')['Concentration'].count().reset_index()\
        .sort_values(by='Concentration', ascending=False).head(1).reset_index()['AirQualityStation'][0]
    df_v1 = df[df['AirQualityStation']==station].copy()
    df_v1['DatetimeEnd'] = pd.to_datetime(df_v1['DatetimeEnd'], format="%Y-%m-%d %H:%M:%S")
    df_ts = create_timeseries(df_v1)
    ts_aqi = aqi(df_ts)
    ts_aqi2 = ts_aqi.copy().reset_index()
    os.chdir(os.getcwd())
    ts_aqi2.to_csv( "ts_final.csv", index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    main()