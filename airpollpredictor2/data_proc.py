import pandas as pd
import argparse

pol_dict_rev ={'SO2': 1,'PM10': 5,'O3': 7,'NO2': 8,'CO': 10,'PM2.5': 6001}
column_list = ['AirQualityStation', 'SamplingProcess', 'AirPollutant',\
               'AveragingTime', 'Concentration', 'DatetimeBegin', 'DatetimeEnd', 'Validity']
drop_list = ['SamplingProcess', 'AveragingTime', 'DatetimeBegin', 'Validity', 'DatetimeDelta']

### Data_processor script needs to run in terminal with two named arguments
### inputfile - path to data file, outputfile - output file path


### data_proc function accepts filepath referring to csv file
### then reads it, processes the copy and returns a processed copy
def data_proc(data) -> {Copy}:
    df = pd.read_csv(data, usecols=column_list , low_memory=True)
    print(f'Reading data')
    ### Convert pollutant names in main dataset to integer
    for ele in pol_dict_rev.keys():
        df.loc[df['AirPollutant'] == ele, 'AirPollutant'] = pol_dict_rev[ele]
    print(f'Converting pollutant names')
    ### Convert columns into proper format
    df['DatetimeBegin'] = pd.to_datetime(df['DatetimeBegin'], format="%Y-%m-%d %H:%M:%S %z")
    df['DatetimeEnd'] = pd.to_datetime(df['DatetimeEnd'], format="%Y-%m-%d %H:%M:%S %z")
    df['AirPollutant'] = df['AirPollutant'].astype('int64')
    print(f'Converting relevant columns types')
    ### Drop all non-valid data
    df = df[df['Validity'] > 0].copy()
    print(f'Removing non-valid data')
    ### Drop duplicates with keep first parameter
    df = df.drop_duplicates(['AirQualityStation', 'AirPollutant', 'DatetimeEnd'], keep='first')
    print(f'Dropping duplicates')
    ### Replace all negative values with 0.0001
    df.loc[df['Concentration'] < 0, 'Concentration'] = 0.0001
    print(f'Replacing negative values')
    ### Make sure that there are only measurements where TimeDelta equals to 1 hour
    df = df[df['AveragingTime'] == 'hour'].copy()
    df['DatetimeDelta'] = df['DatetimeEnd'] - df['DatetimeBegin']
    df = df[df['DatetimeDelta'] == "0 days 01:00:00"].copy()
    print(f'Selecting data with hourly averaging time')
    ### Drop all unnecessary columns
    df.drop(columns=drop_list, inplace=True)
    print(f'Dropping remaining unnecessary columns')
    return df

### file_exp function saves pandas dataframe in chosen location
### It accepts pandas dataframe as first parameter
### and filepath for output file as second parameter
def data_exp(data, outputfile) -> None:
    df = data.copy()
    ### Export final version of dataset
    df.to_csv(outputfile, index=False, encoding='utf-8-sig')
    print(f'Saving final csv file')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfile", type=str, required=True, help="path to input data")
    parser.add_argument("--outputfile", type=str, required=True, help="path to output data")
    args = parser.parse_args()
    data_exp(data_proc(args.inputfile), args.outputfile)
    print(f'done')
if __name__ == "__main__":
    main()