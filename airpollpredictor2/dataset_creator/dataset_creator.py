import os
import sys
import glob
import pandas as pd

### To use run *.py file with parameters separated by space
### Do not mix countrycodes, localIDs or citynames together
### The current script directory should contain data folder
### Also dataset_by_city function requires final_metadata.csv
### Datasets are created in current script directory


### Function to create a list of csv files within data folder
def folder_scanner() -> list[str|None]: ### get list of csv files in data
    list_of_csv_files = glob.glob(os.getcwd() + '/data' + '/**/*.csv', recursive=True)
    return list_of_csv_files

### Function to extract list of countrycodes from filenames within data
def get_countrycodes() -> list[str|None]: ### extract countrycodes from data
    csv1 = folder_scanner()
    codes = []
    for file in csv1:
        codes.append(file.split('\\')[-1][:2])
    list_of_codes = list(set(codes))
    return list_of_codes

### The following 3 functions are used to merge all csv files
### corresponding to specified countries, localids or cities
def dataset_by_country(*codes) -> None:  ### pass countrycodes separated by comma
    codelist = [*codes]
    csv1 = folder_scanner()
    print(csv1[:4])
    csv2 = []
    for file in csv1:
        for code in codelist:
            if str(code).upper() in file:
                csv2.append(file)
    df = pd.concat([pd.read_csv(f) for f in csv2])
    df.to_csv( "df_" + '_'.join(code for code in codelist) + ".csv", index=False, encoding='utf-8-sig')
    return

def dataset_by_localid(*codes) -> None:  ### pass localcodes separated by comma
    codelist = [*codes]
    csv1 = folder_scanner()
    print(len(csv1))
    csv3 = []
    for file in csv1:
        if file.split('_')[-3] in codelist:
            csv3.append(file)
    print(len(csv3))
    df = pd.concat([pd.read_csv(file) for file in csv3])
    df.to_csv( "df_" + '_'.join(code for code in codelist) + ".csv", index=False, encoding='utf-8-sig')
    return

def dataset_by_city(*names) -> None:     ### pass citynames separated by comma
    dfm = pd.read_csv("final_metadata.csv", low_memory=False)
    citynames = [*names]
    localids = []
    csv1 = folder_scanner()
    csv4 = []
    cities_lower = []
    cities_regular = []
    for city in list(dfm.query('StationCity==StationCity')['StationCity'].unique()):
        cities_lower.append(city.lower().split(' ')[0])
        cities_regular.append(city)
    for i in range(len(cities_lower)):
        for cityname in citynames:
            if cityname.lower() in cities_lower[i]:
                localids.extend(list(dfm[(dfm['StationCity']==cities_regular[i]) & (dfm['LocalCode'].notna())]['LocalCode'].unique()))
    for ele in localids:
        for file in csv1:
            if int(ele) == int(file.split('_')[-3]):
                csv4.append(file)
    df = pd.concat([pd.read_csv(f) for f in csv4])
    df.to_csv( "df_" + '_'.join(cityname for citynames in citynames) + ".csv", index=False, encoding='utf-8-sig')
    return

def main() -> None:
    arg_list = list(sys.argv[1:])
    arg = arg_list[0]
    print(arg_list)
    if str(arg).isdigit() == True:
        dataset_by_localid(*arg_list)
    else:
        if len(arg)==2:
            dataset_by_country(*arg_list)
        else:
            dataset_by_city(*arg_list)

if __name__ == "__main__":
    main()