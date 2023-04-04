import os
import glob
import time
import pathlib
import requests
import pandas as pd
from io import StringIO

### This is discomap URL template which can be modified according to our needs
discomap = 'https://fme.discomap.eea.europa.eu/fmedatastreaming/AirQualityDownload/AQData_Extract.fmw?CountryCode={Countrycode}&CityName={StationCity}&Pollutant={AirPollutantCode}&Year_from={Year_from}&Year_to={Year_to}&Station={AirQualityStation}&EoICode={AirQualityStationEoICode}&Samplingpoint=&Source=All&Output=TEXT&UpdateDate=&TimeCoverage=Year'


### Url creator function creates a list of 6 URLs corresponding for each pollutant
### for subsequent downloading of lists of csv files from Discomap server.
### The most useful arguments to specify are "Countrycode" and "AirQualityStation"
### If these arguments are empty then there will be too much data and total size
### of the corresponding data will be in the range of several hundreds gigabytes
def url_creator(discomap_url=discomap, \
                Countrycode='DK', \
                StationCity='', \
                AirPollutantCode='', \
                Year_from='2013', \
                Year_to='2023', \
                AirQualityStation='', \
                # AirQualityStation='STA-DK0034A', \
                AirQualityStationEoICode='') -> list[str]:
    pol_codes = [1, 5, 7, 8, 10, 6001]
    url_list1 = []
    for i in range(len(pol_codes)):
        temp = discomap.format(Countrycode=Countrycode, \
                               StationCity=StationCity, \
                               AirPollutantCode=pol_codes[i], \
                               Year_from=Year_from, \
                               Year_to=Year_to, \
                               AirQualityStation=AirQualityStation, \
                               AirQualityStationEoICode=AirQualityStationEoICode)
        url_list1.append(temp)
    return url_list1

### Function to download a lists of links to csv files
def csv_list_loader(data_list) -> list[str]:
    resp = requests.get('https://discomap.eea.europa.eu/map/fme/AirQualityExport.htm')
    csv_rawlist = []
    if resp.status_code == 200:
        timeout_counter = 0
        for i in range(len(data_list)):
            try:
                temp1 = requests.get(data_list[i], timeout=(3.07, 120)).text[3:]
                csv_rawlist.append(temp1)
                time.sleep(3)
            except requests.Timeout as err:
                print(f'Timeout during download #{i+1}')
                csv_rawlist.append("ERROR")
                timeout_counter = timeout_counter + 1
                if timeout_counter>2:
                    print(f'Discomap is temporarily unavailable for downloads')
                    raise err
    else:
        print ('Error. Cannot download a list of links to csv files')
        return
    return csv_rawlist

### Function to extract links to csv files into one list
def link_extractor(csv_rawlist) -> list[str]:
    temp_list3 = []
    csv_list = []
    for i in range(len(csv_rawlist)):
        if csv_rawlist[i] !='ERROR':
            temp_list3.append(csv_rawlist[i].split())
    for i in range(len(temp_list3)):
        csv_list = csv_list + temp_list3[i]
    try:
        assert len(csv_list)!=0
    except AssertionError as err:
        print(f'There is no data to download. Aborting. Check if the arguments are valid')
        raise err
    print(f'There are {len(csv_list)} csv files to download')
    return csv_list

### Function to download csv files and save them
def csv_downloader(csv_list) -> dict[str, str]:
    pathlib.Path(os.getcwd() + '/data').mkdir(parents=True, exist_ok=True)
    download_status_map = {}
    error_link_list = []
    print(f'Attempting to download')
    for csv_link in csv_list:
        try:
            csv_file = str(requests.get(csv_link, timeout=(3.07, 120)).content, encoding='UTF8')
            file_path = os.path.join(pathlib.Path(os.getcwd() + '/data'), f'{csv_link.split("/")[-1]}')
            with open(file_path, 'w', encoding='UTF8') as f:
                f.write(csv_file)
            download_status_map[csv_link]='ok'
            time.sleep(3)
        except:
            download_status_map[csv_link]='error'
            error_link_list.append(csv_link)
    if len(error_link_list)==0:
        print(f'There were no errors during this download attempt')
        return download_status_map
    time.sleep(61)
    for csv_link in error_link_list:
        try:
            csv_file = str(requests.get(csv_link, timeout=(3.07, 120)).content, encoding='UTF8')
            file_path = os.path.join(pathlib.Path(os.getcwd() + '/data'), f'{csv_link.split("/")[-1]}')
            with open(file_path, 'w', encoding='UTF8') as f:
                f.write(csv_file)
            download_status_map[csv_link]='ok'
            time.sleep(3)
        except:
            download_status_map[csv_link]='error'
    download_status_log = ''
    print(f'There may be unfinished downloads. Exporting download status log')
    for key in list(download_status_map.keys()):
        download_status_log = download_status_log + key + ' : ' + download_status_map[key] + '\n'
    with open(pathlib.Path(os.getcwd() + '/data' + '/download_status_log.txt'), 'w', encoding='UTF8') as f:
                f.write(download_status_log)
    return download_status_map

### Combine all previously downloaded csv files into one dataframe
### This dataframe will contain extra column with information about
### the source of data, if source argument is specified as "True"
def csv_combiner(source=False) -> None:
    print(f'Combining downloaded csv files in data folder into one dataframe')
    if source==False:
        df = pd.concat(map(pd.read_csv, \
                           glob.glob(os.path.join(pathlib.Path(os.getcwd() + '/data')) + '/**/*.csv', recursive=True)))
        return df
    splitby = "/"
    if os.name=='nt':
        splitby = "\\"
    df = pd.concat(
        [
            pd.read_csv(filename).assign(source=filename.split(f"{splitby}")[-1])
            for filename in glob.glob(os.path.join(pathlib.Path(os.getcwd() + '/data')) + '/**/*.csv', recursive=True)
        ],
        ignore_index=True
    )
    print(f'done')
    return df

### Function to download data into dictionary
### without saving csv files on a disk
def stringio_loader(csv_list) -> None:
    download_status_map = {}
    csv_contents_map = {}
    error_link_list = []
    print(f'Attempting to download')
    for csv_link in csv_list:
        try:
            csv_file = str(requests.get(csv_link, timeout=(3.05, 120)).content, encoding='UTF8')
            csv_contents_map[csv_link] = csv_file
            download_status_map[csv_link]='ok'
            time.sleep(3)
        except:
            download_status_map[csv_link]='error'
            error_link_list.append(csv_link)
    if len(error_link_list)==0:
        print(f'There were no errors during this download attempt')
        return csv_contents_map
    time.sleep(61)
    for csv_link in error_link_list:
        try:
            csv_file = str(requests.get(csv_link, timeout=(3.05, 120)).content, encoding='UTF8')
            csv_contents_map[csv_link] = csv_file
            download_status_map[csv_link]='ok'
            time.sleep(3)
        except:
            download_status_map[csv_link]='error'
    download_status_log = ''
    print(f'There may be unfinished downloads. Exporting download status log')
    for key in list(download_status_map.keys()):
        download_status_log = download_status_log + key + ' : ' + download_status_map[key] + '\n'
    with open(pathlib.Path(os.getcwd() + '/download_status_log.txt/'), 'w', encoding='UTF8') as f:
                f.write(download_status_log)
    return csv_contents_map

### Combine all downloaded csv data within dict into dataframe
### This dataframe will contain extra column with information about
### the source of data, if source argument is specified as "True"
def stringio_combiner(dict1, source=False) -> None:
    print(f'Combining downloaded csv files within "csv_contents_map" into one dataframe')
    csv_contents_map = dict1
    if source==False:
        df = pd.concat([pd.read_csv(StringIO(ele)) for ele in list(csv_contents_map.values())])
        return df
    splitby = "/"
    df = pd.concat(
        [
            pd.read_csv(StringIO(csv_contents_map[key])).assign(source=key.split(f"{splitby}")[-1])
            for key in list(csv_contents_map.keys())
        ],
        ignore_index=True
    )
    print(f'done')
    return df

### Combined superfunction to create dataset and save downloaded csv files
def ez_data_retriever_v1(Countrycode, Year_from, Year_to, AirQualityStation, source1) -> None:
    csv_downloader(link_extractor(csv_list_loader(url_creator(Countrycode=Countrycode, Year_from=Year_from, Year_to=Year_to, AirQualityStation=AirQualityStation))))
    df46 = csv_combiner(source=source1)
    return df46

### Combined superfunction to create dataset without saving downloaded csv files
def ez_data_retriever_v2(Countrycode, Year_from, Year_to, AirQualityStation, source1) -> None:
    df45 = stringio_combiner(stringio_loader(link_extractor(csv_list_loader(url_creator(Countrycode=Countrycode, Year_from=Year_from, Year_to=Year_to, AirQualityStation=AirQualityStation)))),source=source1)
    return df45

### Script can be run in terminal with required parameters
### By default downloads all data since 2013 from Denmark
### While all parameters are optional, it is highly recommended
### to at least use countrycode and station to narrow it down

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--countrycode", type=str, required=False, help="A two capital letter code")
    parser.add_argument("--years_from", type=str, required=False, help="A year from which to include data")
    parser.add_argument("--years_to", type=str, required=False, help="A year up to which to include data")
    parser.add_argument("--station", type=str, required=False, help="AirQualityStation, for example 'STA-DK0034A'")
    parser.add_argument("--save_csv", type=str, required=False, help="specify this if csv files need to be saved")
    parser.add_argument("--source", type=bool, required=False, help="specify this if you need source within df")
    args = parser.parse_args()
    params = {'Countrycode':'DK', 'Year_from':'2013', 'Year_to':'2023', 'AirQualityStation':'', 'source1':True}
    if args.countrycode:
        params['Countrycode'] = args.countrycode
    if args.years_from:
        params['Year_from'] = args.years_from
    if args.years_to:
        params['Year_to'] = args.years_to
    if args.station:
        params['AirQualityStation'] = args.station
    if args.source:
        params['source1'] = args.source
    if not args.save_csv:
        ez_data_retriever_v2(**params)
        print(f'done')
        return
    ez_data_retriever_v1(**params)
    print(f'done')
    return

if __name__ == "__main__":
    main()