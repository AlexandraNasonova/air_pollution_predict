import datetime
import os
import pathlib
import time
import requests

URL_POLLUTANT_LIST = "https://fme.discomap.eea.europa.eu/fmedatastreaming/AirQualityDownload/AQData_Extract.fmw?CountryCode=$country$&CityName=&Pollutant=$pollutant_id$&Year_from=$year_from$&Year_to=$year_to$&Station=&Samplingpoint=&Source=All&Output=TEXT&UpdateDate=&TimeCoverage=Year"
RELOAD_COUNTER = 2
RELOAD_REPEATING_COUNTER = 5
POLLUTANTS_CODES = [7, 6001, 5, 10, 1, 8]


def create_new_dir():
    date_time_now = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    path = os.path.join(os.path.dirname(__file__), "data", date_time_now)
    pathlib.Path(path).mkdir()
    return path


def create_new_pollutant_dir(save_path, pollutant_id):
    path = os.path.join(save_path, pollutant_id)
    pathlib.Path(path).mkdir(exist_ok=True)
    return path


async def pollutants_txt_lists_load(country=None, year_from=2013, year_to=datetime.datetime.now().year) -> str:
    save_path = create_new_dir()
    print(f"Directory {save_path} created")
    for i in range(len(POLLUTANTS_CODES)):
        url = URL_POLLUTANT_LIST.replace("$pollutant_id$", str(POLLUTANTS_CODES[i]))
        url = url.replace("$country$", ('' if country is None else country))
        url = url.replace("year_from$", str(year_from))
        url = url.replace("year_to", str(year_to))
        await txt_file_load_and_save(f'{str(POLLUTANTS_CODES[i])}.txt', save_path, url)
    return save_path


async def txt_file_load_and_save(file_name, save_path, url) -> None:
    print(f'Request TXT {url} sent')
    response = requests.get(url)
    print(f'Response TXT status_code: {response.status_code}')
    txt_file = str(response.content, encoding='UTF8')
    if response.status_code != 200:
        log_error(save_path, url, txt_file)
        raise ConnectionError()
    file_path = os.path.join(save_path, f'{file_name}')
    with open(file_path, 'w', encoding='UTF8') as f:
        f.write(txt_file)
    print(f'File {file_name} saved')


async def csv_list_load(save_path) -> None:
    txt_lists = list(pathlib.Path(save_path).glob('*.txt'))
    print(f'Loaded {len(txt_lists)} txt lists')
    repeated_reloads_for_dif_files = 0
    for txt_list in txt_lists:
        f = open(txt_list, "r", encoding='utf-8-sig')
        txt_content = f.read()
        csv_list = txt_content.split()
        print(f'Loaded {len(csv_list)} csv urls for file {txt_list}')
        pollutant_id = os.path.splitext(os.path.basename(txt_list))[0]
        sub_path = create_new_pollutant_dir(save_path, pollutant_id)
        for csv_url in csv_list:
            file_name = csv_url.split('/')[-1:][0]
            for i in range(RELOAD_COUNTER + 1):
                if i == RELOAD_COUNTER:
                    repeated_reloads_for_dif_files += 1
                    log_error(sub_path, csv_url)
                    break
                try:
                    await csv_file_load_and_save(file_name, sub_path, csv_url)
                    break
                except Exception as error:
                    print("SHORT SLEEP")
                    time.sleep(30)

            if repeated_reloads_for_dif_files == RELOAD_REPEATING_COUNTER:
                time.sleep(60 * 60)
                print("LONG LONG SLEEP")
                repeated_reloads_for_dif_files = 0


async def csv_file_load_and_save(file_name, save_path, url) -> None:
    print(f'Request CSV {url} sent')
    response = requests.get(url, timeout=120)
    csv_file = str(response.content, encoding='UTF8')
    print(f'Response CSV status_code: {response.status_code}')
    file_path = os.path.join(save_path, f'{file_name}')
    with open(file_path, 'w', encoding='UTF8') as f:
        f.write(csv_file)
    print(f'File {file_name} saved')


def log_error(sub_path, url, text=None):
    path = os.path.join(sub_path, "log.txt")
    with open(path, 'a') as f:
        f.write(url + "\n")
        if text is not None:
            f.write("\n")
            f.write(text + "\n")
