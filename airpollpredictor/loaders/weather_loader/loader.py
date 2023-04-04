import datetime
import requests
from loaders.logger import log_error
from loaders.file_saver import create_local_data_dir, save_csv_file_from_dataframe
import pandas as pd

URL_DAILY_HIST = "https://meteostat.p.rapidapi.com/stations/daily"
RAPID_API_KEY = '636601abdamsheec399674665a87p1878bfjsnf16635dcc483'
RAPID_API_HOST = 'meteostat.p.rapidapi.com'


async def get(station: str, date_from, date_end):
    return requests.get(
        URL_DAILY_HIST,
        params={
            'station': station,
            'start': date_from,
            'end': date_end
        },
        headers={
            'X-RapidAPI-Key': RAPID_API_KEY,
            'X-RapidAPI-Host': RAPID_API_HOST
        }
    )


async def station_data_load(station: str, date_from: datetime.date, date_end=datetime.datetime.now().date()) -> str:
    save_path = create_local_data_dir("data_weather")
    print(f"Directory {save_path} created")
    response = await get(station, date_from, date_end)
    if response.status_code != 200:
        print(response.url)
        log_error(save_path, text=response.text, error_reason=response.reason, station=station, date_from=date_from,
                  date_end=date_end,
                  url=response.url)
        raise ConnectionError()

    df_result = convert_json_to_pandas(response.json())
    await save_csv_file_from_dataframe(save_path, file_name=f'{station}.csv', df=df_result)
    return save_path


def convert_json_to_pandas(response_json) -> pd.DataFrame:
    if 'data' not in response_json:
        raise AttributeError()
    df = pd.DataFrame(data=response_json['data'])
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df.set_index('date', inplace=True)
    return df

# async def txt_file_load_and_save(file_name, save_path, url) -> None:
#     print(f'Request TXT {url} sent')
#     response = requests.get(url)
#     print(f'Response TXT status_code: {response.status_code}')
#     txt_file = str(response.content, encoding='UTF8')
#     if response.status_code != 200:
#         log_error(save_path, url, txt_file)
#         raise ConnectionError()
#     file_path = os.path.join(save_path, f'{file_name}')
#     with open(file_path, 'w', encoding='UTF8') as f:
#         f.write(txt_file)
#     print(f'File {file_name} saved')
