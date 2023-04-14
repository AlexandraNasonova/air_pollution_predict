# pylint: disable=E0401, E0611

import asyncio
import datetime
import os
from fastapi import FastAPI
from pydantic import BaseModel, validator
from . import aqi_report_loader

app = FastAPI()


class AqiLoadParams(BaseModel):
    save_dir_path: str
    date_from: str
    date_to: str
    country_code: str
    city: str | None = None
    stations_per_pollutants: dict[int, str]

    # pylint: disable=E0213, R0201
    @validator('country_code')
    def country_code_must_have_value(cls, value: str):
        if value is None or len(value.strip()) == 0:
            raise ValueError('country_code must have value')
        return value

    # pylint: disable=E0213, R0201
    @validator('save_dir_path')
    def save_dir_path_must_have_value(cls, value: str):
        if value is None or len(value.strip()) == 0:
            raise ValueError('save_dir_path must have value')
        return value

    # pylint: disable=E0213, R0201
    @validator('stations_per_pollutants')
    def stations_per_pollutants_must_have_value(cls, value: dict[int, str]):
        if value is None or len(value.keys()) == 0:
            raise ValueError('stations_per_pollutants must have values')
        return value


@app.post("/download")
def download(load_params: AqiLoadParams):
    date_from = datetime.datetime.strptime(load_params.date_from, "%Y-%m-%d").date()
    date_to = datetime.datetime.strptime(load_params.date_to, "%Y-%m-%d").date()

    urls_path = asyncio.run(
        aqi_report_loader.pollutants_txt_lists_load(
            save_dir_path=load_params.save_dir_path,
            year_from=date_from.year,
            year_to=date_to.year,
            pollutant_codes=list(load_params.stations_per_pollutants.keys()),
            country=load_params.country_code,
            city=load_params.city,
            station_per_pollutant=load_params.stations_per_pollutants))
    asyncio.run(aqi_report_loader.csv_list_load(load_params.save_dir_path, urls_path))


@app.post("/test")
def test(save_dir_path: str, text: str):
    file_path = os.path.join(save_dir_path, 'test.txt')
    with open(file_path, 'w', encoding='UTF8') as file_stream:
        file_stream.write(text)


"""
{
  "pollutants_codes":  [7, 6001, 5, 8],
  "country_code": "NL",
  "city": "Rotterdam",
  "stations_per_pollutants": {"7": "STA-NL00418", "5": "STA-NL00418", "6001": "STA-NL00448", "8": "STA-NL00418"}
}
"""
