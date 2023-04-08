import asyncio
import datetime
from fastapi import FastAPI
from pydantic import BaseModel, validator
import loaders.aqreport_loader.loader as aqi_loader

app = FastAPI()


class AqiLoadParams(BaseModel):
    year_from_train: int | None = 2015
    pollutants_codes: list[int]
    country_code: str
    city: str | None = None
    stations_per_pollutants: dict[int, str]

    @validator('pollutants_codes')
    def pollutants_codes_must_have_value(cls, v: list[int]):
        if v is None or len(v) == 0:
            raise ValueError('pollutants_codes must have values')
        return v

    @validator('country_code')
    def country_code_must_have_value(cls, v: str):
        if v is None or len(v.strip()) == 0:
            raise ValueError('country_code must have value')
        return v

    @validator('stations_per_pollutants')
    def stations_per_pollutants_must_have_value(cls, v: dict[int, str]):
        if v is None or len(v.keys()) == 0:
            raise ValueError('stations_per_pollutants must have values')
        return v


@app.post("/download_prev_years")
def download_prev_years(save_dir_path: str, load_params: AqiLoadParams):
    urls_path = asyncio.run(
        aqi_loader.pollutants_txt_lists_load(save_dir_path=save_dir_path,
                                             year_from=load_params.year_from_train,
                                             year_to=datetime.datetime.now().year-1,
                                             pollutant_codes=load_params.pollutants_codes,
                                             country=load_params.country_code,
                                             city=load_params.city,
                                             station_per_pollutant=load_params.stations_per_pollutants))
    asyncio.run(aqi_loader.csv_list_load(save_dir_path, urls_path))


@app.post("/download_current_year")
def download_current_year(save_dir_path, load_params: AqiLoadParams):
    urls_path = asyncio.run(
        aqi_loader.pollutants_txt_lists_load(save_dir_path=save_dir_path,
                                             year_from=datetime.datetime.now().year,
                                             year_to=datetime.datetime.now().year,
                                             pollutant_codes=load_params.pollutants_codes,
                                             country=load_params.country_code,
                                             city=load_params.city,
                                             station_per_pollutant=load_params.stations_per_pollutants))

    asyncio.run(aqi_loader.csv_list_load(save_dir_path, urls_path))


"""
{
  "pollutants_codes":  [7, 6001, 5, 8],
  "country_code": "NL",
  "city": "Rotterdam",
  "stations_per_pollutants": {"7": "STA-NL00418", "5": "STA-NL00418", "6001": "STA-NL00448", "8": "STA-NL00418"}
}
"""
