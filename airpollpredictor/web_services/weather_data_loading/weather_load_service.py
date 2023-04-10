# pylint: disable=E0401, E0611

import asyncio
import datetime
from fastapi import FastAPI
from pydantic import BaseModel, validator
from . import weather_loader

app = FastAPI()


class WeatherLoadParams(BaseModel):
    station_id: str
    save_file_path: str
    date_from: str | None = None

    # pylint: disable=E0213, R0201
    @validator('station_id')
    def station_id_must_have_value(cls, value: str):
        if value is None or len(value.strip()) == 0:
            raise ValueError('station_id must have value')
        return value

    # pylint: disable=E0213, R0201
    @validator('save_file_path')
    def save_file_path_must_have_value(cls, value: str):
        if value is None or len(value.strip()) == 0:
            raise ValueError('save_file_path must have value')
        return value


@app.post("/download_prev_years")
def download_prev_years(load_params: WeatherLoadParams):
    date_from = datetime.datetime.strptime(load_params.date_from, "%Y-%m-%d").date()
    date_to = datetime.date(year=datetime.datetime.now().year - 1, month=12, day=31)

    asyncio.run(weather_loader.load_weather_history_from_station(
        save_file_path=load_params.save_file_path, station=load_params.station_id,
        date_from=date_from, date_end=date_to))


@app.post("/download_current_year")
def download_current_year(load_params: WeatherLoadParams):
    date_from = datetime.date(year=datetime.datetime.now().year, month=1, day=1)
    date_to = datetime.datetime.now().date()
    asyncio.run(weather_loader.load_weather_history_from_station(
        save_file_path=load_params.save_file_path, station=load_params.station_id,
        date_from=date_from, date_end=date_to))


"""
{
  "station_id": "06344"
}
"""
