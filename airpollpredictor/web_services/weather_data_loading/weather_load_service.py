import asyncio
import datetime
from fastapi import FastAPI
from pydantic import BaseModel, validator
import loaders.weather_loader.loader as weather_loader

app = FastAPI()


class WeatherLoadParams(BaseModel):
    station_id: str
    date_from: str | None = None

    @validator('station_id')
    def station_id_must_have_value(cls, v: str):
        if v is None or len(v.strip()) == 0:
            raise ValueError('station_id must have value')
        return v


@app.post("/download_prev_years")
def download_prev_years(save_dir_path: str, load_params: WeatherLoadParams):
    date_from = datetime.datetime.strptime(load_params.date_from, "%Y-%m-%d").date()
    date_to = datetime.date(year=datetime.datetime.now().year - 1, month=12, day=31)

    asyncio.run(weather_loader.load_weather_history_from_station(
        save_dir_path=save_dir_path, station=load_params.station_id,
        date_from=date_from, date_end=date_to))


@app.post("/download_current_year")
def download_current_year(save_dir_path, load_params: WeatherLoadParams):
    date_from = datetime.date(year=datetime.datetime.now().year, month=1, day=1)
    date_to = datetime.datetime.now().date()
    asyncio.run(weather_loader.load_weather_history_from_station(
        save_dir_path=save_dir_path, station=load_params.station_id,
        date_from=date_from, date_end=date_to))


"""
{
  "station_id": "06344"
}
"""
