import asyncio
import loaders.weather_loader.loader as weather_loader
from datetime import datetime

if __name__ == '__main__':
    # save_path = asyncio.run(aq_loader.pollutants_txt_lists_load())

    # save_path = asyncio.run(
    #     aq_loader.pollutants_txt_lists_load(
    #         pollutant_codes = [7, 6001, 5, 8], country="NL", city='Rotterdam', year_from=2015,
    #         station_per_pollutant={7: 'STA-NL00418', 5: 'STA-NL00418', 6001: 'STA-NL00448', 8: 'STA-NL00418'}))
    # save_path =
    # "/home/alexna/work/projects/air_pollution_predict/airpollpredictor/aqreport_loader/data/02_11_2022_12_12_44"

    # asyncio.run(aq_loader.csv_list_load(save_path))
    date_from = datetime.strptime("2015-01-01", "%Y-%m-%d").date()
    save_path = asyncio.run(weather_loader.load_weather_history_from_station(station="06344", date_from=date_from))
