import asyncio
import aqreport_loader.loader as aq_loader


if __name__ == '__main__':
    # save_path = asyncio.run(aq_loader.pollutants_txt_lists_load())
    save_path = asyncio.run(
        aq_loader.pollutants_txt_lists_load(
            pollutant_codes = [7, 6001, 5, 8], country="NL", city='Rotterdam', year_from=2015,
            station_per_pollutant={7: 'STA-NL00418', 5: 'STA-NL00418', 6001: 'STA-NL00448', 8: 'STA-NL00418'}))
    # save_path = "/home/alexna/work/projects/air_pollution_predict/airpollpredictor/aqreport_loader/data/02_11_2022_12_12_44"
    asyncio.run(aq_loader.csv_list_load(save_path))
