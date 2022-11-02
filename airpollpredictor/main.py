import asyncio
import aqreport_loader.loader as aq_loader


if __name__ == '__main__':
    # save_path = asyncio.run(aq_loader.pollutants_txt_lists_load())
    save_path = asyncio.run(aq_loader.pollutants_txt_lists_load(country="DE", year_from=2022))
    # save_path = "/home/alexna/work/projects/air_pollution_predict/airpollpredictor/aqreport_loader/data/02_11_2022_12_12_44"
    asyncio.run(aq_loader.csv_list_load(save_path))
