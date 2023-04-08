# pylint: disable=E0401, R0913, R0914, W0703, R0902

"""
Unit tests for AQI calculations
"""
import unittest
import pandas as pd
import dvc_pipeline.data_preprocessing.aqi_calculations.aqi_calculator as aqc


class AqiCalculatorTestCase(unittest.TestCase):
    """
    Unit tests for AQI calculations
    """
    def test_co2_good(self):
        """Test CO2 AQI calculations for a day. Good level"""
        pollutant_id = 10
        concentrations_per_24 = [0.40861201, 0.43800252, 0.49664002, 0.49141851, 0.5072295,
                                 0.48276847, 0.45546349, 0.47943699, 0.517641, 0.55236554,
                                 0.61161899, 0.608967, 0.59491753, 0.57926702, 0.56780552,
                                 0.56874346, 0.64490449, 0.578799, 0.56269752, 0.55784749,
                                 0.55327248, 0.51070499, 0.52470409, 0.539514]
        aqi = aqc.calculate_aqi_for_day(pollutant_id, concentrations_per_24, 'mg/m3')
        aqi_info = aqc.get_aqi_info_by_value(aqi)
        self.assertEqual(aqi_info.value, 6)

    def test_co2_moderate(self):
        """Test CO2 AQI calculations for a day. Moderate level"""
        pollutant_id = 10
        concentrations_per_24 = [5.40861201, 5.43800252, 5.49664002, 5.49141851,
                                 5.5072295, 5.48276847, 5.45546349, 5.47943699,
                                 5.517641, 4.55236554, 5.61161899, 3.608967, 4.59491753,
                                 5.57926702, 4.76780552, 5.76874346, 5.64490449, 5.578799,
                                 53.56269752, 4.55784749, 5.55327248, 3.71070499,
                                 4.72470409, 3.739514]
        aqi = aqc.calculate_aqi_for_day(pollutant_id, concentrations_per_24, 'mg/m3')
        aqi_info = aqc.get_aqi_info_by_value(aqi)
        self.assertEqual(aqi_info.value, 100)

    def test_o3_good(self):
        """Test O3 AQI calculations for a day. Good level"""
        pollutant_id = 7
        concentrations_per_24 = [20.46, 15.67, 14.9, 14.76, 18.24, 20.5, 15.49,
                                 17.57, 16.14, 20.82, 25.48, 28.57, 32.22, 34.8,
                                 38.36, 34.54, 23.65, 23.77, 24.53, 20.58, 20.98,
                                 22.32, 21.54, 20.74]
        aqi = aqc.calculate_aqi_for_day(pollutant_id, concentrations_per_24, 'µg/m3')
        aqi_info = aqc.get_aqi_info_by_value(aqi)
        self.assertEqual(aqi_info.value, 14)

    def test_pm10_good(self):
        """Test PM10 AQI calculations for a day. Good level"""
        pollutant_id = 5
        concentrations_per_24 = [30.8, 11.7, 10.1, 10.1, 12.2, 9.8, 8, 10.8, 5.9,
                                 7.5, 13.6, 14.5, 9, 9.2, 13, 10.5, 8.9, 6, 6.3,
                                 6.9, 8.6, 8.6, 5, 8.2]
        aqi = aqc.calculate_aqi_for_day(pollutant_id, concentrations_per_24, 'µg/m3')
        aqi_info = aqc.get_aqi_info_by_value(aqi)
        self.assertEqual(aqi_info.value, 9)

    def test_pm10_unhealthy(self):
        """Test PM10 AQI calculations for a day. Unhealthy level"""
        pollutant_id = 5
        concentrations_per_24 = [160.8, 171.7, 180.1, 170.1, 192.2, 191.8, 208, 180.8, 159.9,
                                 177.5, 183.6, 194.5, 209, 209.2, 163, 170.5, 182.9, 167, 165.3,
                                 168.9, 182.6, 187.6, 205, 180.2]
        aqi = aqc.calculate_aqi_for_day(pollutant_id, concentrations_per_24, 'µg/m3')
        aqi_info = aqc.get_aqi_info_by_value(aqi)
        self.assertEqual(aqi_info.value, 114)

    def test_pm10_pandas(self):
        """Test PM10 AQI calculations for period"""
        pollutant_id = 5
        measure = 'µg/m3'
        df_source = pd.read_csv("PL_5_29754_2022_timeseries.csv", parse_dates=True,
                                index_col='DatetimeEnd')
        df_result = aqc.calc_aqi_for_day_pd(pollutant_id, df_source, measure)
        self.assertEqual(df_result.iloc[0, -1], 10)

    def test_03_pandas(self):
        """Test O3 AQI calculations for period"""
        pollutant_id = 7
        measure = 'µg/m3'
        df_source = pd.read_csv("NL_7_28294_2021_timeseries.csv", parse_dates=True,
                                index_col='DatetimeEnd')
        df_result = aqc.calc_aqi_for_day_pd(pollutant_id, df_source, measure)
        self.assertEqual(df_result.iloc[0, -1], 14)

    def test_pm25_target(self):
        """Test PM2.5 AQI calculations for a day. Moderate level"""
        pollutant_id = 6001
        concentrations_daily = [23.835134, 33.95706016, 33.15522659, 21.98010993, 20.41169521,
                                16.04986439, 18.85318373]
        aqi_res = aqc.calculate_aqi_for_day_target(pollutant_id, concentrations_daily, 'µg/m3')

        self.assertEqual(aqi_res[0], 75)
