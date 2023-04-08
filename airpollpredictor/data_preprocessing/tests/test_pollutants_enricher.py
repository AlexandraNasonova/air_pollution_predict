# pylint: disable=E0401, R0913, R0914, W0703, R0902

"""
Unit tests for AQI calculations
"""
import datetime
import os
import unittest
from .. import pollutants_enricher as pol_enrich


class PollutantsEnricherTestCase(unittest.TestCase):
    """
    Unit tests for Lag Features generation
    """
    def test_calc_aqi_and_mean_concentration_and_merge(self):
        """Test CO2 AQI calculations for a day. Good level"""
        date_column_name = "DatetimeEnd"
        pollutant_codes = [7, 6001]

        source_path = "data_preprocessing/tests/datasets_tests/pollutants-clean-data"
        output_file = "data_preprocessing/tests/datasets_tests/pollutants-enrich-data/aqi_enriched.csv"
        if os.path.exists(output_file):
            os.remove(output_file)
        year_from = 2023
        date_from = str(datetime.date(year=year_from, month=1, day=1))
        date_to = str(datetime.datetime.now().date())

        df_aqi_mean = pol_enrich.calc_aqi_and_mean_concentration_and_merge(
            source_data_path=source_path,
            pollutants_codes=pollutant_codes,
            date_from=date_from, date_end=date_to
        )

        # df_aqi_mean.to_csv("data_preprocessing/tests/datasets_tests/pollutants-merged-data/pol_merged.csv")

        self.assertTrue(df_aqi_mean.shape == (97, 6))
        self.assertTrue("AQI_O3" in df_aqi_mean.columns.values)
        self.assertTrue("C_MEAN_PM25" in df_aqi_mean.columns.values)
        self.assertTrue(df_aqi_mean.index.name == date_column_name)

    def test_generate_features(self):
        """Test CO2 AQI calculations for a day. Good level"""
        pollutant_codes = [7, 6001]

        source_path = "data_preprocessing/tests/datasets_tests/pollutants-clean-data"
        output_file = "data_preprocessing/tests/datasets_tests/pollutants-enrich-data/aqi_enriched.csv"
        if os.path.exists(output_file):
            os.remove(output_file)

        year_from = 2023
        lags_shift = [7, 28]
        filters_aqi = ['weekday', 'month']
        windows_aqi = {
            'NoFilter': ['3D', '14D'],
            'weekday': ['56D'],
            'month': ['90D']
        }
        lags_agg_aqi = [7, 28]
        methods_agg_aqi = ['mean', 'percentile(10)']
        ewm_filters_aqi = {
            'NoFilter': [21, 28],
            'weekday': [28],
            'month': [90],
        }
        date_from = str(datetime.date(year=year_from, month=1, day=1))
        date_to = str(datetime.datetime.now().date())

        pol_enrich.generate_features(
            source_data_path=source_path,
            output_file=output_file,
            pollutants_codes=pollutant_codes,
            date_from=date_from, date_end=date_to,
            lags_shift=lags_shift,
            filters_aqi=filters_aqi, windows_filters_aqi=windows_aqi,
            methods_agg_aqi=methods_agg_aqi, lags_agg_aqi=lags_agg_aqi,
            ewm_filters_aqi=ewm_filters_aqi
        )
        self.assertTrue(os.path.exists(output_file))
