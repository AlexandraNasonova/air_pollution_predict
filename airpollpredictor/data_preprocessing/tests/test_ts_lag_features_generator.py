# pylint: disable=E0401, R0913, R0914, W0703, R0902

"""
Unit tests for AQI calculations
"""
import datetime
import os
import unittest

import pandas as pd

from ..features_generations import ts_lag_features_generator as lag_gen


class TsLagFeaturesGeneratorTestCase(unittest.TestCase):
    """
    Unit tests for Lag Features generation
    """
    def test_calc_preag_fill(self):
        date_col = 'DatetimeEnd'
        filter_col = 'NoFilter'
        target_cols = ['AQI', 'AQI_O3', 'AQI_PM25']
        source_file = "data_preprocessing/tests/datasets_tests/pollutants-merged-data/pol_merged.csv"
        data = pd.read_csv(source_file)
        data[filter_col] = 1
        data = data.sort_values(date_col)
        group_col = [filter_col] + [] + [date_col]
        preagg = 'mean'
        data_preag_filled = lag_gen.calc_preag_fill(data, group_col, date_col, target_cols, preagg)
        self.assertTrue(data_preag_filled.shape == (97, 5))