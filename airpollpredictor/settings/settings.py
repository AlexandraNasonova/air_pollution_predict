"""
Constants used in data preprocessing
"""

PREDICTION_VALUE_TYPES = ['AQI']

POL_USE_COLUMNS = ['Countrycode', 'AirQualityStation', 'SamplingPoint',
                   'SamplingProcess', 'UnitOfMeasurement', 'Concentration',
                   'AveragingTime', 'DatetimeEnd', 'Validity', 'Verification']
DATE_COLUMN_NAME = 'DatetimeEnd'
DATE_COLUMN_NUM_IND_NAME = 'date_idx'

CONCENTRATION_COLUMN_NAME = "Concentration"
AQI_COLUMN_NAME = "AQI"
POLLUTANT_COLUMN_NAME = 'Pollutant'

POL_NAMES = {7: "O3", 6001: "PM25", 5: "PM10", 8: "NO2", 10: "CO2", 1: "SO2"}
POL_NAMES_REVERSE = {"O3": 7, "PM25": 6001, "PM10": 5, "NO2": 8, "CO2": 10, "SO2": 1}
POL_MEASURES = {7: "µg/m3", 6001: "µg/m3", 5: "µg/m3", 10: "mg/m3", 1: "µg/m3", 8: "µg/m3"}


DATE_WEATHER_COLUMN_NAME = 'date'
DATE_COLUMNS = ['weekday', 'day', 'month', 'year', 'season', 'is_weekend', 'is_new_year']
WEATHER_COLUMNS = ['tavg', 'tmin', 'tmax', 'prcp', 'wdir', 'wspd', 'wpgt', 'pres']
POL_CODES = [7, 6001, 5, 8]
METRIC = 'rmse'
OBJECTIVE = "regression"


# NO_FILTER = 'NoFilter'
# ID_COLS = []
# LAGS_SHIFT = [7, 8, 9, 10, 11, 12, 13, 14, 21, 28]
#
# FILTERS_FOR_AQI = ['weekday', 'month']
# WINDOWS_WITH_FILTERS_FOR_AQI = {
#     'NoFilter': ['3D', '5D', '7D', '14D', '28D'],
#     'weekday': ['28D', '56D'],
#     'month': ['90D']
# }
# LAGS_AGGREGATES_FOR_AQI = [7, 10, 14, 21, 28]
# METHODS_AGGREGATES_FOR_AQI = ['mean', 'median', lag_gen.percentile(10),
#                               lag_gen.percentile(90)]  # , pd.Series.skew, pd.Series.kurtosis]
# EWM_PARAMS_FOR_AQI = {
#     'NoFilter': [7, 14, 21, 28],
#     'weekday': [28, 56],
#     'month': [90],
# }
#
# FILTERS_FOR_MEAN_CONCENTRATION = ['weekday', 'month']
# WINDOWS_WITH_FILTERS_FOR_MEAN_CONCENTRATION = {
#     'NoFilter': ['3D', '5D', '7D', '14D', '28D'],
#     'weekday': ['28D', '42D'],
#     'month': ['7D', '14D', '28D']
# }
# LAGS_AGGREGATES_FOR_MEAN_CONCENTRATION = [7, 10, 14, 21, 28]
# METHODS_AGGREGATES_FOR_MEAN_CONCENTRATION = ['mean']
# EWM_PARAMS_FOR_MEAN_CONCENTRATION = {
#     'NoFilter': [7, 14, 21, 28],
#     'weekday': [7, 14, 21, 28],
#     'month': [7, 14, 21, 28],
# }
