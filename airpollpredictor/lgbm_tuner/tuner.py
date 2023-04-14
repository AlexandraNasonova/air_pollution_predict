import pandas as pd
import datetime
from ml_tune_helpers import ts_splitter
from ml_tune_helpers.lgbm_optuna.optuna_lgb_search import OptunaLgbSearch
from . import columns_filter
# from ..settings import settings
import settings.settings as settings


def init_optuna(df_timeseries: pd.DataFrame, pol_id: int,
                prediction_value_type: str,
                train_start_dt: datetime.date, train_end_dt: datetime.date,
                test_start_dt: datetime.date, test_end_dt: datetime.date,
                use_aqi_cols: bool, use_c_mean_cols: bool, use_lag_cols: bool,
                use_gen_lags_cols: bool, use_weather_cols: bool,
                use_c_median_cols=False, use_c_max_cols=False,
                use_c_min_cols=False, use_pol_cols=False,
                default_params=None, default_category=None,
                categories_for_optimization=None,
                default_top_features_count=-1) -> \
        (OptunaLgbSearch, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Splits timeseries for experiments, initializes Optuna LightGbm Wrapper
    @param df_timeseries: The timeseries
    @param pol_id: The standard identificator of the pollutant
    @param prediction_value_type: The type of the prediction value
    (only Aqi and mean concentration were tested)
    @param train_start_dt: The first date of the train dataset
    @param train_end_dt: The last date of the train dataset
    @param test_start_dt: The first date of the test dataset
    @param test_end_dt: The last date of the train dataset
    @param use_aqi_cols: The flag if the AQI columns should be included to
    the features datasets_tests
    @param use_c_mean_cols: The flag if the Mean Concentration columns should be
    included to the features datasets_tests
    @param use_lag_cols: The flag if the Lag columns should be included to
    the features datasets_tests
    @param use_gen_lags_cols: The flag if the Aggregated Lag columns should
    be included to the features datasets_tests
    @param use_weather_cols: The flag if the Weather columns should
    be included to the features datasets_tests
    @param use_c_median_cols: The flag if the Median Concentration columns should be included to
    the features datasets_tests
    @param use_c_max_cols: The flag if the Max Concentration columns should be included to
    the features datasets_tests
    @param use_c_min_cols: The flag if the Min Concentration columns should be included to
    the features datasets_tests
    @param use_pol_cols: The flag if the Pollutant columns should be included to
    the features datasets_tests (!not tested yet)
    @param default_params: The default model params (optional)
    @param default_category: The default set of categories (optional)
    @param categories_for_optimization: The list of sets of categories for
    search the best one (optional)
    @param default_top_features_count: The default quantity of the most important
    features to be used
    @return: The Optuna LightGBM wrapper instance and train/test X/y filtered dataframes
    """
    target_column_name = columns_filter.get_target_column(prediction_value_type, pol_id=pol_id)

    df_use = columns_filter.filter_data_frame(
        df_timeseries=df_timeseries,
        pol_id=pol_id,
        target_column_name=target_column_name,
        use_aqi_cols=use_aqi_cols,
        use_c_mean_cols=use_c_mean_cols,
        use_c_median_cols=use_c_median_cols,
        use_c_max_cols=use_c_max_cols,
        use_c_min_cols=use_c_min_cols,
        use_lag_cols=use_lag_cols,
        use_gen_lags_cols=use_gen_lags_cols,
        use_pol_cols=use_pol_cols,
        use_weather_cols=use_weather_cols,
        date_columns=settings.DATE_COLUMNS,
        weather_columns=settings.WEATHER_COLUMNS,
        pol_codes=settings.POL_CODES)

    x_train_filt, y_train_filt = \
        ts_splitter.split_x_y_for_period(df_timeseries=df_use, index_cols='DatetimeEnd',
                                         y_value_col=target_column_name,
                                         dt_start=train_start_dt, dt_end=train_end_dt)
    x_val_filt, y_val_filt = \
        ts_splitter.split_x_y_for_period(df_timeseries=df_use, index_cols='DatetimeEnd',
                                         y_value_col=target_column_name, dt_start=test_start_dt,
                                         dt_end=test_end_dt)

    optuna_helper = OptunaLgbSearch(study_name=f'lgbm_{pol_id if pol_id > 0 else "all"}',
                                    metric=settings.METRIC,
                                    objective=settings.OBJECTIVE,
                                    x_train=x_train_filt, y_train=y_train_filt,
                                    x_val=x_val_filt, y_val=y_val_filt,
                                    default_params=default_params,
                                    default_category=default_category,
                                    categories_for_optimization=categories_for_optimization,
                                    default_top_features_count=default_top_features_count)
    return optuna_helper, x_train_filt, y_train_filt, x_val_filt, y_val_filt

# def convert_conc_to_aqi(pol_id: int, target_values):
#     measure = POL_MEASURES[pol_id]
#     df_aqi = aqc.calc_aqi_for_day_pd(pol_id, target_values, measure)
#     return df_aqi

# def filter_data(df_timeseries: pd.DataFrame,
#                 pol_id: int,
#                 target_column_name: str,
#                 use_aqi_cols: bool,
#                 use_c_mean_cols: bool,
#                 use_c_median_cols: bool,
#                 use_c_max_cols: bool,
#                 use_c_min_cols: bool,
#                 use_lag_cols: bool,
#                 use_gen_lags_cols: bool,
#                 use_pol_cols: bool,
#                 use_weather_cols: bool) -> pd.DataFrame:
#     req_cols = __get_required_columns(df_timeseries=df_timeseries,
#                                       pol_id=pol_id,
#                                       target_column_name=target_column_name,
#                                       use_aqi_cols=use_aqi_cols,
#                                       use_c_mean_cols=use_c_mean_cols,
#                                       use_c_median_cols=use_c_median_cols,
#                                       use_c_max_cols=use_c_max_cols,
#                                       use_c_min_cols=use_c_min_cols,
#                                       use_lag_cols=use_lag_cols,
#                                       use_gen_lags_cols=use_gen_lags_cols,
#                                       use_pol_cols=use_pol_cols,
#                                       use_weather_cols=use_weather_cols)
#     return df_timeseries[req_cols]
