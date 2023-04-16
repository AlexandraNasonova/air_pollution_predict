# pylint: disable=E0401
"""Module for calculating AQI by concentrations of pollutant."""


import pandas as pd
from . import aqi_dto as aqd
from . import pollutant_dto as pld
from . import breakpoint_dto as bpd

_BREAKPOINTS_COUNT = 7

_POLLUTANTS_INFO = \
    {7: pld.PollutantDto(pollutant_id=7,
                         pollutant_code="O3",
                         hourly_intervals=[8, 1],
                         decimals_count=3,
                         breakpoints_by_intervals={
                             8: [0.054, 0.07, 0.085, 0.105, 0.2, None, None],
                             1: [None, 0.125, 0.164, 0.204, 0.404, 0.504, 0.604]},
                         measure='ppm'),
     6001: pld.PollutantDto(pollutant_id=6001,
                            pollutant_code="PM2.5",
                            hourly_intervals=[24],
                            decimals_count=1,
                            breakpoints_by_intervals={
                                24: [12, 35.4, 55.4, 150.4, 250.4, 350.4, 500.4]},
                            measure='µg/m3'),
     5: pld.PollutantDto(pollutant_id=5, pollutant_code="PM10",
                         hourly_intervals=[24],
                         decimals_count=0,
                         breakpoints_by_intervals={
                             24: [54, 154, 254, 354, 424, 504, 604]},
                         measure='µg/m3'),
     10: pld.PollutantDto(pollutant_id=10,
                          pollutant_code="CO",
                          hourly_intervals=[8],
                          decimals_count=1,
                          breakpoints_by_intervals={
                              8: [4.4, 9.4, 12.4, 15.4, 30.4, 40.4, 50.4]},
                          measure='ppm'),
     1: pld.PollutantDto(pollutant_id=1,
                         pollutant_code="SO2",
                         hourly_intervals=[1, 24],
                         decimals_count=3,
                         breakpoints_by_intervals={
                             1: [None, None, 185, 304, 604, 804, 1004],
                             24: [None, None, 185, 304, 604, 804, 1004]},
                         measure='ppb'),
     8: pld.PollutantDto(pollutant_id=8,
                         pollutant_code="NO2",
                         hourly_intervals=[1],
                         decimals_count=1,
                         breakpoints_by_intervals={
                             1: [53, 100, 360, 649, 1249, 1649, 2049]},
                         measure='ppb')
     }

_AQI_INFO = [aqd.AqiDto(category="Good", aqi_from=0, aqi_to=50, color=(0, 228, 0)),
             aqd.AqiDto(category="Moderate", aqi_from=51, aqi_to=100, color=(255, 255, 0)),
             aqd.AqiDto(category="Unhealthy for Sensitive Groups",
                        aqi_from=101, aqi_to=150, color=(255, 126, 0)),
             aqd.AqiDto(category="Unhealthy", aqi_from=151, aqi_to=200, color=(255, 0, 0)),
             aqd.AqiDto(category="Very unhealthy", aqi_from=201, aqi_to=300, color=(143, 63, 151)),
             aqd.AqiDto(category="Hazardous", aqi_from=301, aqi_to=400, color=(126, 0, 35)),
             aqd.AqiDto(category="Hazardous", aqi_from=401, aqi_to=500, color=(126, 0, 35)),
             aqd.AqiDto(category="Beyond the AQI", aqi_from=501, aqi_to=1000, color=(0, 0, 0))]

_AQI_INFO_SHORT = [50, 100, 150, 200, 300, 400, 500]


def calc_aqi_for_day_pd(pollutant_id: int, df_concentrations: pd.DataFrame, measure: str):
    """
    Calculates AQI for 1 pollutant per day
    :param measure: units of measurement
    :param pollutant_id: Pollutant id in AQI data
    :param df_concentrations: DataFrame with concentration by hours.
    Must have DateIndex and Concentration column
    :return: DataFrame with AQI by days
    """

    pollutant_info = _POLLUTANTS_INFO[pollutant_id]
    df_aqi = __recalculate_concentration_pd(pollutant_info, df_concentrations, measure)
    df_aqi['AQI'] = df_aqi.apply(lambda x: __get_aqi_for_day(pollutant_info, x.values), axis=1)
    df_aqi.drop(columns=df_aqi.columns.values[:-1], inplace=True)
    return df_aqi


def calculate_aqi_for_day_target(pollutant_id: int, target: [], measure: str):
    """
    Calculates AQI for 1 pollutant per day
    :param measure: units of measurement
    :param pollutant_id: Pollutant id in AQI data
    :param target: np array from target with concentration by hours
    :return: DataFrame with AQI by days
    """

    pollutant_info = _POLLUTANTS_INFO[pollutant_id]
    targ_conv_mes = [__convert_measure(concentration=x, measure_source=measure,
                                       measure_target=pollutant_info.measure,
                                       pollutant_id=pollutant_info.pollutant_id) for x in target]

    target_res = [__get_aqi_for_day(pollutant_info, [x]) for x in targ_conv_mes]
    return target_res


def __recalculate_concentration_pd(pollutant_info: pld.PollutantDto,
                                   df_concentrations: pd.DataFrame, measure: str):
    df_result_concentrations = pd.DataFrame(
        index=df_concentrations.groupby(pd.Grouper(freq="24H")).count().index)
    for i in pollutant_info.hourly_intervals:
        if i == 24:
            df_result_concentrations = df_result_concentrations.merge(
                df_concentrations['Concentration'].groupby(
                    pd.Grouper(freq="24H")).mean(), left_index=True, right_index=True)
        elif i == 1:
            df_result_concentrations = df_result_concentrations.merge(
                df_concentrations['Concentration'].groupby(
                    pd.Grouper(freq="24H")).max(), left_index=True, right_index=True)
        else:
            df_result_concentrations = df_result_concentrations.merge(
                (df_concentrations['Concentration'].rolling(window=i, min_periods=1).mean())
                .groupby(pd.Grouper(freq="1D")).max(),
                left_index=True, right_index=True)

    for col in df_result_concentrations.columns:
        df_result_concentrations[col] = \
            df_result_concentrations[col].map(
                lambda x: __convert_measure(concentration=x, measure_source=measure,
                                            measure_target=pollutant_info.measure,
                                            pollutant_id=pollutant_info.pollutant_id))
    return df_result_concentrations


def calculate_aqi_for_day(pollutant_id: int, concentrations_per_24: list, measure: str) -> int:
    """
    Calculates AQI for pollutant
    :param measure: units of measurement
    :param pollutant_id: Pollutant id in AQI data
    :param concentrations_per_24: List of 24 values
    with pollutant concentration by hours. All values must be provided
    :return: AQI for pollutant
    """
    pollutant_info = _POLLUTANTS_INFO[pollutant_id]
    concentrations = __recalc_concentration(pollutant_info, concentrations_per_24, measure)
    return __get_aqi_for_day(pollutant_info, concentrations)


def __get_aqi_for_day(pollutant_info: pld.PollutantDto, concentrations: []) -> int:
    aqi_val = 0
    if pollutant_info.pollutant_code == "O3":
        breakpoints = __get_concentration_breakpoints_o3(
            pollutant_info, concentrations[0], concentrations[1])
        aqi_no2, aqi_so2 = 0, 0
        if breakpoints[0].index > -1:
            aqi_no2 = __calculate_aqi_by_formula(breakpoints[0])
        if breakpoints[1].index > -1:
            aqi_so2 = __calculate_aqi_by_formula(breakpoints[1])
        aqi_val = max(aqi_no2, aqi_so2)
    elif pollutant_info.pollutant_code == "SO2":
        breakpoint_single = __get_concentration_breakpoints_so2(
            pollutant_info, concentrations[0], concentrations[1])
        if breakpoint_single.index > -1:
            aqi_val = __calculate_aqi_by_formula(breakpoint_single)
    else:
        breakpoint_single = __get_concentration_breakpoints_norm(pollutant_info, concentrations[0])
        if breakpoint_single.index > -1:
            aqi_val = __calculate_aqi_by_formula(breakpoint_single)
    return aqi_val


def __convert_measure(concentration: float, measure_source: str,
                      measure_target: str, pollutant_id: int) -> float:
    if measure_source == measure_target:
        return concentration

    concentration_converted = concentration
    if measure_source == 'µg/m3':
        concentration_converted = concentration / 1000
    if measure_target in ('ppm', 'ppb'):
        if pollutant_id == 7:
            concentration_converted = concentration_converted * 24.45 / 48
        elif pollutant_id == 10:
            concentration_converted = concentration_converted * 24.45 / 28.01
        elif pollutant_id == 1:
            concentration_converted = concentration_converted * 24.45 / 64.06
        elif pollutant_id == 8:
            concentration_converted = concentration_converted * 24.45 / 46.01

    if measure_target == 'ppm':
        return concentration_converted
    if measure_target == 'ppb':
        return concentration_converted * 1000
    if measure_target == 'μg/m3':
        return concentration_converted

    return concentration


def get_aqi_info_by_value(aqi_val: float) -> aqd.AqiDto:
    """Returns additional information for the AQI-value
    :param aqi_val: AQI value
    :return: AQI for pollutant
    """
    aqi_info = None
    for i in range(_BREAKPOINTS_COUNT):
        if _AQI_INFO[i].aqi_from <= aqi_val <= _AQI_INFO[i].aqi_to:
            aqi_info = _AQI_INFO[i]
            break
    if aqi_info is None:
        aqi_info = _AQI_INFO[_BREAKPOINTS_COUNT]
    aqi_info_with_value = aqd.AqiDto(aqi_from=aqi_info.aqi_from,
                                     aqi_to=aqi_info.aqi_to, category=aqi_info.category,
                                     color=aqi_info.color)
    aqi_info_with_value.value = aqi_val
    return aqi_info_with_value


def calc_aqi_for_day_target(pollutant_id: int, target: [], measure: str):
    """
    Calculates AQI for 1 pollutant per day
    :param measure: units of measurement
    :param pollutant_id: Pollutant id in AQI data
    :param target: np array from target with concentration by hours
    :return: DataFrame with AQI by days
    """

    pollutant_info = _POLLUTANTS_INFO[pollutant_id]
    targ_conv_mes = [__convert_measure(concentration=x, measure_source=measure,
                                       measure_target=pollutant_info.measure,
                                       pollutant_id=pollutant_info.pollutant_id) for x in target]

    target_res = [__get_aqi_for_day(pollutant_info, [x]) for x in targ_conv_mes]
    return target_res


def __recalc_concentration(pollutant_info: pld.PollutantDto,
                           concentrations_per_24: list[float],
                           measure: str) -> [float]:
    concentrations = []
    for i in pollutant_info.hourly_intervals:
        concentration_day_mean = []
        j_count = int(24 / i)
        for j in range(j_count):
            concentration = sum(concentrations_per_24[j * i:j * i + i]) / i
            concentration = round(concentration, pollutant_info.decimals_count)
            concentration_day_mean.append(concentration)
        concentrations.append(__convert_measure(concentration=max(concentration_day_mean),
                                                measure_source=measure,
                                                measure_target=pollutant_info.measure,
                                                pollutant_id=pollutant_info.pollutant_id))
    return concentrations


def __get_concentration_breakpoints_norm(pollutant_info: pld.PollutantDto, concentration: float) \
        -> bpd.BreakpointDto:
    interval = pollutant_info.hourly_intervals[0]
    breakpoints_main = pollutant_info.breakpoints_by_intervals[interval]
    breakpoint_low, breakpoint_high = 0, 0
    breakpoint_ind = -1
    for i in range(_BREAKPOINTS_COUNT):
        breakpoint_high = breakpoints_main[i]
        if concentration < breakpoint_high:
            breakpoint_ind = i
            break
        breakpoint_low = breakpoint_high + pollutant_info.step

    return bpd.BreakpointDto(breakpoint_ind, breakpoint_low, breakpoint_high, concentration)


def __get_concentration_breakpoints_o3(pollutant_info: pld.PollutantDto, concentration_no2: float,
                                       concentration_so2: float) -> list[bpd.BreakpointDto]:
    interval_1 = pollutant_info.hourly_intervals[0]
    interval_2 = pollutant_info.hourly_intervals[1]
    breakpoint_no2 = pollutant_info.breakpoints_by_intervals[interval_1]
    breakpoint_so2 = pollutant_info.breakpoints_by_intervals[interval_2]

    breakpoint_low_no2, breakpoint_high_no2, breakpoint_high_so2 = 0, 0, 0
    breakpoint_low_so2 = None
    breakpoint_ind_no2, breakpoint_ind_so2 = -1, -1
    for i in range(_BREAKPOINTS_COUNT):
        breakpoint_high_no2 = breakpoint_no2[i]
        if breakpoint_high_no2 is None:
            break
        if concentration_no2 < breakpoint_high_no2:
            breakpoint_ind_no2 = i
            break
        breakpoint_low_no2 = breakpoint_high_no2 + pollutant_info.step

    for i in range(_BREAKPOINTS_COUNT):
        breakpoint_high_so2 = breakpoint_so2[i]
        if breakpoint_high_so2 is None:
            continue
        if concentration_so2 < breakpoint_high_so2:
            if breakpoint_low_so2 is None:
                break
            breakpoint_ind_so2 = i
            break
        breakpoint_low_so2 = breakpoint_high_so2 + pollutant_info.step

    return [bpd.BreakpointDto(breakpoint_ind_no2, breakpoint_low_no2,
                              breakpoint_high_no2, concentration_no2),
            bpd.BreakpointDto(breakpoint_ind_so2, breakpoint_low_so2,
                              breakpoint_high_so2, concentration_so2)]


def __get_concentration_breakpoints_so2(pollutant_info: pld.PollutantDto,
                                        concentration_per_hour: float,
                                        concentration_per_24_hours: float) \
        -> bpd.BreakpointDto:
    interval_1 = pollutant_info.hourly_intervals[0]
    interval_2 = pollutant_info.hourly_intervals[1]
    breakpoints_1 = pollutant_info.breakpoints_by_intervals[interval_1]
    breakpoints_24 = pollutant_info.breakpoints_by_intervals[interval_2]

    breakpoint_low, breakpoint_high = 0, 0
    breakpoint_ind = -1
    concentration = concentration_per_hour
    to24 = False
    for i in range(_BREAKPOINTS_COUNT):
        breakpoint_high = breakpoints_1[i] if not to24 else breakpoints_24[i]
        if breakpoint_high is None:
            concentration = concentration_per_24_hours
            breakpoint_high = breakpoints_24[i]
            to24 = True

        if concentration < breakpoint_high:
            breakpoint_ind = i
            break
        breakpoint_low = breakpoint_high + pollutant_info.step

    return bpd.BreakpointDto(breakpoint_ind, breakpoint_low, breakpoint_high, concentration)


def __get_aqi_breakpoints(bp_ind):
    aqi_lo, aqi_hi = 0, 0
    if bp_ind > -1:
        aqi_hi = _AQI_INFO_SHORT[bp_ind]
        aqi_lo = _AQI_INFO_SHORT[bp_ind - 1] if bp_ind > 0 else 0
    return aqi_lo, aqi_hi


def __calculate_aqi_by_formula(breakpoint_info: bpd.BreakpointDto) -> int:
    aqi_lo, aqi_hi = __get_aqi_breakpoints(breakpoint_info.index)
    aqi = int(round((aqi_hi - aqi_lo)
                    * (breakpoint_info.concentration - breakpoint_info.low_border)
                    / (breakpoint_info.high_border - breakpoint_info.low_border)
                    + aqi_lo, 0))
    return aqi
