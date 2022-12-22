import pandas as pd
import aqi_calculator.aqi_dto as aqd
import aqi_calculator.pollutant_dto as pld
import aqi_calculator.breakpoint_dto as bpd

_BREAKPOINTS_COUNT = 7

_POLLUTANTS_INFO = {7: pld.PollutantDto(pollutant_id=7, pollutant_code="O3", hourly_intervals=[8, 1],
                                        decimals_count=3,
                                        breakpoints_by_intervals={8: [0.054, 0.07, 0.085, 0.105, 0.2, None, None],
                                                                  1: [None, 0.125, 0.164, 0.204, 0.404, 0.504, 0.604]},
                                        measure='ppm'),
                    6001: pld.PollutantDto(pollutant_id=6001, pollutant_code="PM2.5", hourly_intervals=[24],
                                           decimals_count=1,
                                           breakpoints_by_intervals={24: [12, 35.4, 55.4, 150.4, 250.4, 350.4, 500.4]},
                                           measure='µg/m3'),
                    5: pld.PollutantDto(pollutant_id=5, pollutant_code="PM10", hourly_intervals=[24],
                                        decimals_count=0,
                                        breakpoints_by_intervals={24: [54, 154, 254, 354, 424, 504, 604]},
                                        measure='µg/m3'),
                    10: pld.PollutantDto(pollutant_id=10, pollutant_code="CO", hourly_intervals=[8],
                                         decimals_count=1,
                                         breakpoints_by_intervals={8: [4.4, 9.4, 12.4, 15.4, 30.4, 40.4, 50.4]},
                                         measure='ppm'),
                    1: pld.PollutantDto(pollutant_id=1, pollutant_code="SO2", hourly_intervals=[1, 24],
                                        decimals_count=3,
                                        breakpoints_by_intervals={1: [None, None, 185, 304, 604, 804, 1004],
                                                                  24: [None, None, 185, 304, 604, 804, 1004]},
                                        measure='ppb'),
                    8: pld.PollutantDto(pollutant_id=8, pollutant_code="NO2", hourly_intervals=[1],
                                        decimals_count=1,
                                        breakpoints_by_intervals={1: [53, 100, 360, 649, 1249, 1649, 2049]},
                                        measure='ppb')
                    }

_AQI_INFO = [aqd.AqiDto(category="Good", aqi_from=0, aqi_to=50, color=(0, 228, 0)),
             aqd.AqiDto(category="Moderate", aqi_from=51, aqi_to=100, color=(255, 255, 0)),
             aqd.AqiDto(category="Unhealthy for Sensitive Groups", aqi_from=101, aqi_to=150, color=(255, 126, 0)),
             aqd.AqiDto(category="Unhealthy", aqi_from=151, aqi_to=200, color=(255, 0, 0)),
             aqd.AqiDto(category="Very unhealthy", aqi_from=201, aqi_to=300, color=(143, 63, 151)),
             aqd.AqiDto(category="Hazardous", aqi_from=301, aqi_to=400, color=(126, 0, 35)),
             aqd.AqiDto(category="Hazardous", aqi_from=401, aqi_to=500, color=(126, 0, 35)),
             aqd.AqiDto(category="Beyond the AQI", aqi_from=501, aqi_to=1000, color=(0, 0, 0))]

_AQI_INFO_SHORT = [50, 100, 150, 200, 300, 400, 500]


def calc_aqi_for_day_pd(pollutant_id: int, df: pd.DataFrame, measure: str):
    """
    Calculates AQI for 1 pollutant per day
    :param measure: units of measurement
    :param pollutant_id: Pollutant id in AQI data
    :param df: DataFrame with concentration by hours. Must have DateIndex and Concentration column
    :return: DataFrame with AQI by days
    """

    pollutant_info = _POLLUTANTS_INFO[pollutant_id]
    g = __recalc_concentration_pd(pollutant_info, df, measure)
    g['AQI'] = g.apply(lambda x: __get_aqi_for_day(pollutant_info, x.values), axis=1)
    g.drop(columns=g.columns.values[:-1], inplace=True)
    return g


def __recalc_concentration_pd(pollutant_info: pld.PollutantDto, df: pd.DataFrame, measure: str):
    g = pd.DataFrame(index=df.groupby(pd.Grouper(freq="24H")).count().index)
    for i in pollutant_info.hourly_intervals:
        if i == 24:
            g = g.merge(df['Concentration'].groupby(pd.Grouper(freq="24H")).mean(), left_index=True, right_index=True)
        elif i == 1:
            g = g.merge(df['Concentration'].groupby(pd.Grouper(freq="24H")).max(), left_index=True, right_index=True)
        else:
            g = g.merge(
                (df['Concentration'].rolling(window=i, min_periods=1).mean()).groupby(pd.Grouper(freq="1D")).max(),
                left_index=True, right_index=True)

    for col in g.columns:
        g[col] = g[col].map(lambda x: __convert_measure(c=x, measure_s=measure,
                                                        measure_t=pollutant_info.measure,
                                                        pollutant_id=pollutant_info.pollutant_id))
    return g


def calc_aqi_for_day(pollutant_id: int, concentrations_per_24: list, measure: str) -> int:
    """
    Calculates AQI for pollutant
    :param measure: units of measurement
    :param pollutant_id: Pollutant id in AQI data
    :param concentrations_per_24: List of 24 values with pollutant concentration by hours. All values must be provided
    :return:
    """
    pollutant_info = _POLLUTANTS_INFO[pollutant_id]
    concentrations = __recalc_concentration(pollutant_info, concentrations_per_24, measure)
    return __get_aqi_for_day(pollutant_info, concentrations)


def __get_aqi_for_day(pollutant_info: pld.PollutantDto, c: []) -> int:
    aqi_val = 0
    if pollutant_info.pollutant_code == "O3":
        bps = __get_concentration_breakpoints_o3(pollutant_info, c[0], c[1])
        aqi_8, aqi_1 = 0, 0
        if bps[0].bp_ind > -1:
            aqi_8 = __aqi_calc_formula(bps[0])
        if bps[1].bp_ind > -1:
            aqi_1 = __aqi_calc_formula(bps[1])
        aqi_val = max(aqi_8, aqi_1)
    elif pollutant_info.pollutant_code == "SO2":
        bp = __get_concentration_breakpoints_so2(pollutant_info, c[0], c[1])
        if bp.bp_ind > -1:
            aqi_val = __aqi_calc_formula(bp)
    else:
        bp = __get_concentration_breakpoints_norm(pollutant_info, c[0])
        if bp.bp_ind > -1:
            aqi_val = __aqi_calc_formula(bp)
    return aqi_val


def __convert_measure(c: float, measure_s: str, measure_t: str, pollutant_id: int) -> float:
    if measure_s == measure_t:
        return c

    cn = c
    if measure_s == 'µg/m3':
        cn = c / 1000
    if measure_t == 'ppm' or measure_t == 'ppb':
        if pollutant_id == 7:
            cn = cn * 24.45 / 48
        elif pollutant_id == 10:
            cn = cn * 24.45 / 28.01
        elif pollutant_id == 1:
            cn = cn * 24.45 / 64.06
        elif pollutant_id == 8:
            cn = cn * 24.45 / 46.01

    if measure_t == 'ppm':
        return cn
    if measure_t == 'ppb':
        return cn * 1000
    if measure_t == 'μg/m3':
        return cn

    return c


def get_aqi_info_by_value(aqi_val: float) -> aqd.AqiDto:
    aqi_info = None
    for i in range(_BREAKPOINTS_COUNT):
        if _AQI_INFO[i].aqi_from <= aqi_val <= _AQI_INFO[i].aqi_to:
            aqi_info = _AQI_INFO[i]
            break
    if aqi_info is None:
        aqi_info = _AQI_INFO[_BREAKPOINTS_COUNT]
    aqi_info_with_value = aqd.AqiDto(aqi_from=aqi_info.aqi_from, aqi_to=aqi_info.aqi_to, category=aqi_info.category,
                                     color=aqi_info.color, value=aqi_val)
    return aqi_info_with_value


def __recalc_concentration(pollutant_info: pld.PollutantDto, concentrations_per_24: list[float], measure: str) \
        -> [int]:
    concentrations = []
    for i in pollutant_info.hourly_intervals:
        c_day_mean = []
        j_count = int(24 / i)
        for j in range(j_count):
            c = sum(concentrations_per_24[j * i:j * i + i]) / i
            c = round(c, pollutant_info.decimals_count)
            c_day_mean.append(c)
        concentrations.append(__convert_measure(c=max(c_day_mean), measure_s=measure, measure_t=pollutant_info.measure,
                                                pollutant_id=pollutant_info.pollutant_id))
    return concentrations


def __get_concentration_breakpoints_norm(pollutant_info: pld.PollutantDto, c: float) \
        -> bpd.BreakpointDto:
    interval = pollutant_info.hourly_intervals[0]
    breakpoints_main = pollutant_info.breakpoints_by_intervals[interval]
    bp_lo, bp_hi = 0, 0
    bp_ind = -1
    for i in range(_BREAKPOINTS_COUNT):
        bp_hi = breakpoints_main[i]
        if c < bp_hi:
            bp_ind = i
            break
        bp_lo = bp_hi + pollutant_info.step

    return bpd.BreakpointDto(bp_ind, bp_lo, bp_hi, c)


def __get_concentration_breakpoints_o3(pollutant_info: pld.PollutantDto, c8: float, c1: float) \
        -> list[bpd.BreakpointDto]:
    interval_1 = pollutant_info.hourly_intervals[0]
    interval_2 = pollutant_info.hourly_intervals[1]
    breakpoint_8 = pollutant_info.breakpoints_by_intervals[interval_1]
    breakpoint_1 = pollutant_info.breakpoints_by_intervals[interval_2]

    bp_lo_8, bp_hi_8, bp_hi_1 = 0, 0, 0
    bp_lo_1 = None
    bp_ind_8, bp_ind_1 = -1, -1
    for i in range(_BREAKPOINTS_COUNT):
        bp_hi_8 = breakpoint_8[i]
        if bp_hi_8 is None:
            break
        if c8 < bp_hi_8:
            bp_ind_8 = i
            break
        bp_lo_8 = bp_hi_8 + pollutant_info.step

    for i in range(_BREAKPOINTS_COUNT):
        bp_hi_1 = breakpoint_1[i]
        if bp_hi_1 is None:
            continue
        if c1 < bp_hi_1:
            if bp_lo_1 is None:
                break
            else:
                bp_ind_1 = i
                break
        bp_lo_1 = bp_hi_1 + pollutant_info.step

    return [bpd.BreakpointDto(bp_ind_8, bp_lo_8, bp_hi_8, c8), bpd.BreakpointDto(bp_ind_1, bp_lo_1, bp_hi_1, c1)]


def __get_concentration_breakpoints_so2(pollutant_info: pld.PollutantDto, c1: float, c24: float) \
        -> bpd.BreakpointDto:
    interval_1 = pollutant_info.hourly_intervals[0]
    interval_2 = pollutant_info.hourly_intervals[1]
    breakpoints_1 = pollutant_info.breakpoints_by_intervals[interval_1]
    breakpoints_24 = pollutant_info.breakpoints_by_intervals[interval_2]

    bp_lo, bp_hi = 0, 0
    bp_ind = -1
    c = c1
    to24 = False
    for i in range(_BREAKPOINTS_COUNT):
        bp_hi = breakpoints_1[i] if not to24 else breakpoints_24[i]
        if bp_hi is None:
            c = c24
            bp_hi = breakpoints_24[i]
            to24 = True

        if c < bp_hi:
            bp_ind = i
            break
        bp_lo = bp_hi + pollutant_info.step

    return bpd.BreakpointDto(bp_ind, bp_lo, bp_hi, c)


def __get_aqi_breakpoints(bp_ind):
    aqi_lo, aqi_hi = 0, 0
    if bp_ind > -1:
        aqi_hi = _AQI_INFO_SHORT[bp_ind]
        aqi_lo = _AQI_INFO_SHORT[bp_ind - 1] if bp_ind > 0 else 0
    return aqi_lo, aqi_hi


def __aqi_calc_formula(bp: bpd.BreakpointDto) -> int:
    aqi_lo, aqi_hi = __get_aqi_breakpoints(bp.bp_ind)
    aqi = int(round((aqi_hi - aqi_lo) * (bp.c - bp.bp_lo) / (bp.bp_hi - bp.bp_lo) + aqi_lo, 0))
    return aqi
