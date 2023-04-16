# pylint: disable=E0401, R0913, R0914, W0703, R0902

"""Storage for Pollutant information"""


class PollutantDto:
    """Storage for Pollutant information"""
    def __init__(self, pollutant_id: int, pollutant_code: str,
                 hourly_intervals: list[int], decimals_count: int,
                 breakpoints_by_intervals: dict[int, list[float]], measure: str):
        self.__pollutant_id = pollutant_id
        self.__pollutant_code = pollutant_code
        self.__hourly_intervals = hourly_intervals
        self.__decimals_count = decimals_count
        self.__step = 1.0 / (10**decimals_count)
        self.__breakpoints_by_intervals = breakpoints_by_intervals
        self.__measure = measure

    @property
    def pollutant_id(self) -> int:
        """
        Pollutant EU standard identificator (1 - for SO2, etc)
        """
        return self.__pollutant_id

    @property
    def pollutant_code(self) -> str:
        """
        Pollutant formula code (SO2 etc)
        """
        return self.__pollutant_code

    @property
    def hourly_intervals(self) -> []:
        """
        The list of intervals (some pollutants must be calculated
        for 2 intervals, i.e. 1 hour and 8 hour)
        """
        return self.__hourly_intervals

    @property
    def decimals_count(self) -> int:
        """
        The required number of decimals for the calculated AQI
        """
        return self.__decimals_count

    @property
    def step(self):
        """
        The step length between the previous high border of the creak point
        and the next low one
        @return:
        """
        return self.__step

    @property
    def breakpoints_by_intervals(self) -> dict:
        """
        The list of breakpoints
        """
        return self.__breakpoints_by_intervals

    @property
    def measure(self) -> str:
        """
        The units of measurements
        """
        return self.__measure
