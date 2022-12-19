class PollutantDto:

    def __init__(self, pollutant_id: int, pollutant_code: str, hourly_intervals: list[int], decimals_count: int,
                 breakpoints_by_intervals: dict[int, list[float]], measure: str):
        self.__pollutant_id = pollutant_id
        self.__pollutant_code = pollutant_code
        self.__hourly_intervals = hourly_intervals
        self.__decimals_count = decimals_count
        self.__step = 1.0 / (10**decimals_count)
        self.__breakpoints_by_intervals = breakpoints_by_intervals
        self.__measure = measure

    @property
    def pollutant_id(self):
        return self.__pollutant_id

    @property
    def pollutant_code(self):
        return self.__pollutant_code

    @property
    def hourly_intervals(self):
        return self.__hourly_intervals

    @property
    def decimals_count(self):
        return self.__decimals_count

    @property
    def step(self):
        return self.__step

    @property
    def breakpoints_by_intervals(self):
        return self.__breakpoints_by_intervals

    @property
    def measure(self):
        return self.__measure
