"""Storage for AQI breakpoint information"""


class BreakpointDto:
    """Storage for AQI breakpoint information"""

    def __init__(self, breakpoint_ind: int, breakpoint_low: float,
                 breakpoint_high: float, concentration: float):
        self.__breakpoint_ind = breakpoint_ind
        self.__breakpoint_low = breakpoint_low
        self.__breakpoint_high = breakpoint_high
        self.__concentration = concentration

    @property
    def index(self) -> int:
        """
        Breakpoint index
        """
        return self.__breakpoint_ind

    @property
    def low_border(self) -> float:
        """
        Breakpoint low border
        """
        return self.__breakpoint_low

    @property
    def high_border(self) -> float:
        """
        Breakpoint high border
        """
        return self.__breakpoint_high

    @property
    def concentration(self) -> float:
        """
        Concentration of the pollutant
        """
        return self.__concentration
