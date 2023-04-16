"""Storage for AQI information"""


class AqiDto:
    """Storage for AQI information"""
    def __init__(self, category: str, aqi_from: int, aqi_to: int,
                 color: tuple[int, int, int]):
        self.__category = category
        self.__aqi_from = aqi_from
        self.__aqi_to = aqi_to
        self.__color = color
        self.__value = 0

    @property
    def category(self) -> str:
        """
        Category of AQI value (Good, Moderate, Unhealthy etc.)
        """
        return self.__category

    @property
    def aqi_from(self) -> int:
        """
        Low border of the AQI category
        """
        return self.__aqi_from

    @property
    def aqi_to(self) -> int:
        """
        High border of the AQI category
        """
        return self.__aqi_to

    @property
    def color(self) -> tuple[int, int, int]:
        """
        RGB color for AQI category
        """
        return self.__color

    @property
    def value(self):
        """
        AQI value
        """
        return self.__value

    @value.setter
    def value(self, val):
        """
        AQI value - setter
        """
        self.__value = val

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
