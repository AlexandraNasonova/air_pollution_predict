class AqiDto:

    def __init__(self, category: str, aqi_from: int, aqi_to: int, color: tuple[int, int, int], value=0):
        self.__category = category
        self.__aqi_from = aqi_from
        self.__aqi_to = aqi_to
        self.__color = color
        self.__value = value

    @property
    def category(self):
        return self.__category

    @property
    def aqi_from(self):
        return self.__aqi_from

    @property
    def aqi_to(self):
        return self.__aqi_to

    @property
    def color(self):
        return self.__color

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, val):
        self.__value = val

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)