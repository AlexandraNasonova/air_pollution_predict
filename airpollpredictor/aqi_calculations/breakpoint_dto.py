class BreakpointDto:

    def __init__(self, bp_ind: int, bp_lo: float, bp_hi: float, c: float):
        self.__bp_ind = bp_ind
        self.__bp_lo = bp_lo
        self.__bp_hi = bp_hi
        self.__c = c

    @property
    def bp_ind(self):
        return self.__bp_ind

    @property
    def bp_lo(self):
        return self.__bp_lo

    @property
    def bp_hi(self):
        return self.__bp_hi

    @property
    def c(self):
        return self.__c
