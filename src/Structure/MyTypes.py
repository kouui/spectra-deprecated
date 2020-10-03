

from enum import Enum

class T_DATA(Enum):
    INTERPOLATE = 1
    CALCULATE   = 2

class T_ATOM(Enum):
    HYDROGEN      = 1
    HYDROGEN_LIKE = 2
    NORMAL        = 3
