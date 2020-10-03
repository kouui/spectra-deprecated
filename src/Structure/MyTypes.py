

from enum import Enum
import numba as nb

class T_DATA(Enum):
    INTERPOLATE = 1
    CALCULATE   = 2

#T_DATA_NBTYPE = nb.typeof(T_DATA.INTERPOLATE)

class T_ATOM(Enum):
    HYDROGEN      = 1
    HYDROGEN_LIKE = 2
    NORMAL        = 3

#T_ATOM_NBTYPE = nb.typeof(T_ATOM.HYDROGEN)
