
import numba as nb

from enum import Enum



class T_DATA(Enum):
    INTERPOLATE = 1
    CALCULATE   = 2
T_DATA_NBTYPE = nb.typeof(T_DATA)
T_DATA_MEMBER_NBTYPE = nb.typeof(T_DATA.INTERPOLATE)



@nb.njit( [nb.int64(nb.float64, T_DATA_MEMBER_NBTYPE)] )
def test_func1( x, _type_ ):

    if _type_ == T_DATA.CALCULATE:
        print(T_DATA.CALCULATE)
    elif _type_ == T_DATA.INTERPOLATE:
        print(T_DATA.INTERPOLATE)

    return 1

if __name__ == "__main__":

    print(test_func1( 1., T_DATA.INTERPOLATE ))

    print(nb.typeof(T_DATA))
    print(nb.typeof(T_DATA.INTERPOLATE))
