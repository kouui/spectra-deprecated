#-----------------------------------------------------------------------------
# result :
#   - string can be used as argument and keywrod argument in functions
#-----------------------------------------------------------------------------

import numba as nb

import numpy as np


@nb.njit( nb.float64[:](nb.float64[:], nb.types.unicode_type) )
def test_func(x, s='interpolate'):

    assert s in ("interpolate", "calculate")

    if s == 'interpolate':
        print(s)
        y = np.sin(x[:]) * np.sin(x[:]) * 2.

    else:
        print(s)
        y = np.cos(x[:]) * np.sin(x[:]) * 2.

    return y

@nb.njit
def test_func2(x, s='interpolate'):

    assert s in ("interpolate", "calculate")

    if s == 'interpolate':
        print(s)
        y = np.sin(x[:]) * np.sin(x[:]) * 2.

    else:
        print(s)
        y = np.cos(x[:]) * np.sin(x[:]) * 2.

    return y

if __name__ == "__main__":

    _x = np.linspace(-1,1, 101, endpoint=True, dtype=np.float64)

    _ = test_func(_x, s='interpolate')
    _ = test_func(_x, s='calculate')

    _ = test_func2(_x, s='interpolate')
    _ = test_func2(_x, s='calculate')
