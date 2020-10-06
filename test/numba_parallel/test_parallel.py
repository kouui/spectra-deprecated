import numpy as np
import numba as nb
import sys
import timeit, functools, collections

import matplotlib.pyplot as plt

sys.path.append('../../')

from src.Util import Benchmark

import math


@nb.vectorize( ['float64(float64)'], nopython=True, target='parallel' )
def f_vec(x):

    z = 0.
    for k in range(10):
        #print(k)
        for i in range(1000000):
            z += x * np.sin(x)

    return z

@nb.njit( ['float64(float64[:])'], parallel=True )
def f_njit(x):

    z = 0.
    for j in range(x.shape[0]):
        for k in nb.prange(10):
            print(k)
            for i in range(1000000):
                z += x[j] * np.sin(x[j])

    return z


if __name__ == "__main__":

    _nLoop = 10

    _ns = [10, 100]#, 1000, 10000]

    _functions = collections.OrderedDict({
        #"vectorize" : f_vec,
        "njit" : f_njit,
    })
    _result = collections.OrderedDict()
    for _key in _functions.keys():
        _result[_key] = []

    for _n in _ns:

        print(f"n = {_n}")

        _x = np.random.rand(_n)[:]
        _y = np.random.rand(_n)[:]

        for _name, _fun in _functions.items():

            print(f"running {_name}")

            _ = _fun( _x)
            _t = timeit.Timer( functools.partial( _fun, _x) )
            _result[_name].append( _t.timeit(_nLoop)/_nLoop )
