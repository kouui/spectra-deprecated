
import numpy as np
import numba as nb
import sys
import timeit, functools, collections

import matplotlib.pyplot as plt

sys.path.append('../../')

from src.Util import Benchmark

#import warnings
#warnings.simplefilter('ignore')

from enum import Enum

import math




def f(x, y):
    for k in range(10):
        z = x + y
        z *= np.sin(x) * np.cos(y)
        z += np.exp( -np.abs(x+y) )
    return z

def f_math(x, y):
    for k in range(10):
        z = x + y
        z *= math.sin(x) * math.cos(y)
        z += math.exp( -abs(x+y) )
    return z

def f_math_njit(x, y):
    nL = x.shape[0]
    for k in range(10):
        z = np.empty(nL, dtype=np.double)
        for i in range(nL):
            z[i] = x[i] + y[i]
            z[i] *= math.sin(x[i]) * math.cos(y[i])
            z[i] += math.exp( -abs(x[i]+y[i]) )
    return z



sig = [nb.float64[:](nb.float64[:],nb.float64[:]) ]
f_nb = nb.njit( sig ) (f)
f_math_nb = nb.njit( sig ) (f_math_njit)

sig = [nb.float64(nb.float64,nb.float64 ) ]
f_vec = nb.vectorize( sig, nopython=True ) (f)
f_math_vec = nb.vectorize( sig, nopython=True ) (f_math)


if __name__ == "__main__":

    _functions = collections.OrderedDict({
        "np" : f,
        "np+njit" : f_nb,
        "math+njit" : f_math_nb,
        "np+vec" : f_vec,
        "math+vec" : f_math_vec,
    })
    _result = collections.OrderedDict()
    for _key in _functions.keys():
        _result[_key] = []

    _nLoop = 3


    _ns = [10, 100, 1000, 10000, 100000]

    for _n in _ns:

        print(f"n = {_n}")

        _x = np.random.rand(_n)[:]
        _y = np.random.rand(_n)[:]

        for _name, _fun in _functions.items():

            _ = _fun( _x, _y)
            _t = timeit.Timer( functools.partial( _fun, _x, _y) )
            _result[_name].append( _t.timeit(_nLoop)/_nLoop )

    Benchmark.print_table(_ns, _result, outFile='./result.txt')

    Benchmark.make_figure(_ns, _result, './result.png', title='#op~10, numpy v.s. math', _xlabel='array size n')
