
import numpy as np
import numba as nb
import sys
import timeit, functools, collections

sys.path.append('../../')

from src.Util import Benchmark

#import warnings
#warnings.simplefilter('ignore')

from enum import Enum

class T_DATA(Enum):
    INTERPOLATE = 1
    CALCULATE   = 2

class T_ATOM(Enum):
    HYDROGEN      = 1
    HYDROGEN_LIKE = 2
    NORMAL        = 3



def func(x, y, key1, key2):
    for k in range(100):
        z = x * y
        #z /= np.exp(x + y)
        #for k in range(10):
        z /= np.exp(x + y)
        if key1 == 1:
            if key2 == 1:
                z *= 1.0
        #x *= y
        z *= np.sin(x) * np.cos(y)
    return z

def func_enum(x, y, key1, key2):
    for k in range(100):
        z = x * y
        #z /= np.exp(x + y)
        #for k in range(10):
        z /= np.exp(x + y)
        if key1 == T_DATA.INTERPOLATE.value:
            if key2 == T_ATOM.HYDROGEN.value:
                z *= 1.0
        #x *= y
        z *= np.sin(x) * np.cos(y)
    return z


sig = [nb.float64[:](nb.float64[:],nb.float64[:], nb.int64,nb.int64 ) ]
func_nb = nb.njit( sig ) (func)
func_nb_fas = nb.njit( sig , fastmath=True ) (func)
func_nb_par = nb.njit( sig , parallel=True ) (func)
func_nb_fas_par = nb.njit( sig, fastmath=True, parallel=True ) (func)

sig = [nb.float64(nb.float64,nb.float64, nb.int64,nb.int64 ) ]
func_nb_vec = nb.vectorize( sig ) (func)
func_enum_nb_vec = nb.vectorize( sig ) (func_enum)
func_nb_vec_par = nb.vectorize( sig, target='parallel' ) (func)


if __name__ == "__main__":

    _functions = collections.OrderedDict({
        "numpy" : func,
        "nj" : func_nb,
        "nj+fas" : func_nb_fas,
        "nj+par" : func_nb_par,
        "nj+fas+par" : func_nb_fas_par,
        "vec" : func_nb_vec,
        "vec+enum" : func_enum_nb_vec,
        "vec+par" : func_nb_vec_par,
    })
    _result = collections.OrderedDict()
    for _key in _functions.keys():
        _result[_key] = []

    _nLoop = 10


    _ns = [10, 100, 1000, 10000, 100000]

    for _n in _ns:

        print(f"n = {_n}")

        _x = np.random.rand(_n)[:]
        _y = np.random.rand(_n)[:]

        for _name, _fun in _functions.items():

            _ = _fun( _x, _y, 1, 1 )
            _t = timeit.Timer( functools.partial( _fun, _x, _y, 1, 1 ) )
            _result[_name].append( _t.timeit(_nLoop)/_nLoop )

    Benchmark.print_table(_ns, _result, outFile='./result.txt')
    Benchmark.make_figure(_ns, _result, './result.png', title="which numba decorator to use?")
