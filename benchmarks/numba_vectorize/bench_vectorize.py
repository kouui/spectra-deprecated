
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




def func(x, y, nk):
    for k in range(nk):
        z = x * y
        z *= (x + y)
    return z



sig = [nb.float64[:](nb.float64[:],nb.float64[:], nb.int64) ]
func_nb = nb.njit( sig ) (func)


sig = [nb.float64(nb.float64,nb.float64, nb.int64 ) ]
func_nb_vec = nb.vectorize( sig, nopython=True ) (func)


if __name__ == "__main__":

    _functions = collections.OrderedDict({
        "numpy" : func,
        "nb.njit" : func_nb,
        "nb.vec" : func_nb_vec,
    })
    _result = collections.OrderedDict()
    for _key in _functions.keys():
        _result[_key] = []

    _nLoop = 2


    _ns = [10, 100, 1000, 10000, 100000]

    for _n in _ns:

        print(f"n = {_n}")

        _x = np.random.rand(_n)[:]
        _y = np.random.rand(_n)[:]

        for _name, _fun in _functions.items():

            _ = _fun( _x, _y, 100 )
            _t = timeit.Timer( functools.partial( _fun, _x, _y, 100 ) )
            _result[_name].append( _t.timeit(_nLoop)/_nLoop )

    Benchmark.print_table(_ns, _result, outFile='./result.txt')

    fig, axs = plt.subplots(1,3, figsize=(12,4), dpi=100, sharey=True)

    Benchmark.make_figure(_ns, _result, './result.png', title='function-loop k=100, array-size n changes', _ax=axs[0], _xlabel='array size n')


    _result = collections.OrderedDict()
    for _key in _functions.keys():
        _result[_key] = []

    _x = np.random.rand(10)[:]
    _y = np.random.rand(10)[:]
    for _n in _ns:

        print(f"n = {_n}")

        _x = np.random.rand(_n)[:]
        _y = np.random.rand(_n)[:]

        for _name, _fun in _functions.items():

            _ = _fun( _x, _y, _n )
            _t = timeit.Timer( functools.partial( _fun, _x, _y, _n ) )
            _result[_name].append( _t.timeit(_nLoop)/_nLoop )

    Benchmark.print_table(_ns, _result, outFile='./result.txt')

    Benchmark.make_figure(_ns, _result, './result.png', title='function-loop k changes, array-size n=10', _ax=axs[1], _xlabel='#operation k')

    _result = collections.OrderedDict()
    for _key in _functions.keys():
        _result[_key] = []

    _x = np.random.rand(100)[:]
    _y = np.random.rand(100)[:]
    for _n in _ns:

        print(f"n = {_n}")

        _x = np.random.rand(_n)[:]
        _y = np.random.rand(_n)[:]

        for _name, _fun in _functions.items():

            _ = _fun( _x, _y, _n )
            _t = timeit.Timer( functools.partial( _fun, _x, _y, _n ) )
            _result[_name].append( _t.timeit(_nLoop)/_nLoop )

    Benchmark.print_table(_ns, _result, outFile='./result.txt')

    Benchmark.make_figure(_ns, _result, './result.png', title='function-loop k changes, array-size n=100', _ax=axs[2], _xlabel='#operation k')

    fig.savefig('./result.png', dpi=100)
