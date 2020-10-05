

import numpy as np
import numba as nb
import sys

import matplotlib.pyplot as plt

import timeit, functools, collections

from enum import Enum

class T_DATA(Enum):
    INTERPOLATE = 1
    CALCULATE   = 2
T_DATA_MEMTYPE = nb.typeof( T_DATA.CALCULATE )

class T_ATOM(Enum):
    HYDROGEN      = 1
    HYDROGEN_LIKE = 2
    NORMAL        = 3
T_ATOM_MEMTYPE = nb.typeof( T_ATOM.HYDROGEN )

def func0(x,y):

    for k in range(10):
        z = x * y
        z /= np.exp(x + y)
        z *= np.sin(x) * np.cos(y)

    return z

def func0_loop(x,y):

    nL = x.shape[0]
    z = np.empty(nL, dtype=np.float64)

    for k in range(10):
        for j in range(nL):
            z[j] = x[j] * y[j]
            z[j] /= np.exp(x[j] + y[j])
            z[j] *= np.sin(x[j]) * np.cos(y[j])

    return z

def func(x,y, key1, key2):

    for k in range(10):

        if key1 == 1:
            if key2 == 1:
                z = x * y
                z /= np.exp(x + y)
                z *= np.sin(x) * np.cos(y)
            else:
                z = x * y
                z /= np.exp(x + y)
                z *= np.sin(x) * np.cos(y)
        else:
            z = x * y

    return z



def func_enum(x,y, ekey1, ekey2):

    for k in range(10):
        if ekey1 == T_ATOM.HYDROGEN:
            if ekey2 == T_DATA.CALCULATE:
                z = x * y
                z /= np.exp(x + y)
                z *= np.sin(x) * np.cos(y)
            else:
                z = x * y
                z /= np.exp(x + y)
                z *= np.sin(x) * np.cos(y)
        else:
            z = x * y

    return z

func0_nb = nb.njit( [nb.float64[:](nb.float64[:],nb.float64[:]) ] ) (func0)
func0_nbvec = nb.vectorize( [nb.float64(nb.float64,nb.float64) ], nopython=True ) (func0)

func0_loop_nb = nb.njit( [nb.float64[:](nb.float64[:],nb.float64[:]) ] ) (func0_loop)

func_nb = nb.njit( [nb.float64[:](nb.float64[:],nb.float64[:],nb.int64,nb.int64) ] ) (func)
func_nbvec = nb.vectorize( [nb.float64(nb.float64,nb.float64,nb.int64,nb.int64) ], nopython=True ) (func)

func_enum_nb = nb.njit( [nb.float64[:](nb.float64[:],nb.float64[:],T_ATOM_MEMTYPE,T_DATA_MEMTYPE)] )(func_enum)

def print_table(ns, result):


    nbar = 16 + 16*len(result.keys())
    s = '-'*nbar + '\n'
    s += "unit [1E-6 sec] "
    for key in result.keys():
        s += f"| {key:^13s} "
    s += '\n'
    s += '-'*nbar + '\n'

    for k in range(len(ns)):
        s += f"n={str(ns[k]):<13s} "
        for val in result.values():
            ss = f"{val[k]*1E6:<.1f}"
            s +=  f"| {ss:>9s}     "
        s += '\n'
    s += '-'*nbar + '\n'

    print(s)

if __name__ == "__main__":

    _result = {
        "np (noif)" : [],
        "np" : [],
        "np+Enum" : [],
        "nb (noif)" : [],
        'nb loop (noif)' : [],
        "nb vec (noif)" : [],
        "nb" : [],
        "nb vec" : [],
        "nb+Enum" : [],
    }
    _result = collections.OrderedDict( _result )

    _nLoop = 1000


    _ns = [10, 100, 1000, 10000]#, 100000]

    _dtype = np.dtype([('x', 'f8'), ('y', np.float64)])

    for _n in _ns:

        print(f"n = {_n}")

        _p = np.recarray(_n, dtype=_dtype)
        _p['x'][:] = np.random.rand(_n)[:]
        _p['y'][:] = np.random.rand(_n)[:]

        _t = timeit.Timer( functools.partial( func0, _p['x'], _p['y'] ) )
        _result["np (noif)"].append( _t.timeit(_nLoop)/_nLoop )

        _t = timeit.Timer( functools.partial( func, _p['x'], _p['y'], 1, 1 ) )
        _result["np"].append( _t.timeit(_nLoop)/_nLoop )

        _t = timeit.Timer( functools.partial( func_enum, _p['x'], _p['y'], T_ATOM.HYDROGEN, T_DATA.CALCULATE ) )
        _result["np+Enum"].append( _t.timeit(_nLoop)/_nLoop )

        _ = func0_nb( _p['x'], _p['y'] )
        _t = timeit.Timer( functools.partial( func0_nb, _p['x'], _p['y'] ) )
        _result["nb (noif)"].append( _t.timeit(_nLoop)/_nLoop )

        _ = func0_loop_nb( _p['x'], _p['y'] )
        _t = timeit.Timer( functools.partial( func0_loop_nb, _p['x'], _p['y'] ) )
        _result["nb loop (noif)"].append( _t.timeit(_nLoop)/_nLoop )

        _ = func0_nbvec( _p['x'], _p['y'] )
        _t = timeit.Timer( functools.partial( func0_nbvec, _p['x'], _p['y'] ) )
        _result["nb vec (noif)"].append( _t.timeit(_nLoop)/_nLoop )

        _ = func_nb( _p['x'], _p['y'], 1, 1 )
        _t = timeit.Timer( functools.partial( func_nb, _p['x'], _p['y'], 1, 1 ) )
        _result["nb"].append( _t.timeit(_nLoop)/_nLoop )

        _ = func_nbvec( _p['x'], _p['y'], 1, 1 )
        _t = timeit.Timer( functools.partial( func_nbvec, _p['x'], _p['y'], 1, 1 ) )
        _result["nb vec"].append( _t.timeit(_nLoop)/_nLoop )

        _ = func_enum_nb( _p['x'], _p['y'], T_ATOM.HYDROGEN, T_DATA.CALCULATE )
        _t = timeit.Timer( functools.partial( func_enum_nb, _p['x'], _p['y'], T_ATOM.HYDROGEN, T_DATA.CALCULATE ) )
        _result["nb+Enum"].append( _t.timeit(_nLoop)/_nLoop )

    print_table(_ns, _result)
#-----------------------------------------------------------------------------

    _fig, _axs = plt.subplots(1,1,figsize=(6,4), dpi=100)
    _ax = _axs
    for _key in _result.keys():
        _ax.plot(_ns, _result[_key], "--o", markersize=5, label=_key)
    _ax.legend(loc="best")
    _ax.set_yscale("log")
    _ax.set_xscale("log")
    _ax.set_title( "integer keyword v.s. Enum keyword" )
    _ax.set_ylim(1E-6, 1E-1)

    _fig.savefig(sys.argv[0].replace('.py', '.png'), dpi=100)
#-----------------------------------------------------------------------------
