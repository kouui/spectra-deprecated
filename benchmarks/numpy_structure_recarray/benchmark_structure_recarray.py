
import numpy as np
import numba as nb
import sys

import matplotlib.pyplot as plt

import timeit, functools, collections


def make_figure(x, result, save_path, title):

    fig, ax = plt.subplots(1,1,figsize=(6,4), dpi=100)
    for key in result.keys():
        ax.plot(x, result[key], "--o", markersize=5, label=key)

    ax.legend(loc="best")
    ax.set_yscale("log")
    ax.set_xscale("log")

    ax.set_title( title )

    fig.savefig(save_path, dpi=100)

def print_table(ns, result):

    s = '-'*120 + '\n'
    s += "unit [1E-6 sec] "
    for key in result.keys():
        s += f"| {key:^13s} "
    s += '\n'
    s += '-'*120 + '\n'

    for k in range(len(ns)):
        s += f"n={str(ns[k]):<13s} "
        for val in result.values():
            ss = f"{val[k]*1E6:<.1f}"
            s +=  f"| {ss:>9s}     "
        s += '\n'
    s += '-'*120 + '\n'

    print(s)


if __name__ == "__main__":

    _result = {
        "structure" : [],
        "recarray" : [],
        "recarray_attr" : [],
        "structure_nb" : [],
        "recarray_nb" : [],
        "recarray_attr_nb" : [],
    }
    _result = collections.OrderedDict( _result )

    _nLoop = 1000

    _ns = [10, 100, 1000, 10000, 100000]
    _dtype = np.dtype([('x', 'f8'), ('y', np.float64)])


    def func(x,y):
        for k in range(10):
            z = x * y
            z /= np.exp(x + y)
            z *= np.sin(x) * np.cos(y)
        return z

    func_nb = nb.vectorize( ['float64(float64, float64)'], nopython=True ) (func)

    for _n in _ns:

        print(f"n = {_n}")

        _p = np.ones(_n, dtype=_dtype)
        _p['x'][:] = np.random.rand(_n)[:]
        _p['y'][:] = np.random.rand(_n)[:]

        _t = timeit.Timer( functools.partial( func, _p['x'], _p['y'] ) )
        _result["structure"].append( _t.timeit(_nLoop)/_nLoop )

        _ = func_nb( _p['x'], _p['y'] )
        _t = timeit.Timer( functools.partial( func_nb, _p['x'], _p['y'] ) )
        _result["structure_nb"].append( _t.timeit(_nLoop)/_nLoop )

        _p = np.recarray(_n, dtype=_dtype)
        _p['x'][:] = np.random.rand(_n)[:]
        _p['y'][:] = np.random.rand(_n)[:]

        _t = timeit.Timer( functools.partial( func, _p['x'], _p['y'] ) )
        _result["recarray"].append( _t.timeit(_nLoop)/_nLoop )

        _ = func_nb( _p['x'], _p['y'] )
        _t = timeit.Timer( functools.partial( func_nb, _p['x'], _p['y'] ) )
        _result["recarray_nb"].append( _t.timeit(_nLoop)/_nLoop )

        _t = timeit.Timer( functools.partial( func, _p.x, _p.y ) )
        _result["recarray_attr"].append( _t.timeit(_nLoop)/_nLoop )

        _ = func_nb( _p.x, _p.y )
        _t = timeit.Timer( functools.partial( func_nb, _p.x, _p.y ) )
        _result["recarray_attr_nb"].append( _t.timeit(_nLoop)/_nLoop )

    #pprint(_result)
    print_table(_ns, _result)
#-----------------------------------------------------------------------------
    #make_figure( _ns, _result, sys.argv[0].replace('.py', '.input_array.png'), "numpy : structure v.s. recarray" )
    _fig, _axs = plt.subplots(1,2,figsize=(9,4), dpi=100, sharey=True)
    _ax = _axs[0]
    for _key in _result.keys():
        _ax.plot(_ns, _result[_key], "--o", markersize=5, label=_key)
    _ax.legend(loc="best")
    _ax.set_yscale("log")
    _ax.set_xscale("log")
    _ax.set_title( "structure v.s. recarray | input array" )
#-----------------------------------------------------------------------------

    _result = {
        "structure" : [],
        "structure_nb" : [],
        "recarray" : [],
        "recarray_nb" : [],
        "recarray_attr" : [],
        "recarray_attr_nb" : [],
    }
    _result = collections.OrderedDict( _result )

    def func_field(p):
        for k in range(10):
            x = p['x'][:]
            y = p['y'][:]
            z = x * y
            z /= np.exp(x + y)
            z *= np.sin(x) * np.cos(y)
        return z

    func_field_nb = nb.njit(func_field)

    for _n in _ns:

        print(f"n = {_n}")

        _p = np.ones(_n, dtype=_dtype)
        _p['x'][:] = np.random.rand(_n)[:]
        _p['y'][:] = np.random.rand(_n)[:]

        _t = timeit.Timer( functools.partial( func_field, _p ) )
        _result["structure"].append( _t.timeit(_nLoop)/_nLoop )

        _ = func_field_nb( _p )
        _t = timeit.Timer( functools.partial( func_field_nb, _p ) )
        _result["structure_nb"].append( _t.timeit(_nLoop)/_nLoop )

    func_field_nb = nb.njit(func_field)

    for _n in _ns:

        print(f"n = {_n}")

        _p = np.recarray(_n, dtype=_dtype)
        _p['x'][:] = np.random.rand(_n)[:]
        _p['y'][:] = np.random.rand(_n)[:]

        _t = timeit.Timer( functools.partial( func_field, _p ) )
        _result["recarray"].append( _t.timeit(_nLoop)/_nLoop )

        _ = func_field_nb( _p )
        _t = timeit.Timer( functools.partial( func_field_nb, _p ) )
        _result["recarray_nb"].append( _t.timeit(_nLoop)/_nLoop )

    def func_attr(p):
        for k in range(10):
            x = p.x[:]
            y = p.y[:]
            z = x * y
            z /= np.exp(x + y)
            z *= np.sin(x) * np.cos(y)
        return z

    func_attr_nb = nb.njit(func_attr)

    for _n in _ns:

        print(f"n = {_n}")

        _p = np.recarray(_n, dtype=_dtype)
        _p['x'][:] = np.random.rand(_n)[:]
        _p['y'][:] = np.random.rand(_n)[:]

        _t = timeit.Timer( functools.partial( func_attr, _p ) )
        _result["recarray_attr"].append( _t.timeit(_nLoop)/_nLoop )

        _ = func_attr_nb( _p )
        _t = timeit.Timer( functools.partial( func_attr_nb, _p ) )
        _result["recarray_attr_nb"].append( _t.timeit(_nLoop)/_nLoop )

    #pprint(_result)
    print_table(_ns, _result)
#-----------------------------------------------------------------------------
    _ax = _axs[1]
    for _key in _result.keys():
        _ax.plot(_ns, _result[_key], "--o", markersize=5, label=_key)
    _ax.legend(loc="best")
    _ax.set_yscale("log")
    _ax.set_xscale("log")
    _ax.set_title( "structure v.s. recarray | input structure" )

    _fig.savefig(sys.argv[0].replace('.py', '.png'), dpi=100)
