
import numpy as np
import numba as nb
import sys
import timeit, functools, collections

import matplotlib.pyplot as plt

sys.path.append('../../')

from src.Util import Benchmark

import math




def f_sum(x):
    z = x.sum()
    return z

def f_sum_loop(x):
    nL = x.shape[0]
    z = 0.
    for i in range(nL):
        z += x[i]
    return z



sig = [nb.float64(nb.float64[:]) ]
f_sum_nb = nb.njit( sig ) (f_sum)
f_sum_loop_nb = nb.njit( sig ) (f_sum_loop)


def f_dot(x, y):
    z = np.dot(x, y)
    return z

def f_dot_loop(x, y):
    nL = x.shape[0]
    z = 0.
    for i in range(nL):
        z += x[i] * y[i]
    return z

sig = [nb.float64(nb.float64[:],nb.float64[:]) ]
f_dot_nb = nb.njit( sig ) (f_dot)
f_dot_loop_nb = nb.njit( sig ) (f_dot_loop)

def f_mul(x, y):
    z = x @ y
    return z

def f_mul_loop_nb(x, y):

    nRow = x.shape[0]
    nCol = y.shape[1]
    z = np.empty((nRow,nCol),dtype=np.double)

    for i in range(nRow):
        for j in range(nCol):
            z[i,j] = f_dot_loop_nb(x[i,:], y[:,j])

    return z

def f_mul_loop_loop(x, y):

    nRow = x.shape[0]
    nCol = y.shape[1]
    nD   = x.shape[1]
    z = np.empty((nRow,nCol),dtype=np.double)

    for i in range(nRow):
        for j in range(nCol):
            z[i,j] = 0
            for k in range(nD):
                z[i,j] += x[i,k] * y[k,j]
    return z

sig = [nb.float64[:,:](nb.float64[:,:],nb.float64[:,:]) ]
f_mul_nb = nb.njit( sig ) (f_mul)
f_mul_loop_nb_nb = nb.njit( sig ) (f_mul_loop_nb)
f_mul_loop_loop_nb = nb.njit( sig ) (f_mul_loop_loop)

def f_sol(a,b):
    x = np.linalg.solve(a, b)
    return x

def f_sol_loop(a,b):
    x = np.empty_like(b, dtype=np.double)
    for k in range(a.shape[0]):
        x[k,:] = np.linalg.solve(a[k,:,:],b[k,:])
    return x

def f_sol_loop_gu(a, b, x):
    x = np.empty_like(b, dtype=np.double)
    for k in range(a.shape[0]):
        x[k,:] = np.linalg.solve(a[k,:,:],b[k,:])

sig = [nb.float64[:,:](nb.float64[:,:,:],nb.float64[:,:]) ]
f_sol_loop_nb = nb.njit( sig ) (f_sol_loop)
f_sol_loop_guvec = nb.guvectorize( ['(float64[:,:,:],float64[:,:],float64[:,:])'], '(m,n,n),(m,n)->(m,n)' ) (f_sol_loop_gu)



if __name__ == "__main__":

    _nLoop = 10


    _ns = [10, 100, 1000, 10000, 100000, 1000000, 10000000]

    _functions = collections.OrderedDict({
        "np.sum" : f_sum,
        "np.sum+njit" : f_sum_nb,
        "loop+njit" : f_sum_loop_nb,
    })
    _result = collections.OrderedDict()
    for _key in _functions.keys():
        _result[_key] = []

    for _n in _ns:

        print(f"n = {_n}")

        _x = np.random.rand(_n)[:]
        _y = np.random.rand(_n)[:]

        for _name, _fun in _functions.items():

            _ = _fun( _x)
            _t = timeit.Timer( functools.partial( _fun, _x) )
            _result[_name].append( _t.timeit(_nLoop)/_nLoop )

    fig, axs = plt.subplots(2,2, figsize=(10,10), dpi=120, sharey=False)
    Benchmark.print_table(_ns, _result, outFile='./result.txt')
    Benchmark.make_figure(_ns, _result, './result.png', title='sum, numpy v.s. loop', _ax=axs[0,0])


    _functions = collections.OrderedDict({
        "np.dot" : f_dot,
        "np.dot+njit" : f_dot_nb,
        "loop+njit" : f_dot_loop_nb,
    })
    _result = collections.OrderedDict()
    for _key in _functions.keys():
        _result[_key] = []

    for _n in _ns:

        print(f"n = {_n}")

        _x = np.random.rand(_n)[:]
        _y = np.random.rand(_n)[:]

        for _name, _fun in _functions.items():

            _ = _fun( _x, _y)
            _t = timeit.Timer( functools.partial( _fun, _x, _y) )
            _result[_name].append( _t.timeit(_nLoop)/_nLoop )

    Benchmark.print_table(_ns, _result, outFile='./result.txt')
    Benchmark.make_figure(_ns, _result, './result.png', title='dot, numpy v.s. loop', _ax=axs[0,1])

##    _functions = collections.OrderedDict({
##        "np @" : f_mul,
##        "np @+njit" : f_mul_nb,
##        "loop(njit)+njit" : f_mul_loop_nb_nb,
##        "loop(loop)+njit" : f_mul_loop_loop_nb,
##    })
##    _result = collections.OrderedDict()
##    for _key in _functions.keys():
##        _result[_key] = []
##    _ns = [10, 100, 300]
##    for _n in _ns:
##
##        print(f"n = {_n}")
##
##        _x = np.random.rand(_n, _n)
##        _y = np.random.rand(_n, _n)
##
##        for _name, _fun in _functions.items():
##
##            _ = _fun( _x, _y)
##            _t = timeit.Timer( functools.partial( _fun, _x, _y) )
##            _result[_name].append( _t.timeit(_nLoop)/_nLoop )
##
##    Benchmark.print_table(_ns, _result, outFile='./result.txt')
##    Benchmark.make_figure(_ns, _result, './result.png', title='matrix mul, numpy v.s. loop', _xlabel='array size n -> nxn', _ax=axs[1,0])


    _functions = collections.OrderedDict({
        "np solve" : f_sol,
        "sol+loop" : f_sol_loop,
        "sol+loop+njit" : f_sol_loop_nb,
        "sol+loop+guvec" : f_sol_loop_guvec,
    })
    _result = collections.OrderedDict()
    for _key in _functions.keys():
        _result[_key] = []
    _ns = [10, 100, 1000, 10000]
    _nn = 100
    for _n in _ns:

        print(f"n = {_n}")

        _x = np.random.rand(_n, _nn, _nn)
        _y = np.random.rand(_n, _nn)

        for _name, _fun in _functions.items():
            print(f"running {_name}")
            
            _ = _fun( _x, _y)
            _t = timeit.Timer( functools.partial( _fun, _x, _y) )
            _result[_name].append( _t.timeit(_nLoop)/_nLoop )

    Benchmark.print_table(_ns, _result, outFile='./result.txt')
    Benchmark.make_figure(_ns, _result, './result.png', title='solve, only 2D solve supported in numba\n size:100x100', _xlabel='number of array', _ax=axs[1,0])

    _functions = collections.OrderedDict({
        "sol" : f_sol,
        "sol+loop" : f_sol_loop,
        "sol+loop+njit" : f_sol_loop_nb,
        "sol+loop+guvec" : f_sol_loop_guvec,
    })
    _result = collections.OrderedDict()
    for _key in _functions.keys():
        _result[_key] = []
    _ns = [10, 100, 1000]
    _nn = 100
    for _n in _ns:

        print(f"n = {_n}")

        _x = np.random.rand(_nn, _n, _n)
        _y = np.random.rand(_nn, _n)

        for _name, _fun in _functions.items():
            print(f"running {_name}")

            _ = _fun( _x, _y)
            _t = timeit.Timer( functools.partial( _fun, _x, _y) )
            _result[_name].append( _t.timeit(_nLoop)/_nLoop )

    Benchmark.print_table(_ns, _result, outFile='./result.txt')
    Benchmark.make_figure(_ns, _result, './result.png', title='solve, only 2D solve supported in numba\n number of array = 100', _xlabel='array size', _ax=axs[1,1])

    fig.savefig('./result.png', dpi=100)
