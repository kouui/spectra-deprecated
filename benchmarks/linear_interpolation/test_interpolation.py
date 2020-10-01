#-----------------------------------------------------------------------------
# result :
#   - numpy.interp is the fastest
#-----------------------------------------------------------------------------


import sys
sys.path.append("../../")

import timeit, functools

import numpy as np
from numpy import interp as interp_numpy
from scipy.interpolate import interp1d as interp_scipy_
from src.Util.External.interpolation import interp as interp_numba

import numba as nb

import matplotlib.pyplot as plt

def search_interp_numpy(u, x, y):

    _idx = np.searchsorted( x, [u[0]-5, u[-1]+5] )
    return interp_numpy( u[:], x[_idx[0]:_idx[1]], y[_idx[0]:_idx[1]] )


def interp_scipy(x, y, u):

    f = interp_scipy_(x, y)
    return f( u )

@nb.njit([nb.float64[:](nb.float64[:],nb.float64[:],nb.float64[:])])
def interp_numpy_numba(x, y, u):

    return interp_numpy(u, x, y)

def make_figure(x, result, save_path):

    fig, ax = plt.subplots(1,1,figsize=(6,4), dpi=100)
    for key in result.keys():
        ax.plot(x, result[key], "--o", markersize=5, label=key)

    ax.legend(loc="best")
    ax.set_yscale("log")
    ax.set_xscale("log")

    ax.set_title( "Benchmark for Linear Interpolation" )

    fig.savefig(save_path, dpi=100)

if __name__ == "__main__":

    #_x = np.linspace(0,1,100)**2 # non-uniform points
    #_y = np.linspace(0,1,100)    # values
    #_u = np.linspace(0,1,1000)

    #_ = interp_numpy( _u, _x, _y )

    #_ = interp_numba( _x, _y ,_u )

    #_f = interp_scipy_( _x, _y )
    #_ = _f( _u )

    _backRad = np.load("../../data/intensity/atlas/QS/atlas_QS.npy")
    _x = _backRad[0,:]
    _y = _backRad[1,:]


    _isSearch = True if sys.argv[2]=='full' else False
    if _isSearch:
        _x1 = 0
        _x2 = _backRad.shape[1]
        _w1 = 1000
        _w2 = 11000
        _r = _w2 - _w1
    else:
        _w1, _r = int(sys.argv[1]), int(sys.argv[2])
        _w2 = _w1 + + _r
        _x1 = np.where(_x<_w1)[0][-1]
        _x2 = np.where(_x>_w2)[0][0]+1

    _x = _backRad[0,_x1:_x2]
    _y = _backRad[1,_x1:_x2]

    print(f"wavelength range {_r}[A] with array size equals {_x2-_x1}")


    _result = {
        "numpy.interp" : [],
        "scipy.interpolate.interp1d" : [],
        "interpolation.interp(numba)" : [],
        "numpy.interp(numba)" : [],
    }

    if _isSearch:
        _result["numpy.searchsorted + numpy.interp"] = []


    _k = 1000
    _ns = (2,10,100,1000,10000,100000)

    _f = interp_scipy_(_x, _y)
    for _n in _ns:

        _u = np.linspace(_w1,_w2,_n, endpoint=True, dtype=np.float64)

        _t = timeit.Timer( functools.partial( interp_numpy, _u, _x, _y ) )
        _result["numpy.interp"].append( _t.timeit(_k)/_k )
        #print("tested numpy")

        if _isSearch:
            _t = timeit.Timer( functools.partial( search_interp_numpy, _u, _x, _y ) )
            _result["numpy.searchsorted + numpy.interp"].append( _t.timeit(_k)/_k )

        #_t = timeit.Timer( functools.partial( interp_scipy, _x, _y, _u ) )
        _t = timeit.Timer( functools.partial( _f, _u ) )
        _result["scipy.interpolate.interp1d"].append( _t.timeit(_k)/_k )
        #print("tested scipy")

        _ = interp_numba( _x, _y ,100 )
        _ = interp_numba( _x, _y ,1000 )
        _t = timeit.Timer( functools.partial( interp_numba, _x, _y, _u ) )
        _result["interpolation.interp(numba)"].append( _t.timeit(_k)/_k )
        #print("tested numba")

        _ = interp_numpy_numba( _x, _y ,_u )
        _ = interp_numpy_numba( _x, _y ,_u )
        _t = timeit.Timer( functools.partial( interp_numpy_numba, _x, _y, _u ) )
        _result["numpy.interp(numba)"].append( _t.timeit(_k)/_k )

    make_figure( _ns, _result, sys.argv[0].replace('.py', '.png') )
