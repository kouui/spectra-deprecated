
import numpy as np
import numba as nb
import sys
import timeit, functools, collections

sys.path.append('../../')

from src.Util import Benchmark
from src.Structure import AtomCls
import src.Constants as Cst

from src.Atomic import LTELib

@nb.guvectorize([(nb.uint8[:], nb.uint8[:], nb.float64[:], nb.float64[:], nb.float64[:,:])], '(n),(n),(n),(m)->(n,m)', nopython=True, target='parallel')
def _nj_by_ni_Line_gu_nopy_par(gi, gj, w0, Te, nj_by_ni_L):

    nL = gi.shape[0]
    for l in nb.prange(nL*10):
        l = l // 10
        nj_by_ni_L[l] = LTELib.Boltzmann_distribution(gi[l], gj[l], Cst.h_ * Cst.c_ / w0[l], Te[:])

@nb.guvectorize([(nb.uint8[:], nb.uint8[:], nb.float64[:], nb.float64[:], nb.float64[:,:])], '(n),(n),(n),(m)->(n,m)', nopython=True)
def _nj_by_ni_Line_gu_nopy(gi, gj, w0, Te, nj_by_ni_L):

    nL = gi.shape[0]
    for l in range(nL*10):
        l = l // 10
        nj_by_ni_L[l] = LTELib.Boltzmann_distribution(gi[l], gj[l], Cst.h_ * Cst.c_ / w0[l], Te[:])


@nb.guvectorize([(nb.uint8[:], nb.uint8[:], nb.float64[:], nb.float64[:], nb.float64[:,:])], '(n),(n),(n),(m)->(n,m)')
def _nj_by_ni_Line_gu(gi, gj, w0, Te, nj_by_ni_L):

    nL = gi.shape[0]
    for l in range(nL*10):
        l = l // 10
        nj_by_ni_L[l] = LTELib.Boltzmann_distribution(gi[l], gj[l], Cst.h_ * Cst.c_ / w0[l], Te[:])

@nb.njit( ['float64[:,:](uint8[:],uint8[:],float64[:],float64[:])'] )
def _nj_by_ni_Line_njit(gi, gj, w0, Te):

    nj_by_ni_L = np.empty(gi.shape+Te.shape, dtype=np.double)
    nL = gi.shape[0]
    for l in range(nL*10):
        l = l // 10
        nj_by_ni_L[l,:] = LTELib.Boltzmann_distribution(gi[l], gj[l], Cst.h_ * Cst.c_ / w0[l], Te[:])

    return nj_by_ni_L

def ni_nj_LTE_gu_nopy_par(atom, Te, Ne):
    r""" """
    Level = atom.Level
    Line  = atom.Line
    Cont  = atom.Cont

    assert Te.shape == Ne.shape, "Te and Ne should have the same shape"

    ## ! could be optimized to Te-Ne-array
    nj_by_ni_L = _nj_by_ni_Line_gu_nopy_par(Line['gi'][:], Line['gj'][:], Line['w0'][:], Te[:,])
    return nj_by_ni_L

def ni_nj_LTE_gu_nopy(atom, Te, Ne):
    r""" """
    Level = atom.Level
    Line  = atom.Line
    Cont  = atom.Cont

    assert Te.shape == Ne.shape, "Te and Ne should have the same shape"

    ## ! could be optimized to Te-Ne-array
    nj_by_ni_L = _nj_by_ni_Line_gu_nopy(Line['gi'][:], Line['gj'][:], Line['w0'][:], Te[:,])
    return nj_by_ni_L

def ni_nj_LTE_gu(atom, Te, Ne):
    r""" """
    Level = atom.Level
    Line  = atom.Line
    Cont  = atom.Cont

    assert Te.shape == Ne.shape, "Te and Ne should have the same shape"

    ## ! could be optimized to Te-Ne-array
    nj_by_ni_L = _nj_by_ni_Line_gu(Line['gi'][:], Line['gj'][:], Line['w0'][:], Te[:,])
    return nj_by_ni_L

def ni_nj_LTE_njit(atom, Te, Ne):
    r""" """
    Level = atom.Level
    Line  = atom.Line
    Cont  = atom.Cont

    assert Te.shape == Ne.shape, "Te and Ne should have the same shape"

    ## ! could be optimized to Te-Ne-array
    nj_by_ni_L = _nj_by_ni_Line_njit(Line['gi'][:], Line['gj'][:], Line['w0'][:], Te[:,])
    return nj_by_ni_L

def ni_nj_LTE(atom, Te, Ne):
    r""" """
    Level = atom.Level
    Line  = atom.Line
    Cont  = atom.Cont

    assert Te.shape == Ne.shape, "Te and Ne should have the same shape"

    ## ! could be optimized to Te-Ne-array
    nj_by_ni_L = np.empty( (atom.nLine,)+Te.shape, dtype=np.double )
    Eji = Cst.h_ * Cst.c_ /Line['w0'][:]
    gi = Line['gi'][:]
    gj = Line['gj'][:]
    for l in range(atom.nLine):
        nj_by_ni_L[l,:] = LTELib.Boltzmann_distribution(gi[l], gj[l], Eji[l], Te[:])

    return nj_by_ni_L
#    idxI_L = Line['idxI'][:]
#    idxJ_L = Line['idxJ'][:]

#    stage = atom.Level['stage'][:]
#
#    if atom.hasContinuum:
#        ## ! could be optimized to Te-Ne-array
#        _nj_by_ni_C = LTELib.Saha_distribution(_Cont['gi'][:], _Cont['gj'][:], Cst.h_ * Cst.c_ / _Cont['w0'][:], _Ne, _Te)
#        _idxI_C = _Cont['idxI'][:]
#        _idxJ_C = _Cont['idxJ'][:]
#    else:
#        _nj_by_ni_C = None
#        _idxI_C = None
#        _idxJ_C = None
#
#    ## ! could be optimized to Te-Ne-array
#    _ni = LibArray.convert_nj_by_ni_to_ni(_nj_by_ni_L, _idxI_L, _idxJ_L, _stage,
#                          _hasContinuum=atom.hasContinuum,
#                          _nj_by_ni_C=_nj_by_ni_C, _idxI_C=_idxI_C, _idxJ_C=_idxJ_C)
#
#    return _ni, _nj_by_ni_L, _nj_by_ni_C



if __name__ == "__main__":

    _functions = collections.OrderedDict({
        "normal" : ni_nj_LTE,
        "guvec"  : ni_nj_LTE_gu,
        "guvec_nopy"  : ni_nj_LTE_gu_nopy,
        "guvec_nopy_par"  : ni_nj_LTE_gu_nopy_par,
        "njit"   : ni_nj_LTE_njit,
    })
    _result = collections.OrderedDict()
    for _key in _functions.keys():
        _result[_key] = []

    _nLoop = 3

    _atom, _ = AtomCls.InitAtom('../../data/conf/H.conf', isHydrogen=True)


    _ns = [10, 100, 1000, 10000, 100000, 1000000]

    for _n in _ns:

        print(f"n = {_n}")

        _Te = np.ones(_n, dtype=np.double) * 6.E3
        _Ne = np.ones(_n, dtype=np.double) * 1.E11

        for _name, _fun in _functions.items():

            _ = _fun( _atom, _Te, _Ne )
            _t = timeit.Timer( functools.partial( _fun, _atom, _Te, _Ne ) )
            _result[_name].append( _t.timeit(_nLoop)/_nLoop )

    Benchmark.print_table(_ns, _result, outFile='./result.txt')
    Benchmark.make_figure(_ns, _result, './result.png', title="which numba decorator to use?")
