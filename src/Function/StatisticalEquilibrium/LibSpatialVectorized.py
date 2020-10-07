import numpy as np
import numba as nb

from numpy import exp
#from math import exp
#-----------------------------------------------------------------------------
# module namespace
#-----------------------------------------------------------------------------
from ... import Constants as Cst
from ...Config import isJIT_


from ...Atomic import Collision, PhotoIonize, LTELib, SEsolver, BasicP, SEsolver
from ...RadiativeTransfer import Profile
from ...Math import Integrate
from ...Atomic import Hydrogen

from ...Structure.MyTypes import T_ATOM, T_DATA

#-----------------------------------------------------------------------------
# high level functions with class object as function argument
#-----------------------------------------------------------------------------

def ni_nj_LTE(_atom, _Te, _Ne):
    r""" """
    _Level = _atom.Level
    _Line  = _atom.Line
    _Cont  = _atom.Cont

    ## ! could be optimized to Te-Ne-array
    _nj_by_ni_L = LTELib.Boltzmann_distribution(_Line['gi'][:], _Line['gj'][:], Cst.h_ * Cst.c_ / _Line['w0'][:], _Te)
    _idxI_L = _atom.Line['idxI'][:]
    _idxJ_L = _atom.Line['idxJ'][:]

    _stage = _atom.Level['stage'][:]

    if _atom.hasContinuum:
        ## ! could be optimized to Te-Ne-array
        _nj_by_ni_C = LTELib.Saha_distribution(_Cont['gi'][:], _Cont['gj'][:], Cst.h_ * Cst.c_ / _Cont['w0'][:], _Ne, _Te)
        _idxI_C = _Cont['idxI'][:]
        _idxJ_C = _Cont['idxJ'][:]
    else:
        _nj_by_ni_C = None
        _idxI_C = None
        _idxJ_C = None

    ## ! could be optimized to Te-Ne-array
    _ni = LibArray.convert_nj_by_ni_to_ni(_nj_by_ni_L, _idxI_L, _idxJ_L, _stage,
                          _hasContinuum=_atom.hasContinuum,
                          _nj_by_ni_C=_nj_by_ni_C, _idxI_C=_idxI_C, _idxJ_C=_idxJ_C)

    return _ni, _nj_by_ni_L, _nj_by_ni_C

#-----------------------------------------------------------------------------
# Array level functions which could be optimized by numba
#-----------------------------------------------------------------------------
def nj_by_ni_To_ni(_nj_by_ni, _idxI, _idxJ, _isGround):
    r""" """

    _nLevel = _isGround.shape[0]
    _nLine  = _idxI.shape[0]
    _ni = np.ones(_nLevel, dtype=np.double)

    for k in range( _nLine ):
        i = _idxI[k]
        j = _idxJ[k]
        if _isGround[ i ]:
            _ni[ j ] = _nj_by_ni[ k ] * _ni[ i ]

    return _ni

#-----------------------------------------------------------------------------
# numba optimization
#-----------------------------------------------------------------------------
if isJIT_:
    nj_by_ni_To_ni = nb.njit( ['float64[:](float64[:], uint16[:], uint16[:], bool[:])'] ) (nj_by_ni_To_ni)
