import numpy
import numba

from numpy import exp
#from math import exp
#-----------------------------------------------------------------------------
# module namespace
#-----------------------------------------------------------------------------
from ... import Constants as Cst
from ...Config import isJIT_, dtDOUBLE_


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

    ## convert _Te, _Ne into numpy array
    _Te = numpy.atleast_1d( _Te )
    _Ne = numpy.atleast_1d( _Ne )


    ## grab arrays
    _Level = _atom.Level
    _Line  = _atom.Line
    _nLevel = _atom.nLevel
    _nLine = _atom.nLine
    if _atom.hasContinuum:
        _Cont  = _atom.Cont
        _nCont = _atom.nCont
        _nTran = _nLine + _nCont
    else:
        _nTran = _nLine

    ## initilize _nj_by_ni
    _nj_by_ni = numpy.empty( (_nTran,)+_Te.shape, dtype=dtDOUBLE_ )

    ## for line transitions
    _gi = _Line['gi'][:]
    _gj = _Line['gj'][:]
    _Eji = Cst.h_ * _Line['f0'][:]

    for k in range(0, _nLine):
        _nj_by_ni[k,:] = LTELib.Boltzmann_distribution(_gi[k], _gj[k], _Eji[k], _Te[:])

    _idxI_L = _atom.Line['idxI'][:]
    _idxJ_L = _atom.Line['idxJ'][:]

    #_stage = _atom.Level['stage'][:]

    ## if there is continuum transition
    if _atom.hasContinuum:

        _gi = _Cont['gi'][:]
        _gj = _Cont['gj'][:]
        _chi = Cst.h_ * _Cont['f0'][:]

        for k in range(_nCont):
            _nj_by_ni[k+_nLine,:] = LTELib.Saha_distribution(_gi[k], _gj[k], _chi[k], _Ne[:], _Te[:])

        _idxI = numpy.append( _Line['idxI'][:],_Cont['idxI'][:] )
        _idxJ = numpy.append( _Line['idxJ'][:],_Cont['idxJ'][:] )
    else:
        _idxI = _Line['idxI'][:]
        _idxJ = _Line['idxJ'][:]

    _isGround = _Level['isGround'][:]
    ## compute _ni
    _ni = nj_by_ni_To_ni(_nj_by_ni[:,], _idxI[:], _idxJ[:], _isGround[:], _nLine)

    return _ni, _nj_by_ni[:_nLine,:], _nj_by_ni[_nLine:,:]

#-----------------------------------------------------------------------------
# Array level functions which could be optimized by numba
#-----------------------------------------------------------------------------
def nj_by_ni_To_ni(_nj_by_ni, _idxI, _idxJ, _isGround, _nLine):
    r""" """

    _nLevel = _isGround.shape[0]
    _nTran  = _idxI.shape[0]

    _ni = numpy.empty((_nLevel,)+_nj_by_ni.shape[1:], dtype=dtDOUBLE_)
    _ni[0,:] = 1.

    for k in range(_nLine, _nTran):
        i = _idxI[k]
        j = _idxJ[k]
        if _isGround[ i ]:
            _ni[ j ] = _nj_by_ni[ k ] * _ni[ i ]

    for k in range(0, _nLine):
        i = _idxI[k]
        j = _idxJ[k]
        if _isGround[ i ]:
            _ni[ j ] = _nj_by_ni[ k ] * _ni[ i ]

    return _ni[:,:] / _ni[:,:].sum(axis=0)

#-----------------------------------------------------------------------------
# numba optimization
#-----------------------------------------------------------------------------
if isJIT_:
    nj_by_ni_To_ni = numba.njit( nj_by_ni_To_ni )
