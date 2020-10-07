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

from . import LibArray

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

def bf_R_rate(_atom, _Te, _nj_by_ni_Cont, _Tr=None):
    r""" """
    ## convert _Te, _Ne into numpy array
    _Te = numpy.atleast_1d( _Te )


    _Cont = _atom.Cont
    #_nCont = _atom.nCont
    _meshCont = _atom.Mesh.Cont[:,:]

    if _Tr is None:
        assert False, "only computation with radiation temperature is available."

    else:
        _PI_I = LTELib.Planck_cm(_meshCont[:,:],_Tr)

    #-------------------------------------------------------------------------
    # we compute/interpolate photoionizatoin cross section only once
    # and assume that while suffering Doppler shift
    #    - continuum wavelength mesh might shift
    #    - photoionizatoin cross section keep constant
    #-------------------------------------------------------------------------

    _PI_alpha = _atom.PI_alpha[:,:]

    ## ! could not be optimized to Te-Ne-array
    _Rik, _Rki_stim, _Rki_spon = bf_R_rate_loop(_waveMesh=_meshCont[:,:],
                                       _Jnu=_PI_I[:,:],
                                       _alpha=_PI_alpha[:,:],
                                       _Te=_Te,
                                       _nj_by_ni_Cont=_nj_by_ni_Cont)

    return _Rik, _Rki_stim, _Rki_spon

def B_Jbar(_atom, _Te, _Tr=None):
    r""" """
    ## convert _Te, _Ne into numpy array
    _Te = numpy.atleast_1d( _Te )


    _Level = _atom.Level
    _Line  = _atom.Line
    _nLine = _atom.nLine

    _MeshCoe = _atom.Mesh.Coe

    if _Tr is None:
        ## ! could not be optimized to Te-Ne-array
        ## ! need to loop over Te-Be-array
        assert False, "not yet with background radiation"
    else:
        ## ! could be optimized to Te-Ne-array
        _Jbar = LTELib.Planck_cm(_Line["w0"][:], _Tr)

        _Bji_Jbar = numpy.empty( (_nLine,)+_Te.shape, dtype=dtDOUBLE_ )
        _Bij_Jbar = numpy.empty_like( _Bji_Jbar, dtype=dtDOUBLE_ )
        _extra_dim = (1,)*_Te.ndim
        _Bji_Jbar[:] = (_Line["BJI"][:] * _Jbar[:]).reshape(_nLine,*_extra_dim)
        _Bij_Jbar[:] = (_Line["BIJ"][:] * _Jbar[:]).reshape(_nLine,*_extra_dim)

    return _Bij_Jbar, _Bji_Jbar

def get_Cij(_atom, _Te):
    r""" """
    ## convert _Te, _Ne into numpy array
    _Te = numpy.atleast_1d( _Te )
    _Te_1D = _Te.reshape(-1)

    ## grab arrays
    _nLine = _atom.nLine
    if _atom.hasContinuum:
        _nCont = _atom.nCont
        _nTran = _nLine + _nCont
    else:
        _nTran = _nLine

    _Cij = numpy.empty( (_nTran,)+_Te_1D.shape, dtype=dtDOUBLE_ )

    #------
    # for line transition
    #------
    if _atom.ATOM_DATA_TYPES.CE == T_DATA.INTERPOLATE:
        pass
        # interpolate CE

    elif _atom.ATOM_DATA_TYPES.CE == T_DATA.CALCULATE:

        _Line  = _atom.Line

        if _atom.ATOM_TYPE == T_ATOM.HYDROGEN:

            _ni = _Line['ni'][:]
            _nj = _Line['nj'][:]
            for k in range(_nLine):
                _Cij[k,:] = Hydrogen.CE_rate_coe(_ni[k], _nj[k], _Te_1D[:])

    else:
        assert False


    if _atom.hasContinuum:
        #------
        # for continuum transition
        #------

        if _atom.ATOM_DATA_TYPES.CI == T_DATA.INTERPOLATE:
            pass
            # interpolate CI

        elif _atom.ATOM_DATA_TYPES.CI == T_DATA.CALCULATE:

            _Cont  = _atom.Cont

            if _atom.ATOM_TYPE == T_ATOM.HYDROGEN:
                _ni = _Cont['ni'][:]
                for k in range(_nCont):
                    _Cij[k+_nLine,:]= Hydrogen.CI_rate_coe(_ni[k], _Te_1D[:])
        else:
            assert False

    return _Cij.reshape(_nTran,*_Te.shape)

def solve_SE(_atom, _Ne, _Cji, _Cij, _Bji_Jbar, _Bij_Jbar, _Rki_spon, _Rki_stim, _Rik):
    r""" """

    r""" """
    ## convert _Te, _Ne into numpy array
    _Ne = numpy.atleast_1d( _Ne )
    _Ne_1D    = _Ne.reshape(-1)
    _nSpatial = _Ne.size

    _nLevel = _atom.nLevel
    _nCont  = _atom.nCont
    _nLine  = _atom.nLine
    _nTran  = _nCont + _nLine

    _idxI = numpy.append(_atom.Line.idxI[:], _atom.Cont.idxI[:])
    _idxJ = numpy.append(_atom.Line.idxJ[:], _atom.Cont.idxJ[:])

    _Aji  = _atom.Line.AJI[:]

    #----
    # reshape spatial 1D
    #----
    _Rki_spon_1D = _Rki_spon.reshape(_nCont, -1)
    _Rki_stim_1D = _Rki_stim.reshape(_nCont, -1)
    _Rik_1D      = _Rik.reshape(_nCont, -1)
    _Bij_Jbar_1D = _Bij_Jbar.reshape(_nLine, -1)
    _Bji_Jbar_1D = _Bji_Jbar.reshape(_nLine, -1)
    _Cij_1D      = _Cij.reshape(_nTran, -1)
    _Cji_1D      = _Cji.reshape(_nTran, -1)

    _n_SE  = numpy.empty((_nLevel, _nSpatial), dtype=dtDOUBLE_)


    for k in range(_nSpatial):

        _Rji_spon = numpy.append(_Aji[:], _Rki_spon_1D[:,k])
        _Rji_stim = numpy.append(_Bji_Jbar_1D[:,k], _Rki_stim_1D[:,k])
        _Rij      = numpy.append(_Bij_Jbar_1D[:,k], _Rik_1D[:,k])

        _n_SE[:,k] = LibArray.solve_SE(_nLevel, _idxI, _idxJ,
                      _Cji_1D[:,k], _Cij_1D[:,k], _Rji_spon, _Rji_stim, _Rij, _Ne_1D[k])

    return _n_SE.reshape(_nLevel, *_Ne.shape)

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

def bf_R_rate_loop(_waveMesh, _Jnu, _alpha, _Te, _nj_by_ni_Cont):
    r""" """

    _nCont = _waveMesh.shape[0]
    ## flatten spatial ndarray
    _nSpatial = _Te.size
    _Rik      = numpy.empty( (_nCont,_nSpatial), dtype=dtDOUBLE_ )
    _Rki_stim = numpy.empty_like( _Rik, dtype=dtDOUBLE_ )
    _Rki_spon = numpy.empty_like( _Rik, dtype=dtDOUBLE_ )

    _Te_1D = _Te.reshape(-1)
    _nj_by_ni_Cont_2D = _nj_by_ni_Cont.reshape(_nj_by_ni_Cont.shape[0],-1)

    ## loop over continuum transition
    for kL in range(_nCont):
        ## loop over flattened 1D spatial points
        for kX in range(_nSpatial):

            _res = PhotoIonize.bound_free_radiative_transition_coefficient(
                                wave = _waveMesh[kL,::-1],
                                J = _Jnu[kL,::-1],
                                alpha = _alpha[kL,::-1],
                                Te = _Te_1D[kX],
                                nk_by_ni_LTE=_nj_by_ni_Cont_2D[kL,kX])

            _Rik[kL,kX]      = _res[0]
            _Rki_stim[kL,kX] = _res[1]
            _Rki_spon[kL,kX] = _res[2]

    ## convert back to the shape we want
    _shape = (_nCont,) + _Te.shape
    return _Rik.reshape(_shape), _Rki_stim.reshape(_shape), _Rki_spon.reshape(_shape)

#-----------------------------------------------------------------------------
# numba optimization
#-----------------------------------------------------------------------------
if isJIT_:
    nj_by_ni_To_ni = numba.njit( nj_by_ni_To_ni )
    bf_R_rate_loop = numba.njit( bf_R_rate_loop )
