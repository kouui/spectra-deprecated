
import numpy as np

#import sys
#sys.path.append("../../")

from ...Atomic import LTELib, BasicP
from ...Atomic import PhotoIonize, Collision

from ...Structure.MyTypes import T_ATOM, T_DATA
from ...Atomic import Hydrogen, LTELib
from ... import Constants as Cst

from . import LibArray
from ..Hydrogen import LevelN

#-----------------------------------------------------------------------------
# if the spatial matrix is too large, then create temperal multi-dimensional
# matrix to store transition rates values (transition,x,y,z) could be very
# expensive. (50, 50, 50, 50) ~50MB for float64 might be a proper size.
#-----------------------------------------------------------------------------


def atom_gamma_Gamma(_atom):
    r""" """
    BasicP.get_Level_gamma(Aji=_atom.Line.AJI[:],idxJ=_atom.Line.idxJ[:],
                           gamma=_atom.Level.gamma[:])
    BasicP.get_Line_Gamma(idxI=_atom.Line.idxI[:], idxJ=_atom.Line.idxJ[:],
                        gamma=_atom.Level.gamma[:], Gamma=_atom.Line.Gamma[:])


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


def bf_R_rate(_atom, _Te, _nj_by_ni_LTE, _Tr=None):
    r""" """

    if _Tr is None:
        assert False, "only computation with radiation temperature is available."

    else:
        _PI_I = []
        for k in range(_atom.nCont):
            _PI_I.append( LTELib.Planck_cm(_atom.Mesh.Cont[k,:],_Tr) )
        _PI_I = np.array( _PI_I )

    #-------------------------------------------------------------------------
    # we compute/interpolate photoionizatoin cross section only once
    # and assume that while suffering Doppler shift
    #    - continuum wavelength mesh might shift
    #    - photoionizatoin cross section keep constant
    #-------------------------------------------------------------------------

    _PI_alpha = _atom.PI_alpha[:,:]

    ## ! could not be optimized to Te-Ne-array
    _Rik, _Rki_stim, _Rki_spon = LibArray.bf_R_rate(_waveMesh=_atom.Mesh.Cont,
                                       _Jnu=_PI_I,
                                       _alpha=_PI_alpha,
                                       _Te=_Te,
                                       _nj_by_ni_LTE=_nj_by_ni_LTE[:])

    return _Rik, _Rki_stim, _Rki_spon

def B_Jbar_v0(_atom, _Tr):
    r""" """
    _Level = _atom.Level
    _Line  = _atom.Line
    #_lineIndex = _atom.I_Rad.lineIndex
    _lineIndex = _atom.Mesh.Coe.lineIndex[:]

    _Jbars = LTELib.Planck_cm(_atom.Line.w0[_lineIndex],_Tr)
    _Bij_Jbar0, _Bji_Jbar0 = LibArray.B_Jbar_v0(_Level, _Line, _lineIndex, _Jbar=_Jbars)

    _Bij_Jbar = np.zeros(_atom.Line.AJI.shape, np.double)
    _Bji_Jbar = np.zeros(_atom.Line.AJI.shape, np.double)

    for k in range(_Bij_Jbar0.shape[0]):
        _Bij_Jbar[_lineIndex[k]] = _Bij_Jbar0[k]
        _Bji_Jbar[_lineIndex[k]] = _Bji_Jbar0[k]

    return _Bij_Jbar, _Bji_Jbar

def B_Jbar(_atom, _Tr=None):
    r""" """
    _Level = _atom.Level
    _Line  = _atom.Line
    _MeshCoe = _atom.Mesh.Coe

    if _Tr is None:
        ## ! could not be optimized to Te-Ne-array
        ## ! need to loop over Te-Be-array
        assert False, "not yet with background radiation"
    else:
        ## ! could be optimized to Te-Ne-array
        _Bij_Jbar, _Bji_Jbar = LibArray.B_Jbar_Tr(_Line["AJI"][:], _Line["BJI"][:], _Line["BIJ"][:], _Line["w0"][:], _Tr)

    return _Bij_Jbar, _Bji_Jbar



def CEij_rate_coe(_atom, _Te):
    r""" """

    #print()
    if _atom.ATOM_DATA_TYPES.CE == T_DATA.INTERPOLATE:

        _Omega_table = _atom.CE.Omega_table[:,:]
        _Te_table    = _atom.CE.Te_table[:]
        _Coe         = _atom.CE.Coe
        ## ! could/better to be optimized to Te-Ne-array
        _CEij = LibArray.CEij_rate_coe(_Omega_table, _Te_table, _Coe, _Te)

    elif _atom.ATOM_DATA_TYPES.CE == T_DATA.CALCULATE:

        _Line  = _atom.Line

        if _atom.ATOM_TYPE == T_ATOM.HYDROGEN:
            ## ! could/better to be optimized to Te-Ne-array
            _CEij = Hydrogen.CE_rate_coe(_Line['ni'][:], _Line['nj'][:], _Te)
        else:
            assert False

    else:
        assert False

    return _CEij

def CIik_rate_coe(_atom, _Te):
    r""" """

    if _atom.ATOM_DATA_TYPES.CI == T_DATA.INTERPOLATE:

        _Omega_table = _atom.CI.Omega_table[:,:]
        _Te_table    = _atom.CI.Te_table[:]
        _Coe         = _atom.CI.Coe
        ## ! could/better to be optimized to Te-Ne-array
        _CIik = LibArray.CIik_rate_coe(_Omega_table, _Te_table, _Coe, _Te)

    elif _atom.ATOM_DATA_TYPES.CI == T_DATA.CALCULATE:

        _Cont  = _atom.Cont

        if _atom.ATOM_TYPE == T_ATOM.HYDROGEN:
            ## ! could/better to be optimized to Te-Ne-array
            _CIik = Hydrogen.CI_rate_coe(_Cont['ni'][:], _Te)
        else:
            assert False

    else:
        assert False

    return _CIik

def get_Cij(_atom, _Te):
    r""" """
    ## ! could/better to be optimized to Te-Ne-array
    _CEij = CEij_rate_coe(_atom, _Te)
    if _atom.hasContinuum:
        ## ! could/better to be optimized to Te-Ne-array
        _CIik = CIik_rate_coe(_atom, _Te)
        _Cij = np.append( _CEij[:], _CIik[:] )
    else:
        _Cij = _CEij

    return _Cij

def solve_SE(_atom, _Ne, _Cji, _Cij, _Bji_Jbar, _Bij_Jbar, _Rki_spon, _Rki_stim, _Rik):
    r"""
    """
    _nLevel = _atom.nLevel

    _idxI = np.append(_atom.Line.idxI[:], _atom.Cont.idxI[:])
    _idxJ = np.append(_atom.Line.idxJ[:], _atom.Cont.idxJ[:])

    _Rji_spon = np.append(_atom.Line.AJI[:], _Rki_spon[:])
    _Rji_stim = np.append(_Bij_Jbar[:], _Rki_stim[:])
    _Rij = np.append(_Bij_Jbar[:], _Rik[:])

    ## ! could/better to be optimized to Te-Ne-array
    _n_SE = LibArray.solve_SE(_nLevel, _idxI, _idxJ, _Cji, _Cij, _Rji_spon, _Rji_stim, _Rij, _Ne)

    return _n_SE
