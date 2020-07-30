
import numpy as np

import sys
sys.path.append("../../")

from src.Atomic import LTELib, BasicP
from src.Atomic import PhotoIonize, Collision

import Lib

def atom_gamma_Gamma(_atom):
    r"""
    """
    BasicP.get_Level_gamma(Aji=_atom.Line.AJI[:],idxJ=_atom.Line.idxJ[:],
                           gamma=_atom.Level.gamma[:])
    BasicP.get_Line_Gamma(idxI=_atom.Line.idxI[:], idxJ=_atom.Line.idxJ[:],
                        gamma=_atom.Level.gamma[:], Gamma=_atom.Line.Gamma[:])


def ni_nj_LTE(_atom, _Te, _Ne):
    r"""
    """
    _Level = _atom.Level
    _Line  = _atom.Line
    _Cont  = _atom.Cont

    _n_LTE, _ni_LTE, _nj_LTE = Lib.ni_nj_LTE(_Level, _Line, _Cont, _Te, _Ne)

    return _n_LTE, _ni_LTE, _nj_LTE

def bf_R_rate(_atom, _Te, _ni_LTE, _nj_LTE, _Tr=None):
    r"""
    """

    if _Tr is None:
        print("only computation with radiation temperature is available.")

    else:
        _PI_I = []
        for k in range(_atom.nCont):
            _PI_I.append( LTELib.Planck_cm(_atom.Mesh.Cont[k],_Tr) )

    _PI_alpha = PhotoIonize.interpolate_PI_alpha(_atom.PI.alpha_table, _atom.Mesh.Cont)

    _Rik, _Rki_stim, _Rki_spon = Lib.bf_R_rate(_waveMesh=_atom.Mesh.Cont,
                                       _Jnu=0.5*np.array(_PI_I),
                                       _alpha=_PI_alpha,
                                       _Te=_Te,
                                       _ni_LTE=_ni_LTE[_atom.nLine:],
                                       _nj_LTE=_nj_LTE[_atom.nLine:])

    return _Rik, _Rki_stim, _Rki_spon

def B_Jbar(_atom, _Tr):
    r"""
    """
    _Level = _atom.Level
    _Line  = _atom.Line
    _lineIndex = _atom.I_Rad.lineIndex

    _Jbars = LTELib.Planck_cm(_atom.Line.w0[_atom.I_Rad.lineIndex],_Tr)
    _Bij_Jbar0, _Bji_Jbar0 = Lib.B_Jbar(_Level, _Line, _lineIndex, _Jbar=_Jbars)

    _Bij_Jbar = np.zeros(_atom.Line.AJI.shape, np.double)
    _Bji_Jbar = np.zeros(_atom.Line.AJI.shape, np.double)

    for k in range(_Bij_Jbar0.shape[0]):
        _Bij_Jbar[_lineIndex[k]] = _Bij_Jbar0[k]
        _Bji_Jbar[_lineIndex[k]] = _Bji_Jbar0[k]

    return _Bij_Jbar, _Bji_Jbar



def CEij_rate_coe(_atom, _Te):
    r"""
    """
    _Omega_table = _atom.CE.Omega_table[:,:]
    _Te_table    = _atom.CE.Te_table[:]
    _Coe         = _atom.CE.Coe
    _CEij = Lib.CEij_rate_coe(_Omega_table, _Te_table, _Coe, _Te)

    return _CEij

def CIik_rate_coe(_atom, _Te):
    r"""
    """
    _Omega_table = _atom.CI.Omega_table[:,:]
    _Te_table    = _atom.CI.Te_table[:]
    _Coe         = _atom.CI.Coe
    _CIik = Lib.CIik_rate_coe(_Omega_table, _Te_table, _Coe, _Te)

    return _CIik

def get_Cij(_atom, _Te):
    r"""
    """
    _CEij = CEij_rate_coe(_atom, _Te)
    if _atom.hasContinuum:
        _CIik = CIik_rate_coe(_atom, _Te)
        _Cij = np.append( _CEij[:], _CIik[:] )
    else:
        _Cij = _CEij

    return _Cij
