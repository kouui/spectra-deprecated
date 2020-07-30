
import numpy as np

import sys
sys.path.append("../../")

from src.Atomic import Collision, PhotoIonize
from src.Atomic import LTELib, SEsolver, BasicP
from src.RadiativeTransfer import Profile
from src.Math import Integrate
from src import Constants as Cst

from scipy.interpolate import interp1d

def ni_nj_LTE(_Level, _Line, _Cont, _Te, _Ne):
    r"""
    """
    _n_LTE = LTELib.get_LTE_ratio(_erg=_Level.erg[:], _g=_Level.g[:],
                              _stage=_Level.stage[:], _Te=_Te, _Ne=_Ne)
    _nLine = _Line.shape[0]
    _nCont = _Cont.shape[0]
    _nTran = _nLine + _nCont

    _ni_LTE = np.empty(_nTran, dtype=np.double)
    _nj_LTE = np.empty(_nTran, dtype=np.double)

    for k in range(_nLine):
        _ni_LTE[k] = _n_LTE[_Line.idxI[k]]
        _nj_LTE[k] = _n_LTE[_Line.idxJ[k]]
    for k in range(_nCont):
        _ni_LTE[k+_nLine] = _n_LTE[_Cont.idxI[k]]
        _nj_LTE[k+_nLine] = _n_LTE[_Cont.idxJ[k]]

    return _n_LTE, _ni_LTE, _nj_LTE

def bf_R_rate(_waveMesh, _Jnu, _alpha, _Te, _ni_LTE, _nj_LTE):
    r"""
    """
    _nCont = len(_waveMesh)

    _Rik = np.zeros(_nCont,dtype=np.double)
    _Rki_stim = np.zeros(_nCont,dtype=np.double)
    _Rki_spon = np.zeros(_nCont,dtype=np.double)

    for k in range(_nCont):

        _res = PhotoIonize.bound_free_radiative_transition_coefficient(
                            wave = _waveMesh[k][::-1],
                            J = _Jnu[k][::-1],
                            alpha = _alpha[k][::-1],
                            Te = _Te,
                            ni_lte = _ni_LTE[k],
                            nk_lte = _nj_LTE[k])
        _Rik[k] = _res[0]
        _Rki_stim[k] = _res[1]
        _Rki_spon[k] = _res[2]

    return _Rik, _Rki_stim, _Rki_spon

def B_Jbar(_Level, _Line, _lineIndex, _MeshRadLine=None, _Te=None, _Vt=None,_Vd=None,
                _Mass=None, _I_cm=None, _Jbar=None):
    r"""
    """
    _nRadiativeLine = _lineIndex.shape[0]
    _Bij_Jbar = np.zeros(_nRadiativeLine, np.double)
    _Bji_Jbar = np.zeros(_nRadiativeLine, np.double)

    for k in range(_nRadiativeLine):
        _line_index = _lineIndex[k]
        _w0 = _Line.w0[ _line_index ]
        _f0 = _Line.f0[ _line_index ]

        _gi = _Level.g[ _Line.idxI[_line_index] ]
        _gj = _Level.g[ _Line.idxJ[_line_index] ]
        _Aji = _Line.AJI[_line_index]
        _Bji, _Bij = LTELib.EinsteinA_to_EinsteinBs_cm(_Aji, _w0, _gi, _gj)


        if _Jbar is not None:
            _Jbar0 = _Jbar[k]

        else:
            _dopWidth_cm = BasicP.get_Doppler_width(p0=_w0, Te=_Te, Vt=_Vt, am=_Mass)
            _a = BasicP.get_damping_a(_Gamma=_Line.Gamma[_line_index], _dopWidth_hz=_dopWidth_cm*_f0/_w0 )

            # shift wavelength mesh
            _wave_mesh = _MeshRadLine[k][:]
            _wave_mesh_shifted = _wave_mesh - (_w0*_Vd/Cst.c_)/_dopWidth_cm
            _wave_mesh_cm = _wave_mesh_shifted[:] * _dopWidth_cm
            _absorb_mesh_cm = Profile.Voigt(_a, _wave_mesh[:]) / _dopWidth_cm

            _radLine = _I_cm[k]
            _fill_value = (_radLine[1,0],_radLine[1,-1])
            _f = interp1d(x=_radLine[0,:], y=_radLine[1,:],
                          bounds_error=False, fill_value=_fill_value)
            _I_cm_interp = _f(_wave_mesh_cm)

            _integrand = 0.5 * _I_cm_interp[:] * _absorb_mesh_cm
            _Jbar0 = Integrate.Trapze(integrand=_integrand, x=_wave_mesh_cm)

        _Bji_Jbar[k] = _Bji * _Jbar0
        _Bij_Jbar[k] = _Bij * _Jbar0

    return _Bij_Jbar, _Bji_Jbar

def CEij_rate_coe(_Omega_table, _Te_table, _Coe, _Te):
    r"""
    """
    _omega = Collision.interpolate_CE_fac(_table=_Omega_table[:,:],
                                        _Te=_Te,
                                        _Te_table=_Te_table[:],
                                        _f1=_Coe.f1[:], _f2=_Coe.f2[:])
    _CEij = Collision.get_CE_rate_coe(_CE_fac=_omega, _Te=_Te,
                                      _gi=_Coe.gi[:],_dEij=_Coe.dEij[:])

    return _CEij

def CIik_rate_coe(_Omega_table, _Te_table, _Coe, _Te):
    r"""
    """
    _omega = Collision.interpolate_CI_fac(_table=_Omega_table[:,:],
                                       _Te=_Te,
                                       _Te_table=_Te_table[:],
                                       _f2=_Coe.f2[:])
    _CIik = Collision.get_CI_rate_coe(_CI_fac=_omega[:], _Te=_Te,
                                  _dEik=_Coe.dEij[:])

    return _CIik
