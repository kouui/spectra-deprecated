
import numpy as np
import numba as nb
from ...Atomic import Collision, PhotoIonize
from ...Atomic import LTELib, SEsolver, BasicP, SEsolver
from ...RadiativeTransfer import Profile
from ...Math import Integrate
from ... import Constants as Cst

from ...Structure.MyTypes import T_ATOM#, T_ATOM_NBTYPE
from ...Atomic import Hydrogen

#from scipy.interpolate import interp1d



def convert_ni_to_nj_by_ni(_ni, _idxI, _idxJ):
    r"""
    """
    #_nLevel = _ni.shape[0]
    _nLine = _idxI.shape[0]
    _nj_by_ni = np.zeros_like(_idxI)

    for _k in range(_nLine):
        _i = _idxI[_k]
        _j = _idxJ[_k]
        _nj_by_ni[_k] = _ni[_j] / _ni[_i]

    return _nj_by_ni

@nb.njit( ['float64[:](float64[:], uint16[:], uint16[:], bool[:])'] )
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


def convert_nj_by_ni_to_ni_v0(_nj_by_ni_L, _idxI_L, _idxJ_L, _stage, _hasContinuum=False,
                      _nj_by_ni_C=None, _idxI_C=None, _idxJ_C=None):
    r"""
    """
    _nLevel = _stage.shape[0]
    _nLine  = _nj_by_ni_L.shape[0]
    if _hasContinuum:
        _nCont  = _nj_by_ni_C.shape[0]


    #_stage_list = []
    #for k in range(_nLevel):
    #    if _stage[k] not in _stage_list:
    #        _stage_list.append( _stage[k] )
    _stage_unique = np.unique(_stage)

    _ni = np.ones(_nLevel, dtype=np.double)
    _refIdx = 0
    #for _s in _stage_list:
    for _s in _stage_unique:
        # process line transition
        for k in range(_nLine):

            if _idxI_L[k] != _refIdx:
                continue

            _si = _stage[_idxI_L[k]]
            _sj = _stage[_idxJ_L[k]]


            if _si==_s and _sj==_s:
                _ni[ _idxJ_L[k] ] = _ni[ _refIdx ] * _nj_by_ni_L[k]

        # process continuum transition
        if not _hasContinuum:
            break

        for k in range(_nCont):

            if _idxI_C[k] != _refIdx:
                continue

            _si = _stage[_idxI_C[k]]
            _sj = _stage[_idxJ_C[k]]


            if _si==_s and _sj==_s+1:
                _ni[ _idxJ_C[k] ] = _ni[ _refIdx ] * _nj_by_ni_C[k]
                _refIdx = _idxJ_C[k]

    return _ni / _ni.sum()

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

@nb.njit(['Tuple((float64[:],float64[:],float64[:]))(float64[:,:],float64[:,:],float64[:,:],float64,float64[:])'])
def bf_R_rate(_waveMesh, _Jnu, _alpha, _Te, _nj_by_ni_LTE):
    r"""
    """
    _nCont = len(_waveMesh)

    _Rik = np.zeros(_nCont,dtype=np.double)
    _Rki_stim = np.zeros(_nCont,dtype=np.double)
    _Rki_spon = np.zeros(_nCont,dtype=np.double)

    for k in range(_nCont):
        ## ! could not be optimized to Te-Ne-array
        ## ! need to loop over Te-Ne-array
        _res = PhotoIonize.bound_free_radiative_transition_coefficient(
                            wave = _waveMesh[k,::-1],
                            J = _Jnu[k,::-1],
                            alpha = _alpha[k,::-1],
                            Te = _Te,
                            nk_by_ni_LTE=_nj_by_ni_LTE[k])
        _Rik[k] = _res[0]
        _Rki_stim[k] = _res[1]
        _Rki_spon[k] = _res[2]

    return _Rik, _Rki_stim, _Rki_spon

#@nb.njit
def B_Jbar(_Level, _Line, _MeshCoe, _Tr, _Te=1.E4, _Vt=5.E5, _Vd=1.E6, _Mass=1.):
    r"""
    """

    _nLine = _Line['AJI'].shape[0]
    _Bij_Jbar = np.zeros(_nLine, np.double)
    _Bji_Jbar = np.zeros(_nLine, np.double)

    for k in range(_nLine):

        ## ignore lines with Aji < 1E-3
        if _Line['AJI'][ k ] < 1.E-3:
            continue

        _w0 = _Line['w0'][ k ]
        _gi = _Line['gi'][ k ]
        _gj = _Line['gj'][ k ]
        _Aji = _Line['AJI'][ k ]
        _Bji, _Bij = LTELib.EinsteinA_to_EinsteinBs_cm(_Aji, _w0, _gi, _gj)

        ## use radiation temperature to calculate mean intensity
        if _Tr is not None:
            _Jbar0 = LTELib.Planck_cm(_Line['w0'][ k ], _Tr)
        ## use background radiation to calculate mean intensity
        else:
            _f0 = _Line.f0[ k ]
            _dopWidth_cm = BasicP.get_Doppler_width(_w0, _Te, _Vt, _Mass)
            _a = BasicP.get_damping_a(_Line['Gamma'][ k ], _dopWidth_cm*_f0/_w0 )

            # calculate mesh
            if k in _MeshCoe.lineIndex[:]:
                _nLambda = _MeshCoe.nLambda[ k ]
                _qcore = _MeshCoe.qcore[ k ]
                _qwing = _MeshCoe.qwing[ k ]
            else:
                _nLambda = 21
                _qcore = 2.5
                _qwing = 10
            _wave_mesh = Profile.makeLineMesh_Full(_nLambda, _qcore, _qwing) # in Doppler width unit

            ## shift wavelength mesh
            _wave_mesh_shifted = _wave_mesh - (_w0*_Vd/Cst.c_)/_dopWidth_cm
            _wave_mesh_cm = _wave_mesh_shifted[:] * _dopWidth_cm
            _absorb_mesh_cm = Profile.Voigt(_a, _wave_mesh[:]) / _dopWidth_cm

            _I_cm_interp = _absorb_mesh_cm[:]#MAKE_INTERPOLATION_HERE
            _integrand = 0.5 * _I_cm_interp[:] * _absorb_mesh_cm[:]
            _Jbar0 = Integrate.Trapze(integrand=_integrand, x=_wave_mesh_cm)

        _Bji_Jbar[k] = _Bji * _Jbar0
        _Bij_Jbar[k] = _Bij * _Jbar0

    return _Bij_Jbar, _Bji_Jbar


def B_Jbar_Tr_v0(_Aji, _Bji, _Bij, _w0, _Tr):

    _nLine = _Aji.shape[0]
    _Bij_Jbar = np.zeros(_nLine, np.double)
    _Bji_Jbar = np.zeros(_nLine, np.double)

    for k in range(_nLine):

        ## ignore lines with Aji < 1E-3
        if _Aji[k] < 1.E-3:
            continue

        #_Bji, _Bij = LTELib.EinsteinA_to_EinsteinBs_cm(_Aji[k], _w0[k], _gi[k], _gj[k])

        ## use radiation temperature to calculate mean intensity
        _Jbar0 = LTELib.Planck_cm(_w0[ k ], _Tr)

        _Bji_Jbar[k] = _Bji[k] * _Jbar0
        _Bij_Jbar[k] = _Bij[k] * _Jbar0

    return _Bij_Jbar, _Bji_Jbar


def B_Jbar_Tr(_Aji, _Bji, _Bij, _w0, _Jbar):

    _nLine = _Aji.shape[0]
    _Bij_Jbar = np.zeros(_nLine, np.double)
    _Bji_Jbar = np.zeros(_nLine, np.double)

    for k in range(_nLine):

        ## ignore lines with Aji < 1E-3
        if _Aji[k] < 1.E-3:
            continue

        #_Bji, _Bij = LTELib.EinsteinA_to_EinsteinBs_cm(_Aji[k], _w0[k], _gi[k], _gj[k])

        ## use radiation temperature to calculate mean intensity
        _Jbar0 = LTELib.Planck_cm(_w0[ k ], _Tr)

        _Bji_Jbar[k] = _Bji[k] * _Jbar0
        _Bij_Jbar[k] = _Bij[k] * _Jbar0

    return _Bij_Jbar, _Bji_Jbar





def B_Jbar_v0(_Level, _Line, _lineIndex, _MeshRadLine=None, _Te=None, _Vt=None,_Vd=None,
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
            _dopWidth_cm = BasicP.get_Doppler_width(_w0, _Te, _Vt, _Mass)
            _a = BasicP.get_damping_a(_Line.Gamma[_line_index], _dopWidth_cm*_f0/_w0 )

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
    r""" """
    ## ! could/better to be optimized to Te-Ne-array
    _omega = Collision.interpolate_CE_fac(_table=_Omega_table[:,:],
                                        _Te=_Te,
                                        _Te_table=_Te_table[:],
                                        _f1=_Coe.f1[:], _f2=_Coe.f2[:])
    ## ! could/better to be optimized to Te-Ne-array
    _CEij = Collision.get_CE_rate_coe(_CE_fac=_omega, _Te=_Te,
                                      _gi=_Coe.gi[:],_dEij=_Coe.dEij[:])

    return _CEij


def CIik_rate_coe(_Omega_table, _Te_table, _Coe, _Te):
    r""" """
    ## ! could/better to be optimized to Te-Ne-array
    _omega = Collision.interpolate_CI_fac(_table=_Omega_table[:,:],
                                       _Te=_Te,
                                       _Te_table=_Te_table[:],
                                       _f2=_Coe.f2[:])
    ## ! could/better to be optimized to Te-Ne-array
    _CIik = Collision.get_CI_rate_coe(_CI_fac=_omega[:], _Te=_Te,
                                  _dEik=_Coe.dEij[:])

    return _CIik

###@nb.njit([ nb.float64[:](nb.float64,nb.uint8[:],nb.uint8[:],T_ATOM_NBTYPE) ])
##@nb.njit
##def CEij_rate_coe_calculate(_Te, _ni, _nj, _ATOM_TYPE):
##    r""" """
##
##    #if _ATOM_TYPE == T_ATOM.HYDROGEN:
##    #    _CEij = Hydrogen.CE_rate_coe(_ni, _nj, _Te)
##    #else:
##    #    assert False
##    _CEij = Hydrogen.CE_rate_coe(_ni, _nj, _Te)
##    return _CEij
##
###@nb.njit([nb.float64[:](nb.float64,nb.uint8[:],T_ATOM_NBTYPE)])
##@nb.njit
##def CIik_rate_coe_calculate(_Te, _ni, _ATOM_TYPE):
##    r""" """
##
##    #if _ATOM_TYPE == T_ATOM.HYDROGEN:
##    #    _CIik = Hydrogen.CI_rate_coe(_ni, _Te)
##    #else:
##    #    assert False
##    _CIik = Hydrogen.CI_rate_coe(_ni, _Te)
##    return _CIik


def solve_SE(_nLevel, _idxI, _idxJ, _Cji, _Cij, _Rji_spon, _Rji_stim, _Rij, _Ne):
    r""" """

    _Cmat = np.zeros((_nLevel, _nLevel), dtype=np.double)
    SEsolver.setMatrixC(_Cmat=_Cmat[:,:],
                        _Cji=_Cji[:],
                        _Cij=_Cij[:],
                        _idxI=_idxI, _idxJ=_idxJ, _Ne=_Ne)

    _Rmat = np.zeros((_nLevel, _nLevel), dtype=np.double)
    SEsolver.setMatrixR(_Rmat=_Rmat[:,:],
                        _Rji_spon=_Rji_spon[:],
                        _Rji_stim=_Rji_stim[:],
                        _Rij=_Rij[:],
                        _idxI=_idxI, _idxJ=_idxJ)

    _n_SE = SEsolver.solveSE(_Rmat=_Rmat[:,:], _Cmat=_Cmat[:,:])

    return _n_SE

if Cst.isJIT:
    convert_nj_by_ni_to_ni = nb.njit(convert_nj_by_ni_to_ni)
    nj_by_ni_To_ni = nb.njit( ['float64[:](float64[:], uint16[:], uint16[:], bool[:])'] )(nj_by_ni_To_ni)
    B_Jbar_Tr = nb.njit(B_Jbar_Tr)
