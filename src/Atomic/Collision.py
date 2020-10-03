import numpy as np
from .. import Constants as Cst

#from scipy.interpolate import splrep, splev

def interpolate_CE_fac(_table, _Te, _Te_table, _f1, _f2):
    r"""
    given temperature, interpolate collisional excitation coefficient

    Parameters
    ----------

    _table : np.double, array, (nLine, nTemperature)
        a table of CE coefficient as a function of temperature for interpolation

    _Te : scalar
        termperature, [:math:`K`]

    _Te_table : np.double, array, (nTemperature,)
        corresponding termperature points in _table, [:math:`K`]

    _f1 : int, array
        a factor needed to compute CE rate coefficient

    _f2 : int, array
        a factor needed to compute CE rate coefficient

    Returns
    -------

    _CE_fac : np.double, np.array, (nLine,)
        the CE coefficient we need to compute CE rate coefficient

    Notes
    -----

    From [1]_.

    .. math:: n_{e} C_{ij} = n_{e} \frac{8.63e-6 \times (\Omega_{ij} ) }{g_{i} T_{e}^{1/2}}  \exp{\frac{-dE_{ji}}{kT_{e}} } \quad [s^{-1}]

    .. math:: \Omega_{ij} = \Omega_{ij} f1 / f2

    References
    ----------

    .. [1] John T. Mariska, "The Solar Transition Region",
        Cambridge University Press, pp. 22, 1992
    """

    _nLine = _table.shape[0]
    _CE_fac = np.zeros(_nLine, dtype=np.double)
    for k in range(_nLine):
        #--- scipy B-spline interpolation
        #_Bsp_obj = splrep(x=_Te_table[:], y=_table[k,:])
        # ext=3 : return boundary value
        #_CE_fac[k] = splev(_Te, _Bsp_obj, ext=3) * _f1[k] / _f2[k]

        _CE_fac[k] = np.interp( _Te, _Te_table[:], _table[k,:] )  * _f1[k] / _f2[k]

    return _CE_fac

def interpolate_CI_fac(_table, _Te, _Te_table, _f2):
    r"""
    given temperature, interpolate collisional ionization coefficient

    Parameters
    ----------

    _table : np.double, array, (nLine, nTemperature)
        a table of CI coefficient as a function of temperature for interpolation

    _Te : scalar
        termperature, [:math:`K`]

    _Te_table : np.double, array, (nTemperature,)
        corresponding termperature points in _table, [:math:`K`]

    _f2 : int, array
        a factor needed to compute CI rate coefficient

    Returns
    -------

    _CI_fac : np.double, np.array, (nLine,)
        the CI coefficient we need to compute CI rate coefficient

    Notes
    -----

    .. math:: n_{e} C_{ik} = n_{e} * (\Omega_{ik}) * \exp{\frac{-dE}{kT_{e}}} * \sqrt{T_{e}}, \quad [s^{-1}]

    .. math:: \Omega_{ik} = \Omega_{ik} / f2

    References
    ----------

    """

    _nCont = _table.shape[0]
    _CI_fac = np.zeros(_nCont, dtype=np.double)
    for k in range(_nCont):
        #--- scipy B-spline interpolation
        #_Bsp_obj = splrep(x=_Te_table[:], y=_table[k,:])
        # ext=3 : return boundary value
        #_CI_fac[k] = splev(_Te, _Bsp_obj, ext=3) / _f2[k]

        _CI_fac[k] = np.interp( _Te, _Te_table[:], _table[k,:] )  / _f2[k]

    return _CI_fac

def Cij_to_Cji(_Cij,  _nj_by_ni_LTE):
    r"""
    calculate Cji from Cij

    Parameters
    -----------

    _Cij : np.double, np.array, (nLine); scalar
        collisional upward rate coefficient, [:math:`cm^{-3} s^{-1}`]

    _nj_by_ni_LTE : np.double, (nLine); scalar
        population ratio in LTE, nj/ni

    Returns
    -------
    _Cji : np.double, np.array, (nLine); scalar
        collisional downward rate coefficient, [:math:`cm^{-3} s^{-1}`]

    Notes
    -----

    .. math: n_j^{LTE} C_{ji} = n_i^{LTE} C_{ij}

    """

    _Cji = _Cij / _nj_by_ni_LTE

    return _Cji

def get_CE_rate_coe(_CE_fac, _Te, _gi, _dEij):
    r"""
    compute the CE rate coefficient.

    Parameters
    ----------

    _CE_fac : np.double, np.array, (nLine,); scalar
        the coefficient we interpolate from data

    _Te : scalar
        termperature, [:math:`K`]

    _gi : np.uint8, np.array, (nLine,); scalar
        statistical weight of lower level of the transition, [-]

    _dEij : np.double, np.array, (nLine); scalar
        excitation energy, [:math:`erg`]

    Returns
    -------

    _CEij : np.double, np.array, (nLine); scalar
        collisional excitation rate coefficient, [:math:`cm^{3} s^{-1}`]

    Notes
    -----

    For "ECS" (Effective Collisional Strength) [1]_.

    .. math:: n_{e} C_{ij} = n_{e} \frac{8.63e-6 \times (\Omega_{ij} f1 / f2) }{g_{i} T_{e}^{1/2}}  \exp{\frac{-dE_{ji}}{kT_{e}} } \quad [s^{-1}]

    References
    ----------

    .. [1] John T. Mariska, "The Solar Transition Region",
        Cambridge University Press, pp. 22, 1992
    """
    #Cst.pi_ * Cst.a0_**2 * (8*Cst.k_/Cst.pi_/Cst.me_)**0.5 * Cst.E_Rydberg_ / Cst.k_
    #=8.629140599482176e-06 [in CGS unit]

    _kT = Cst.k_ * _Te
    _CEij = (8.63E-06 * _CE_fac) / (_gi * _Te**0.5) * np.exp( - _dEij / _kT )

    return _CEij

def get_CI_rate_coe(_CI_fac, _Te, _dEik):
    r"""
    compute the CI rate coefficient.

    Parameters
    ----------

    _CI_fac : np.double, np.array, (nLine,); scalar
        the coefficient we interpolate from data

    _Te : scalar
        termperature, [:math:`K`]

    _dEik : np.double, np.array, (nLine); scalar
        ionization energy, [:math:`erg`]

    Returns
    -------

    _CIij : np.double, np.array, (nLine); scalar
        collisional ionization rate coefficient, [:math:`cm^{3} s^{-1}`]

    Notes
    -----

    .. math:: n_{e} C_{ik} = n_{e} * (\Omega_{ik} / f2) * \exp{\frac{-dE}{kT_{e}}} * \sqrt{T_{e}}, \quad [s^{-1}]

    References
    ----------
    """
    _kT = Cst.k_ * _Te
    _CIij = _CI_fac * _Te**0.5 * np.exp( - _dEik / _kT )

    return _CIij
