import numpy as np
from .. import Constants as Cst

from scipy.interpolate import splrep, splev

def interpolate_CE_fac(_table, _Te, _Te_table, _f1, _f2, _isCont):
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

    _isCont : boolean, array
        True : is a continuum transition; False : not a continuum transition

    Returns
    -------

    _CE_fac : np.double, np.array, (nLine,)
        the CE coefficient we need to compute CE rate coefficient

    Notes
    -----

    From [1]_.

    .. math:: n_{e} C_{ij} = n_{e} \frac{8.63e-6 \times (\Omega_{ij} f1 / f2) }{g_{i} T_{e}^{1/2}}  \exp{\frac{-dE_{ji}}{kT_{e}} } \quad [s^{-1}]

    References
    ----------

    .. [1] John T. Mariska, "The Solar Transition Region",
        Cambridge University Press, pp. 22, 1992
    """

    _nTran = _table.shape[0]
    _CE_fac = np.zeros(_nTran, dtype=np.double)
    for k in range(_nTran):
        if _isCont[k]:
            continue
        #--- scipy B-spline interpolation
        _Bsp_obj = splrep(x=_Te_table[:], y=_table[k,:])
        # ext=3 : return boundary value
        _CE_fac[k] = splev(_Te, _Bsp_obj, ext=3) * _f1[k] / _f2[k]

    return _CE_fac

def interpolate_CI_fac(_table, _Te, _Te_table, _f2, _lineIndex, _nTran):
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

    _lineIndex : int, array
        correspnding line index of each continuum transition

    _nTran : int
        total number of transitions including line transition and continuum transition

    Returns
    -------

    _CI_fac : np.double, np.array, (nLine,)
        the CI coefficient we need to compute CI rate coefficient

    Notes
    -----

    .. math:: n_{e} C_{ik} = n_{e} * (\Omega_{ik} / f2) * \exp{\frac{-dE}{kT_{e}}} * \sqrt{T_{e}}, \quad [s^{-1}]

    References
    ----------

    """

    _CI_fac = np.zeros(_nTran, dtype=np.double)
    for k in range(_table.shape[0]):
        _line_index = _lineIndex[k]
        #--- scipy B-spline interpolation
        _Bsp_obj = splrep(x=_Te_table[:], y=_table[k,:])
        # ext=3 : return boundary value
        _CI_fac[_line_index] = splev(_Te, _Bsp_obj, ext=3) / _f2[k]

    return _CI_fac

def Cij_to_Cji(_Cij,  _ni_LTE, _nj_LTE):
    r"""
    calculate Cji from Cij

    Parameters
    -----------

    _Cij : np.double, np.array, (nLine); scalar
        collisional upward rate coefficient, [:math:`cm^{-3} s^{-1}`]

    _ni_LTE : np.double, np.array, (nLine); scalar
        population in lower level, [:math:`cm^{-3}`]

    _nj_LTE : np.double, np.array, (nLine); scalar
        population in upper level, [:math:`cm^{-3}`]

    Returns
    -------
    _Cji : np.double, np.array, (nLine); scalar
        collisional downward rate coefficient, [:math:`cm^{-3} s^{-1}`]

    Notes
    -----

    .. math: n_j^{LTE} C_{ji} = n_i^{LTE} C_{ij}

    """

    _Cji = _Cij * _ni_LTE / _nj_LTE

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
