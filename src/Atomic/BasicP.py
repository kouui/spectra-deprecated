################################################################################
# this file defines functions for
#     calculations related to naive/basic physics process
################################################################################

from .. import Constants as Cst
from ..Config import isJIT_


import numba as nb

################################################################################
# wavelength [cm] <--> frequency [hz]
################################################################################

def wave_to_freq(wave):
    r"""
    convert Wavelength to Frequency.

    Parameters
    ----------

    wave : np.double or array-like
        Wavelength, [:math:`cm`]

    Returns
    -------

    freq : np.double or array-like
        Frequency, [:math:`Hz`]

    Notes
    -----

    .. math:: \nu = \frac{c}{\lambda}
    """
    freq = Cst.c_ / wave
    return freq

def freq_to_wave(freq):
    r"""
    convert Frequency to Wavelength.

    Parameters
    ----------

    freq : np.double or array-like
        Frequency, [:math:`Hz`]

    Returns
    -------

    wave : np.double or array-like
        Wavelength, [:math:`cm`]

    Notes
    -----

    .. math:: \lambda = \frac{c}{\nu}
    """
    wave = Cst.c_ / freq
    return wave

################################################################################
# Doppler shift and Doppler width
################################################################################

def Dvlocity_to_Dshift(p0, v):
    r"""
    given Doppler velocity and the line central wavelength/frequency,
    compute Doppler shift in wavelength/frequency.

    Parameters
    ----------

    p0 : np.double or array-like
        Frequency in any frequency unit or Wavelength in any length unit
    v : np.double or array-like
        line of sight velocity, [:math:`cm/s`]

    Returns
    -------

    dp : np.double or array-like
        Frequency or Wavelength, same unit with input `p0`

    Notes
    -----

    in wavelength,

    .. math:: d\lambda = \lambda_0 \frac{v}{c}

    in frequency,

    .. math:: d\nu = \nu_0 \frac{v}{c}
    """
    dp = p0 * v / Cst.c_
    return dp

def get_Doppler_width(p0, Te, Vt, am):
    r"""
    Given central wavelength/frequency, relative atomic mass of a line,
    and the temperature, turbulent velocity, compute the corresponding
    Doppler Width.

    Parameters
    ----------

    p0 : np.double or array-like
        Frequency in any frequency unit or Wavelength in any length unit
    Te : np.double or array-like
        Temperature, [:math:`K`]
    Vt : np.double or array-like
        Turbulent velocity, [:math:`cm/s`]
    am : np.double or array-like
        atomic mass relative to hydrogen atom. [-]

    Returns
    -------

    dp : np.double or array-like
        Frequency/Wavelength Doppler width, same unit with input argument `p0`

    Notes
    -----

    According to [1]_.

    .. math:: \eta_{0} = (2kT/m + Vt^2)^{1/2}

    in wavelength,

    .. math:: d\lambda = \lambda_0 \frac{\eta_{0}}{c}

    in frequency,

    .. math:: d\nu = \nu_0 \frac{\eta_{0}}{c}

    References
    -----------

    .. [1] Robert J. Rutten, "Radiative Transfer in Stellar Atmosphere", pp. 82, 2003.
    """
    eta0_ = (2.*Cst.k_*Te/(Cst.mH_*am) + Vt*Vt)**(0.5)
    dp = p0 * eta0_ / Cst.c_
    return dp

def get_Level_gamma(Aji,idxJ,gamma):
    r"""
    Given Einstein A coefficient of Levels, upper index of Lines,
    compute radiative damping constant for each Level.

    Parameters
    ----------

    Aji : array-like, dtype of np.double
        Einstein A coefficient of Lines, [:math:`s^{-1}`]
    idxJ : array-like, np.uint8 or anyother integer types.
        index of upper level of Lines.
    gamma : array-like, dtype of np.double
        radiative damping constant of each Level, [:math:`s^{-1}`].
        An array to store output result.

    Notes
    -----

    Here only spantaneous radiative de-excitation is included [1]_.

    .. math:: \gamma_j^{rad} = \sum_{i<j}A_{ji}

    To be more accurate, we should also include radiative de-excitation/excitation [2]_.

    .. math:: \gamma_j^{rad} = \sum_{i<j}(A_{ji}+B_{ji}\overline{J}_{ji}) + \sum_{k>j}B_{jk}\overline{J}_{jk}

    Although this will complicate calculation since radiation is update every iteration
    and has spatial dependency in optically thick atmosphere.

    References
    -----------

    .. [1] Robert J. Rutten, "Radiative Transfer in Stellar Atmosphere", pp. 54, 2003.
    .. [2] Ivan Hubeny, Dimitri Mihalas, "Theory of Stellar Atmosphere:
        An Introduction to Astrophysical Non-equilibrium
        Quantitative Spectroscopic Analysis",
        Princeton University Press, pp. 230, 2015.
    """
    gamma[:] = 0
    for i in range(Aji.shape[0]):
        gamma[idxJ[i]] += Aji[i]

def get_Line_Gamma(idxI, idxJ, gamma, Gamma):
    r"""
    Given Einstein radiative damping constant of Levels,
    compute radiative damping constant of Lines.

    Parameters
    ----------

    idxI : array-like, np.uint8 or anyother integer types.
        index of lower level of Lines.
    idxJ : array-like, np.uint8 or anyother integer types.
        index of upper level of Lines.
    gamma : array-like, dtype of np.double
        radiative damping constant of each Level, [:math:`s^{-1}`].
    Gamma : array-like, dtype of np.double
        radiative damping constant of each Lines, [:math:`s^{-1}`].
        An array to store output result.

    Notes
    -----

    Damping constant of a Line is the sum of damping constants of Lower and Upper Levels [1]_.

    .. math:: \Gamma^{rad} = \gamma_i^{rad} + \gamma_j^{rad}

    References
    -----------

    .. [1] Robert J. Rutten, "Radiative Transfer in Stellar Atmosphere", pp. 54, 2003.
    """
    Gamma[:] = 0
    for i in range(Gamma.shape[0]):
        Gamma[i] = gamma[idxI[i]] + gamma[idxJ[i]]

def get_damping_a(_Gamma, _dopWidth_hz):
    r"""
    Given the radiative damping constant and
    the Doppler Width (in frequency unit) of the line,
    compute damping constant a.

    Parameters
    ----------

    _Gamma : np.double,
        radiative damping constant of each Lines, [:math:`s^{-1}`].

    _dopWidth_hz : np.double,
        Doppler Width in frequency unit, [:math:`s^{-1}`].

    Returns
    -------

    _a : np.double,
        damping constant, [-]

    Notes
    -----

    From [1]_.

    .. math::
        a = \frac{\Gamma}{4 \pi \Delta\nu_{D}}

    References
    -----------

    .. [1] Ivan Hubeny, Dimitri Mihalas, "Theory of Stellar Atmosphere:
        An Introduction to Astrophysical Non-equilibrium
        Quantitative Spectroscopic Analysis",
        Princeton University Press, pp. 232, 2015.

    """
    _a = _Gamma / ( 4 * Cst.pi_ * _dopWidth_hz )
    return _a

def refractive_index_in_air(w_, unit_):
    r"""
    J. Opt. Soc. Am. 62, 958 (1972)
    """
    if unit_ == "nm":
        _f = 1E-3
    elif unit_ == "AA":
        _f = 1E-4
    else:
        _f = 1
    sigma_ = w_ * _f # unit --> um
    out_ = 8342.13 + 2406030 / (130-_sigma_*_sigma_) + 15997 / (38.9-_sigma_*_sigma_)
    n_ = out_ * 1E-8 + 1
    return n_

def air_to_vacuum(w_, unit_):
    r"""
    https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html#AIR
    """
    assert unit_ in ("nm","um","AA")

    n_ = refractive_index_in_air(w_, unit_)
    return w_ * n_

#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------

if isJIT_ == True :
    get_Doppler_width = nb.vectorize( ['float64(float64,float64,float64,float64)'] )( get_Doppler_width )
    get_damping_a = nb.vectorize( ['float64(float64,float64)'] )( get_damping_a )
    get_Level_gamma = nb.njit( ['(uint8[:],uint8[:],float64[:])'] )( get_Level_gamma )
    get_Line_Gamma = nb.njit( ['(uint8[:],uint8[:],float64[:],float64[:])'] )( get_Line_Gamma )
