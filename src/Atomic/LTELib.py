################################################################################
# this file defines functions for
#     calculations related to LTE process
# Modification History:
#    2020.2.23   k.i.   Ufunc
################################################################################

import numpy as np
import numba as nb
from .. import Constants as Cst
from .. import Config

def Boltzmann_distribution(_gi, _gj, _Eji, _Te):
    r"""

    calculate the population ratio between upper level j and lower level i under LTE.

    with `nb.vectorize( [nb.float64(nb.uint8, nb.uint8, nb.float64, nb.float64)])`.

    Parameters
    ----------

    _gi  : np.uint8 or array-like
        statistical weight of lower level i, [-]
    _gj  : np.uint8 or array-like
        statistical weight of upper level j, [-]
    _Eji : np.double or array-like
        the gap of level energy between upper level j and lower level i,
        :math:`E_{ji}=E_j-E_i, \quad [erg]`
    _Te  : np.double or array-like
        electron temperature, [:math:`K`]

    Returns
    -------

    _rt : np.double or array-like
        population ratio of upper level j and lower level i.
        :math:`rt = n_j / n_i`, [-]

    Notes
    -----

    The population ratio according to Boltzmann distribution [1]_.

    .. math:: \frac{n_j}{n_i} = \frac{g_j}{g_i} e^{-(E_j-E_i)/{kT}}

    References
    ----------

    .. [1] Ivan Hubeny, Dimitri Mihalas, "Theory of Stellar Atmosphere:
        An Introduction to Astrophysical Non-equilibrium
        Quantitative Spectroscopic Analysis",
        Princeton University Press, pp. 262, 2015.
    """

    _kT = Cst.k_ * _Te
    _rt = _gj / _gi * np.exp(-_Eji/_kT)

    return _rt

def Saha_distribution(_gi, _gk, _chi, _ne, _Te):
    r"""

    calculate the population ratio between the ground states of two subsequent ionization stage under LTE.

    with `nb.vectorize( [nb.float64(nb.uint8,nb.uint8,nb.float64,nb.float64,nb.float64)])`

    Parameters
    ----------

    _gk: np.uint8 or array-like
        statistical weight of the ground state in ionization stage I+1, [-]
    _gi: np.uint8 or array-like
        statistical weight of the ground state in ionization stage I, [-]
    _chi: np.double or array-like
        ionization energy from ground state in ionization stage I
        to ground state in ionization stage I, [:math:`erg`]
    _ne: np.double or array-like
        electron density, [:math:`cm^{-3}`]
    _Te: np.double or array-like
        electron temperature, [:math:`K`]

    Returns
    -------
    _rt: np.double or array-like
        :math:`n_k / n_i`, :math:`n_k` the population of the ground state of ionization stage I+1
        and :math:`n_i` the population of the ground state of ionization stage I. [-]

    Notes
    -----

    The population ratio according to Saha's equation [1]_.

    a factor contains physics constants only,

    .. math:: f = 2(\frac{2 \pi m_e k}{h^2})^{3/2}

    then the population ratio is,

    .. math:: \frac{n_{0,k}}{n_{0,i}} = f \cdot \frac{g_{0,k}}{g_{0,i}} e^{\frac{-\chi_{ki}}{kT}}  T^{3/2} / n_e

    References
    ----------

    .. [1] Ivan Hubeny, Dimitri Mihalas, "Theory of Stellar Atmosphere:
        An Introduction to Astrophysical Non-equilibrium
        Quantitative Spectroscopic Analysis",
        Princeton University Press, pp. 94, 2015.
    """

    _kT = Cst.k_ * _Te
    _rt = Cst.saha_ * _Te**(1.5) * _gk / _gi * np.exp(-_chi/_kT) / _ne

    return _rt


def LTE_ratio_Line(_g, _idxI, _idxJ, _w0, _Te):
    r"""
    Compute LTE population ratio nj/ni for each line transition

    Parameters
    ----------
    _g : numpy.1darray of np.uint8
        statistical weight, [-]

    _idxI : numpy.1darray of np.uint16
        lower level index for each line transition

    _idxJ : numpy.1darray of np.uint16
        upper level index for each line transition

    _w0 : numpy.1darray of np.float64
        wavelength of each line transition, [:math:`cm`]

    _Te : np.double
        electron temperature, [:math:`K`]

    Returns
    --------
    _nRatio : numpy.1darray of np.double
        population ratio. [-]

    """
    _nLine = _w0.size
    _nRatio = np.ones(_nLine, dtype=np.double)
    for k in range(_nLine):
        _gi = _g[ _idxI[k] ]
        _gj = _g[ _idxJ[k] ]
        _chi = Cst.h_ * Cst.c_ / _w0[k]
        _nRatio[k] = Boltzmann_distribution(_gi, _gj, _chi, _Te)

    return _nRatio

def LTE_ratio_Cont(_g, _idxI, _idxJ, _w0, _Te, _Ne):
    r"""
    Compute LTE population ratio nj/ni for each continuum transition

    Parameters
    ----------
    _g : numpy.1darray of np.uint8
        statistical weight, [-]

    _idxI : numpy.1darray of np.uint16
        lower level index for each continuum transition

    _idxJ : numpy.1darray of np.uint16
        upper level index for each continuum transition

    _w0 : numpy.1darray of np.float64
        wavelength of each continuum transition, [:math:`cm`]

    _Te : np.double
        electron temperature, [:math:`K`]

    _Ne : np.double
        electron density, [:math:`cm^{-3}`]

    Returns
    --------
    _nRatio : numpy.1darray of np.double
        population ratio. [-]

    """
    _nLine = _w0.size
    _nRatio = np.ones(_nLine, dtype=np.double)
    for k in range(_nLine):
        _gi = _g[ _idxI[k] ]
        _gk = _g[ _idxJ[k] ]
        _chi = Cst.h_ * Cst.c_ / _w0[k]
        _nRatio[k] = Saha_distribution(_gi, _gk, _chi, _Ne, _Te)

    return _nRatio


def get_LTE_ratio(_erg, _g, _stage, _Te, _Ne):
    r"""
    Compute LTE population ratio relative to 1st level.

    Parameters
    ----------

    _erg : numpy.1darray of np.double
        level energy relative to 1st level, [:math:`erg`]

    _weight : numpy.1darray of np.uint8
        statistical weight, [-]

    _stage : numpy.1darray of np.uint8
        ionization stage, [-]

    _Te : np.double
        electron temperature, [:math:`K`]

    _Ne : np.double
        electron density, [:math:`cm^{-3}`]

    Returns
    --------
    _nRatio : numpy.1darray of np.double
        population ratio. [-]

    Notes
    -----
    We set the population of the 1st level to `1.0`, and then compute the
    population ratio level by level. `Saha_distribution` is applied whenever
    there is a jump of ionization stage.
    Finnaly, we normalize the _nRatio array with respec to the total population.

    """
    _nLevel = _erg.size
    _nRatio = np.empty(_nLevel, np.double)
    _nRatio[0] = 1.
    #kT = Cst.k_*Te
    for i in range(1, _nLevel):
        _gj, _gi = _g[i], _g[i-1]
        if _stage[i] - _stage[i-1] == 0:
            #nRatio[i] = nRatio[i-1] * gj/gi * np.exp(-(erg[i]-erg[i-1])/kT)
            _nRatio[i] = _nRatio[i-1] * Boltzmann_distribution(_gi, _gj, _erg[i]-_erg[i-1], _Te)
        elif _stage[i] - _stage[i-1] == 1:
            #nRatio[i] = nRatio[i-1] * gj/gi * np.exp(-(erg[i]-erg[i-1])/kT) * Cst.saha_ * Te**(1.5) / Ne
            _nRatio[i] = _nRatio[i-1] * Saha_distribution(_gi, _gj, _erg[i]-_erg[i-1], _Ne, _Te)

        #print(f"erg[i] : {_erg[i]}, erg[i-1] : {_erg[i-1]}, Ne : {_Ne}, Te : {_Te}")

        _nRatio[:] /= _nRatio[:].sum()

    return _nRatio


def EinsteinA_to_EinsteinBs_hz(Aji, f0, gi, gj):
    r"""

    given Einstein A coefficient Aij,
    calculate Einstein B coefficients Bij and Bji.

    Parameters
    ----------

    Aji : np.double or array-like
        Einstein A coefficient Aji, [:math:`s^{-1}`]
    f0 : np.double or array-like
        central frequency of corresponding line transition, [:math:`Hz`]
    gi : np.uint8 or array-like
        statistical weight of lower level, [-]
    gj : np.uint8 or array-like
        statistical weight of upper level, [-]

    Returns
    -------

    Bji : np.double or array-like
        Einstein B coefficient Bji, [:math:`s^{-1}/(erg/cm^{2}/Sr/Hz/s)`]
    Bij : np.double or array-like
        Einstein B coefficient Bji, [:math:`s^{-1}/(erg/cm^{2}/Sr/Hz/s)`]

    Notes
    -----

    Einstein Relations for bound-bound transitions [1]_.

    .. math:: B_{ji} = A_{ji} / \frac{2h\nu^{3}}{c^{2}}
    .. math:: B_{ij} = B_{ji} \frac{g_{j}}{g_{i}}

    References
    ----------

    .. [1] Ivan Hubeny, Dimitri Mihalas, "Theory of Stellar Atmosphere:
        An Introduction to Astrophysical Non-equilibrium
        Quantitative Spectroscopic Analysis",
        Princeton University Press, pp. 117, 2015.
    """
    factor_ = 2*Cst.h_*f0**3/Cst.c_**2
    Bji = Aji / factor_
    Bij = Bji * float(gj) / float(gi)

    return Bji, Bij


def EinsteinA_to_EinsteinBs_cm(Aji, w0, gi, gj):
    r"""

    given Einstein A coefficient Aij,
    calculate Einstein B coefficients Bij and Bji.

    Parameters
    ----------

    Aji : np.double or array-like
        Einstein A coefficient Aji, [:math:`s^{-1}`]
    w0 : np.double or array-like
        central wavelength of corresponding line transition, [:math:`cm`]
    gi : np.uint8 or array-like
        statistical weight of lower level, [-]
    gj : np.uint8 or array-like
        statistical weight of upper level, [-]

    Returns
    -------

    Bji : np.double or array-like
        Einstein B coefficient Bji, [:math:`s^{-1}/(erg/cm^{2}/Sr/Hz/s)`]
    Bij : np.double or array-like
        Einstein B coefficient Bji, [:math:`s^{-1}/(erg/cm^{2}/Sr/Hz/s)`]

    Notes
    -----

    Einstein Relations for bound-bound transitions.
    Since we are in wavelength unit, the relation between
    [:math:`A_{ji}`] and [:math:`B_{ji}`] should be evaluated using
    Planck function also in wavelength unit

    .. math:: B_{ji} = A_{ji} / (2 h c^{2} \lambda^{5} )
    .. math:: B_{ij} = B_{ji} \frac{g_{j}}{g_{i}}

    """
    factor_ = 2*Cst.h_*Cst.c_**2/w0**5
    Bji = Aji / factor_
    Bij = Bji * gj / gi

    return Bji, Bij

def Aji_to_Bji_cm(Aji, w0):
    r""" """

    factor- = 2*Cst.h_*Cst.c_**2/w0**5
    Bji = Aji / factor_

    return Bji

def Bji_to_Bij(Bji, gi, gj):
    r""" """

    Bij = Bji * gj / gi

    return Bij

def Planck_hz(F,T):
    r"""
    given frequency and temperature,
    calculate the frequency based planck function.

    with `nb.vectorize( [nb.float64(nb.float64,nb.float64)])`.

    Parameters
    ----------

    F : np.double or array-like
        frequency, [:math:`Hz`]
    T : np.double or array-like
        temperature, [:math:`K`]

    Returns
    -------

    intensity : np.double or array-like
        frequency based intensity, [:math:`erg/cm^2/Sr/Hz/s`]

    Notes
    -----

    The Planck function [1]_.

    .. math:: B_{\nu}(T) = \frac{2h\nu^3}{c^2} \frac{1}{e^{h\nu/kT}-1} \quad [erg/cm^2/Sr/Hz/s]

    References
    ----------

    .. [1] Ivan Hubeny, Dimitri Mihalas, "Theory of Stellar Atmosphere:
        An Introduction to Astrophysical Non-equilibrium
        Quantitative Spectroscopic Analysis",
        Princeton University Press, pp. 102, 2015.
    """
    F_T = F / T
    if F_T > 1.04183e+13:                                        # ignore exponential part, exponential > 500
        intensity = 0
    #elif F_T > 2.08366e+12:                                      # Wien approximation, exponential > 100
    #    intensity = 2.0*Cst.h_*F*F*F/Cst.c_/Cst.c_ * np.exp( -Cst.h_*F_T/(Cst.k_) )
    #elif F_T < 2.08366e+08:                                      # Rayleighâ€“Jeans approximation, exponential < 0.01
    #    intensity = 2.0*F*F/Cst.c_/Cst.c_ * Cst.k_ * T
    else:                                                        # normal formula
        intensity = 2.0*Cst.h_*F*F*F/Cst.c_/Cst.c_ / ( np.exp( Cst.h_*F_T/(Cst.k_) ) - 1.0 )

    return intensity


def Planck_cm(W,T):
    r"""
    given wavelength and temperature,
    calculate the wavelength based planck function.

    with `nb.vectorize( [nb.float64(nb.float64,nb.float64)])`.

    Parameters
    ----------

    W : np.double or array-like
        wavelength, [:math:`cm`]
    T : np.double or array-like
        temperature, [:math:`K`]

    Returns
    -------

    intensity : np.double or array-like
        wavelength based intensity, [:math:`erg/cm^2/Sr/cm/s`]

    Notes
    -----

    The Planck function [1]_.

    .. math:: B_{\lambda}(T) = \frac{2hc^2}{\lambda^5} \frac{1}{e^{hc/\lambda kT}-1} \quad [erg/cm^2/Sr/cm/s]

    References
    ----------

    .. [1] Robert J. Rutten, "Introduction to Astrophysical Radiative Transfer", pp. 43, 2015.
    """
    WT = W*T
    if WT < 2.87755e-03:                                         # ignore exponential part, exponential > 500
        intensity = 0
    #elif WT < 1.43878e-02:                                       # Wien approximation, exponential > 100
    #    intensity = 2.0*Cst.h_*Cst.c_*Cst.c_ /(W)**5 * np.exp( -Cst.h_*Cst.c_/(Cst.k_*WT) )
    #elif WT > 1.43878e+02:                                       # Rayleighâ€“Jeans approximation, exponential < 0.01
    #    intensity = 2.0*Cst.c_*Cst.k_*T / (W)**4
    else:                                                        # normal formula
        intensity = 2.0*Cst.h_*Cst.c_*Cst.c_ /(W)**5 /  ( np.exp( Cst.h_*Cst.c_/(Cst.k_*WT) ) - 1.0 )

    return intensity

################################################################################
def Ufunc(elm, T):
    r"""
    partition function of neutral hydrogen


    Parameters
    ----------

    elm : element & ionization stage as ca_i, fe_ii, etc.
    T   : temperature (k)

    Notes
    -----
    Wanring : this polynomial fitting formula(table) is only avaible for low temperature (:math:`<10000 K`)

    from [1]_, [2]_.
    .. math:: log(u) = c_0 + c_1 \cdot log(\theta) + c_2 \cdot log(\theta)^2 + c_3 \cdot log(\theta)^3 + c_4 \cdot log(\theta)^4
    .. math:: log = log_{10}
    .. math:: \theta = 5040 / T

    modification history:
    - 2006.5.23     k.i.
    - 2015.7.5      k.i.	'h_ii'
    - 2019.11.26    k.i.	'Ba' from Gary 2009 (use poly_ufunc.pro)
    - 2020.2.13     k.i.	from IDL ufunc_gray.pro

    References
    ----------

    .. [1] Gray 1992, "The observation and analysis of stellar photospheres", app.D
    .. [2] 2009  table
    """

    if   elm == 'H_I':
        c = [0.30103, 0., 0., 0., 0.]
    elif elm == 'H_II':
        c = [0., 0., 0., 0., 0.]
    elif elm == 'He_I':
        c = [0., 0., 0., 0., 0.]
    elif elm == 'He_II':
        c = [0.30103, 0., 0., 0., 0.]
    elif elm == 'Li_I':
        c = [0.31804, -0.20616, 0.91456, -1.66121, 1.04195]
    elif elm == 'Be_I':
        c = [0.00801, -0.17135, 0.62921, -0.58945, 0.]
    elif elm == 'Be_II':
        c = [0.30389, -0.00819, 0., 0., 0.]
    elif elm == 'B_i':
        c = [0.78028, -0.01622, 0., 0., 0.]
    elif elm == 'C_I':
        c = [0.96752, -0.09452, 0.08055, 0., 0.]
    elif elm == 'C_II':
        c = [0.77239, -0.02540, 0., 0., 0.]
    elif elm == 'N_I':
        c = [0.60683, -0.08674, 0.30565, -0.28114, 0.]
    elif elm == 'N_II':
        c = [0.94968, -0.06463, -0.1291, 0., 0.]
    elif elm == 'O_I':
        c = [0.05033, -0.05703, 0., 0., 0.]
    elif elm == 'O_II':
        c = [0.60405, -0.03025, 0.04525, 0., 0.]
    elif elm == 'F_I':
        c = [0.76284, -0.03582, -0.05619, 0., 0.]
    elif elm == 'Ne_I':
        c = [0., 0., 0., 0., 0.]  # ufunc = 1
    elif elm == 'Ne_II':
        c = [0.74847, -0.06562, -0.07088, 0., 0.]
    elif elm == 'Na_I':
        c = [0.30955, -0.17778,  1.10594, -2.42847, 1.70721]
    elif elm == 'Na_II':
        c = [0., 0., 0., 0., 0.]  # ufunc = 1
    elif elm == 'Na_III':
        c = [np.log10(6), 0., 0., 0., 0.]  # ufunc1 = 6.0   # Allen, 1976
    elif elm == 'Mg_I':
        c = [0.00556, -0.12840,  0.81506, -1.79635, 1.26292]
    elif elm == 'Mg_II':
        c = [0.30257, -0.00451,  0.,  0., 0.]
    elif elm == 'Mg_III':
        c = [0., 0., 0., 0., 0.]  # ufunc = 1
    elif elm == 'Al_I':
        c = [0.76786, -0.05207,  0.14713,  -0.21376, 0.]
    elif elm == 'Al_II':
        c = [0.00334, -0.00995,  0.,  0., 0.]
    elif elm == 'Si_I':
        c = [0.97896, -0.19208,  0.04753,  0., 0.]
    elif elm == 'Si_II':
        c = [0.75647, -0.05490,  -0.10126,  0., 0.]
    elif elm == 'P_I':
        c = [0.64618, -0.31132,  0.68633,  -0.47505, 0.]
    elif elm == 'P_II':
        c = [0.93588, -0.18848,  0.08921,  -0.22447, 0.]
    elif elm == 'S_I':
        c = [0.95254, -0.15166,  0.02340,  0., 0.]
    elif elm == 'S_II':
        c = [0.61971, -0.17465,  0.48283,  -0.39157, 0.]
    elif elm == 'Cl_I':
        c = [0.74465, -0.07389,  -0.06965,  0., 0.]
    elif elm == 'Cl_II':
        c = [0.92728, -0.15913,  -0.01983,  0., 0.]
    elif elm == 'K_I':
        c = [0.34419, -0.48157,  1.92563,  -3.17826, 1.83211]
    elif elm == 'Ca_I':
        c = [0.07460, -0.75759, 2.58494, -3.53170, -1.65240]
    elif elm == 'Ca_II':
        c = [0.34383, -0.41472, 1.01550, 0.31930, 0.]
    elif elm == 'Ca_III':
        c = [0., 0., 0., 0., 0.]  # ufunc = 1
    elif elm == 'Sc_I':
        c = [1.08209, -0.77814, 1.78504, -1.39179, 0.]
    elif elm == 'Sc_II':
        c = [1.35894, -0.51812, 0.15634, 0., 0.]
    elif elm == 'Ti_I':
        c = [1.47343, -0.97220, 1.47986, -0.93275, 0.]
    elif elm == 'Ti_II':
        c = [1.74561, -0.51230, 0.27621, 0., 0.]
    elif elm == 'Ti_III':
        c = [0., 0., 0., 0., 0.]  # ufunc = 1
    elif elm == 'V_I':
        c = [1.68359, -0.82055, 0.92361, -0.78342, 0.]
    elif elm == 'V_II':
        c = [1.64112, -0.74045, 0.49148, 0., 0.]
    elif elm == 'V_III':
        c = [np.log10(28), 0., 0., 0., 0.]  # ufunc = 28
    elif elm == 'Cr_I':
        c = [1.02332, -1.02540, 2.02181, -1.32723, 0.]
    elif elm == 'Cr_II':
        c = [0.85381, -0.71166, 2.18621, -0.97590, -2.72893]
    elif 'Cr_III':
        c = [np.log10(25), 0., 0., 0., 0.]  # ufunc = 25
    elif elm == 'Mn_I':
        c = [0.80810, -0.39108, 1.74756, -3.13517, 1.93514]
    elif elm == 'Mn_II':
        c = [0.88861, -0.36398, 1.39674, -1.86424, -2.32389]
    elif elm == 'Mn_III':
        c = [np.log10(6), 0., 0., 0., 0.]  # ufunc = 6
    elif elm == 'Fe_I':
        c = [1.44701, -0.67040, 1.01267, -0.81428, 0.]
    elif elm == 'Fe_II':
        c = [1.63506, -0.47118, 0.57918, -0.12293, 0.]
    elif elm == 'Fe_III':
        c = [np.log10(25), 0., 0., 0., 0.]  # ufunc = 25
    elif elm == 'Co_I':
        c = [1.52929, -0.71430, 0.37210, -0.23278, 0.]
    elif elm == 'Ni_I':
        c = [1.49063, -0.33662, 0.08553, -0.19277, 0.]
    elif elm == 'Ni_II':
        c = [1.03800, -0.69572, 0.53893, 0.28861, 0.]
    elif elm == 'Ni_III':
        c = [np.log10(21), 0., 0., 0., 0.]  # ufunc = 21
    elif elm == 'Ba_I':
        c = [4.83034, -10.8244, 10.0364, -4.34979, 0.719568]
    elif elm == 'Ba_II':
        c = [2.54797, -6.38871, 7.98813, -4.38907, 0.862666]
    elif elm == 'Ba_iii':
        c = [0., 0., 0., 0., 0.]  # ufunc = 1
    else:
        print(elm, ': partition function is not defined in "Ufunc/LTElib_ki"!')
        c = [0., 0., 0., 0., 0.]  # return 1

    th = 5040./T
    s = c[0]
    for i in range(1,5):
        s += c[i]*(np.log10(th))**i
    ufunc1 = 10.**s

    return ufunc1

################################################################################
# whether to compile them using numba's LLVM
################################################################################

if Config.isJIT == True :
    Boltzmann_distribution = nb.vectorize( ['float64(uint8,uint8,float64,float64)'], nopython=True)( Boltzmann_distribution )
    Saha_distribution = nb.vectorize( ['float64(uint8,uint8,float64,float64,float64)'], nopython=True)( Saha_distribution )
    Planck_cm = nb.vectorize( [nb.float64(nb.float64,nb.float64)], nopython=True)( Planck_cm )
    Planck_hz = nb.vectorize( [nb.float64(nb.float64,nb.float64)], nopython=True)( Planck_hz )
    Aji_to_Bji_cm = nb.vectorize( ['float64(float64,float64)'], nopython=True)( Aji_to_Bji_cm )
    Bji_to_Bij = nb.vectorize( ['float64(float64,uint8,uint8)','float64(float64,int64,int64)'], nopython=True)( Bji_to_Bij )
    #Ufunc = nb.jit(nb.float64(nb.types.unicode_type,nb.float64), nopython=True)( Ufunc )
    #Ufunc = nb.jit(nopython=True)( Ufunc )
