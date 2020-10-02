import numpy as np
import numba as nb

from .. import Constants as Cst
from ..Math import Special

@nb.vectorize([nb.float64(nb.int64, nb.int64)], nopython=True)
def Gaunt_factor_coe(i, ni):
    r"""
    coefficients to calculate usual Gaunt factor for bound-free transitions

    Parameters
    -----------
    i  : int, [:math:`-`]
        order of the coefficient, gi

    ni : int, [:math:`-`]
        principal quantum number of lower level

    Returns
    --------
    gi : float, [:math:`-`]
        Gaunt factor coefficient

    Notes
    ------
    refer to [1]_ Table 1.


    References
    ------------
    .. [1] L. C. Johnson, "Approximations for collisional and
           radiative transition rates in atomic hydrogen",
           Astrophysical Journal, vol. 174, p.227, May 1972.
           1972ApJ...174..227J
    """
    if ni == 1:
        gi = (1.1330, -0.4059, 0.07014)[i]
    elif ni == 2:
        gi = (1.0785, -0.2319, 0.02947)[i]
    else:
        gi = (
            0.9935 + ( 0.2328 - 0.1296 / ni ) / ni,
            - ( 0.6282 - ( 0.5598 - 0.5299 / ni ) / ni ) / ni,
            ( 0.3887 - ( 1.181 - 1.470 / ni ) / ni ) / ni / ni,
        )[i]

    return gi


@nb.vectorize( [nb.float64(nb.int64,nb.float64)] , nopython=True)
def Gaunt_factor(ni, x):
    r"""
    usual Gaunt factor for bound-free transitions

    Parameters
    -----------
    ni : int, [:math:`-`]
        principal quantum number of lower level

    x : float, [:math:`-`]
        ratio of the transition energy to the ionization energy of the lower level

    Returns
    --------
    g : float, [:math:`-`]
        Gaunt factor

    Notes
    ------
    refer to [1]_ Eq(4).


    References
    ------------
    .. [1] L. C. Johnson, "Approximations for collisional and
           radiative transition rates in atomic hydrogen",
           Astrophysical Journal, vol. 174, p.227, May 1972.
           1972ApJ...174..227J
    """
##    """
##    if ni == 1:
##        g0 = 1.1330
##        g1 = -0.4059
##        g2 = 0.07014
##    elif ni == 2:
##        g0 = 1.0785
##        g1 = -0.2319
##        g2 = 0.02947
##    else:
##        g0 = 0.9935 + ( 0.2328 - 0.1296 / ni ) / ni
##        g1 = - ( 0.6282 - ( 0.5598 - 0.5299 / ni ) / ni ) / ni
##        g2 = ( 0.3887 - ( 1.181 - 1.470 / ni ) / ni ) / ni / ni
##    """
    g0 = Gaunt_factor_coe(0, ni)
    g1 = Gaunt_factor_coe(1, ni)
    g2 = Gaunt_factor_coe(2, ni)
    # Gaunt factor
    g = g0 + ( g1 + g2 / x ) / x

    return g

@nb.vectorize( [nb.float64(nb.int64,nb.int64)] , nopython=True)
def absorption_oscillator_strength(ni, nj):
    r"""
    absorption oscillator strength given by (Bethe and Salpeter 1957)
    without the correction of Gaunt factor.

    Parameters
    -----------
    ni : int, [:math:`-`]
        principal quantum number of lower level

    nj : int, [:math:`-`]
        principal quantum number of upper level

    Returns
    --------
    fij : float, [:math:`-`]
        absorption oscillator strength

    Notes
    ------
    refer to [1]_ Eq(1).


    References
    ------------
    .. [1] L. C. Johnson, "Approximations for collisional and
           radiative transition rates in atomic hydrogen",
           Astrophysical Journal, vol. 174, p.227, May 1972.
           1972ApJ...174..227J
    """
    coef = 32. / (3. * Cst.sqrt3_ * Cst.pi_ )
    x = 1. - (ni / nj)**2

    fij = coef * ni / ( x * nj )**3

    return fij


@nb.vectorize( [nb.float64(nb.int64,nb.int64)] , nopython=True)
def Einstein_A_coefficient(ni, nj):
    r"""
    Einstein coefficient for spontaneous emission in the hydrogen atom

    Parameters
    ------------
    ni : int,  [:math:`-`]
        principal quantum number of lower level

    nj : int,  [:math:`-`]
        principal quantum number of upper level

    Returns
    ---------
    Aji : float, [:math:`s^{-1}`]
        Einstein Aji coefficient

    Notes
    ------
    refer to [1]_ page 117 Eq(5.7) and Eq(5.8), page 135 Eq(5.105)

    .. math:: A_{ji} = C_{1} \cdot \frac{g_{i}}{g_{j}} \lambda^{-2} f_{ij}

    where

    .. math:: C_{1} = \frac{8 \pi^{2} e^{2} }{m_{e} c} \sim 0.6670


    References
    -----------

    .. [1] Ivan Hubeny, Dimitri Mihalas, "Theory of Stellar Atmosphere:
        An Introduction to Astrophysical Non-equilibrium
        Quantitative Spectroscopic Analysis",
        Princeton University Press, 2015.

    """
    # Gaunt factor
    x = 1. - (ni / nj)**2
    g = Gaunt_factor(ni, x)

    # corrected absorption oscillator strength
    fij = absorption_oscillator_strength(ni, nj) * g

    # excitation energy
    Eij = Cst.E_Rydberg_ * (1./ni**2 - 1./nj**2)

    # wavelength
    w = Cst.h_ * Cst.c_ / Eij

    # constant factor
    C1 = 0.667025 # 8 * Cst.pi_**2 * Cst.e_**2 / Cst.me_ / Cst.c_

    Aji = (ni/nj)**2 * C1 / (w*w) * fij

    return Aji

@nb.vectorize( [nb.float64(nb.int64,nb.int64, nb.float64)] , nopython=True)
def CE_rate_coe(ni, nj, Te):
    r"""
    Collisional Excitation rate coefficient qij for the hydrogen atom

    Parameters
    ------------
    ni : int,  [:math:`-`]
        principal quantum number of lower level

    nj : int,  [:math:`-`]
        principal quantum number of upper level

    Te : float, [:math:`K`]
        electron temperature

    Returns
    ---------
    qij : float, [:math:`cm^{3}s^{-1}`]
        Colisional excitation rate coefficient

    Notes
    ------
    refer to [1]_ Eq(36) and [2] page 275 Eq(9.51), Eq(9.54).

    the :math:`S_{e}(n,n')` is the Collisional Excitation rate coefficient :math:`q_{ij}`,

    with

    .. math:: n_{i} C_{ij} = n_{i} n_{e} q_{ij}


    References
    -----------
    .. [1] L. C. Johnson, "Approximations for collisional and
           radiative transition rates in atomic hydrogen",
           Astrophysical Journal, vol. 174, p.227, May 1972.
           1972ApJ...174..227J

    .. [2] Ivan Hubeny, Dimitri Mihalas, "Theory of Stellar Atmosphere:
        An Introduction to Astrophysical Non-equilibrium
        Quantitative Spectroscopic Analysis",
        Princeton University Press, 2015.

    """

    kT = Cst.k_ * Te

    # Gaunt factor
    x = 1. - (ni / nj)**2
    x_m = 1. / x
    g = Gaunt_factor(ni, x)

    # corrected absorption oscillator strength
    fij = absorption_oscillator_strength(ni, nj) * g

    # Eq(11)
    Aij = 2. * ni**2 * x_m * fij

    # Eq(25, 26, 31, 32)
    if ni >= 2:
        ni_m_ = 1 / ni
        bi = ni_m_ * ( 4.0 + ni_m_ * ( -18.63 + ni_m_ * ( 36.24 - ni_m_ * 28.09 ) ) )
        ri = 1.94 * ni_m_**(1.57)
    else:
        bi = -0.603
        ri = 0.45

    # Eq(23)
    Bij = 4. * ni * (ni/nj)**3 * x_m**2 * ( 1. + x_m * ( 4/3 + x_m * bi ) )

    # excitation energy
    Eij = Cst.E_Rydberg_ * (1./ni**2 - 1./nj**2)

    # Eq(37)
    y = Eij / kT
    # Eq(30)
    rij = ri * x
    # Eq(38)
    z = rij + y

    term1 = Aij * ( ( 1./y + 0.5 ) * Special.E1(y) - ( 1./z + 0.5 ) * Special.E1(z) )
    term2 = ( Bij - Aij *  np.log( 2. * ni * ni * x_m ) ) * ( Special.E2(y) / y - Special.E2(z) / z )

    # Eq(36)
    Sij = Cst.C0_ * Te**(0.5) * 2 * ni * ni * x_m * y * y * (term1 + term2)
    qij = Sij

    return qij

@nb.vectorize( [nb.float64(nb.int64, nb.float64)] , nopython=True)
def CI_rate_coe(ni, Te):
    r"""
    Collisional ionization rate coefficient qik for the hydrogen atom

    Parameters
    ------------
    ni : int,  [:math:`-`]
        principal quantum number of lower level

    Te : float, [:math:`K`]
        electron temperature

    Returns
    ---------
    qik : float, [:math:`cm^{3}s^{-1}`]
        Colisional ionization rate coefficient

    Notes
    ------
    refer to [1]_ Eq(39) and [2] page 275 Eq(9.51), Eq(9.54).

    the :math:`S_{e}(n,n')` is the Collisional ionization rate coefficient :math:`q_{ik}`,

    with

    .. math:: n_{i} C_{ik} = n_{i} n_{e} q_{ik}


    References
    -----------
    .. [1] L. C. Johnson, "Approximations for collisional and
           radiative transition rates in atomic hydrogen",
           Astrophysical Journal, vol. 174, p.227, May 1972.
           1972ApJ...174..227J

    .. [2] Ivan Hubeny, Dimitri Mihalas, "Theory of Stellar Atmosphere:
        An Introduction to Astrophysical Non-equilibrium
        Quantitative Spectroscopic Analysis",
        Princeton University Press, 2015.

    """

    kT = Cst.k_ * Te

    # Eq(20)
    i_arr = np.array([0,1,2],dtype=np.int64)
    gi_arr = Gaunt_factor_coe(i_arr, ni)
    Ai = 32./(3.*Cst.sqrt3_*Cst.pi_) * ni * ( gi_arr[:] / (i_arr[:]+3) ).sum()

    # Eq(25, 26, 31, 32)
    if ni >= 2:
        ni_m_ = 1 / ni
        bi = ni_m_ * ( 4.0 + ni_m_ * ( -18.63 + ni_m_ * ( 36.24 - ni_m_ * 28.09 ) ) )
        ri = 1.94 * ni_m_**(1.57)
    else:
        bi = -0.603
        ri = 0.45

    # Eq(24)
    Bi = 2./3. * ni*ni * (5. + bi)

    # ionization energy
    Eik = Cst.E_Rydberg_ * (1./ni**2)

    # Eq(40)
    y = Eik / kT
    # Eq(41)
    z = ri + y

    E1_y = Special.E1(y)
    E1_z = Special.E1(z)
    E2_y = Special.E2(y)
    E2_z = Special.E2(z)

    # Eq(42)
    xi_y = Special.E0(y) - 2.*E1_y + E2_y
    xi_z = Special.E0(z) - 2.*E1_z + E2_z

    term1 = Ai * ( 1./y * E1_y - 1./z * E1_z )
    term2 = ( Bi - Ai*np.log(2*ni*ni) ) * ( xi_y - xi_z )
    # Eq(39)
    Sik = Cst.C0_ * Te**(0.5) * 2 * ni * ni * y * y * (term1 + term2)
    qik = Sik

    return qik

@nb.vectorize( [nb.float64(nb.int64, nb.float64)] , nopython=True)
def CI_rate_coe_clark(ni, Te):
    r"""
    Collisional ionization rate coefficient qik for the hydrogen atom

    Parameters
    ------------
    ni : int,  [:math:`-`]
        principal quantum number of lower level

    Te : float, [:math:`K`]
        electron temperature

    Returns
    ---------
    qik : float, [:math:`cm^{3}s^{-1}`]
        Colisional ionization rate coefficient

    Notes
    ------
    refer to [1]_ Eq(8) and [2] page 275 Eq(9.51), Eq(9.54).

    the :math:`C(y)` is the Collisional ionization rate coefficient :math:`q_{ik}`,

    with

    .. math:: n_{i} C_{ik} = n_{i} n_{e} q_{ik}


    References
    -----------
    .. [1] L. C. Johnson, "Integral and differential cross
           sections for electron impact ionization",
           Astrophysical Journal, vol. 381, 597-600, Nov 1972.
           1991ApJ...381...597C

    .. [2] Ivan Hubeny, Dimitri Mihalas, "Theory of Stellar Atmosphere:
        An Introduction to Astrophysical Non-equilibrium
        Quantitative Spectroscopic Analysis",
        Princeton University Press, 2015.

    """

    kT = Cst.k_ * Te

    C1 =  1.53690
    C2 =  0.99656
    C3 = -0.61916
    C4 =  2.44630
    C5 = -2.47730
    C6 =  3.21510
    C7 = -1.45120
    C8 =  1.72300
    C9 = -0.47075

    G = ( ni - 1 ) * ( 4 * ni + 1 ) / ( 6 * ni )

    # ionization energy
    Eik = Cst.E_Rydberg_ * (1./ni**2)

    y = Eik / kT
    TeV = Te * Cst.K2eV_
    F = 5.89E-9 * np.sqrt( TeV ) * y * ni**4# / Eik**2 # mistake in the paper?

    E_y  = np.exp(-y)
    E1_y = Special.E1(y)
    E2_y = Special.E2(y)

    # Eq(8)
    term1 = ( C1 + (C2 + C3 * G) / ni ) * E1_y
    term2 = ( C4 + (C5 + C6 * G) / ni ) * (E_y - y * E1_y)
    term3 = ( C7 + (C8 + C9 * G) / ni ) * (E_y - 2 * y * E1_y + y * E2_y)
    C_y = F * (term1 + term2 + term3)

    qik = C_y
    return qik
