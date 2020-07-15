import numpy as np
from .. import Constants as Cst

from scipy.interpolate import splrep, splev, interp1d
from ..Math import Integrate


def interpolate_PI_intensity(_backRad, _continuum_mesh_list):
    r"""
    given continuum mesh, interpolate background intensity profile

    Parameters
    ----------

    _backRad : np.double, array, (2, :)
        a table of background intensity, wavelength_cm vs intensity_cm

    _continuum_mesh_list : list of 1darray
        list of continuum mesh

    Returns
    -------

    _intensity_mesh_list : list of 1darray
        list of interpolated continuum intensity profile.
    """

    _fill_value = (_backRad[1,0],_backRad[1,-1])
    _bsp_obj = interp1d(x=_backRad[0,:], y=_backRad[1,:],
                        bounds_error=False, fill_value=_fill_value)

    _intensity_mesh_list = []
    for _cont_mesh in _continuum_mesh_list:
        # ext=3 : return boundary value
        #_int_mesh = splev(_cont_mesh[:], _bsp_obj, ext=3)
        _int_mesh = _bsp_obj(_cont_mesh[:])
        _intensity_mesh_list.append( _int_mesh )

    return _intensity_mesh_list

def interpolate_PI_alpha(_PI_table_list, _continuum_mesh_list):
    r"""
    given continuum mesh, interpolate photoionization cross section

    Parameters
    ----------

    _PI_table_list : list of 2darray,
        list of table of photoionization cross section, wavelength_cm vs alpha_cm^2

    _continuum_mesh_list : list of 1darray
        list of continuum mesh

    Returns
    -------

    _alpha_mesh_list : list of 1darray
        list of interpolated photoionization cross section.
    """

    _alpha_mesh_list = []
    for _alpha_table, _cont_mesh in zip(_PI_table_list, _continuum_mesh_list):
    #for k in range(len(_continuum_mesh_list)):
    #    _alpha_table = _PI_table_dict[k]
    #    _cont_mesh = _continuum_mesh_list[k]
        #_fill_value = (_alpha_table[1,-1], 0)
        _bsp_obj = interp1d(x=_alpha_table[0,:], y=_alpha_table[1,:], kind="cubic",
                            bounds_error=False, fill_value="extrapolate")
        _alpha_mesh = _bsp_obj(_cont_mesh[:])
        _alpha_mesh_list.append( _alpha_mesh )

    return _alpha_mesh_list


def bound_free_radiative_transition_coefficient(wave, J, alpha, Te, ni_lte, nk_lte):
    r"""
    Given wavelength mesh, mean intensity (as function of wavelength),
    photoionization cross section, compute

    - radiative ionization rate,
    - stimulated radiative recombination rate,
    - spontaneous radiative recombination rate.

    Parameters
    ----------

    wave : array-like of np.double
        wavelength mesh, [:math:`cm`]

    J : array-like of np.double
        mean intensity as function of wavelength, [:math:`erg/cm^2/Sr/cm/s`]

    alpha : array-like of np.double
        photoionization cross section as function of wavelength, [:math:`cm^{2}`]

    Te : np.double
        Temperature, [:math:`K`]

    ni_lte : np.double
        LTE population of level i, [:math:`cm^{-3}`]

    nk_lte : np.double
        LTE population of level k, [:math:`cm^{-3}`]

    Returns
    -------

    Rik : np.double
        Radiative ionization rate, [:math:`s^{-1} \cdot cm^{-3}`]

    Rki_stim : np.double
        Stimulated Radiative ionization rate, [:math:`s^{-1} \cdot cm^{-3}`]

    Rki_spon : np.double
        Spontaneous Radiative ionization rate, [:math:`s^{-1} \cdot cm^{-3}`]


    Notes
    -----

    Radiative ionization rate [1]_ Equation(9.43),

        .. math:: R_{ik} = 4\pi \int_{\nu_0}^{\infty} \alpha_{ik}(\nu) (h\nu)^{-1} J(\nu) d\nu

    Stimulated radiative recombination rate [1]_ Equation(9.47),

        .. math:: R_{ki}^{stim} = \frac{n_i^{LTE}}{n_k^{LTE}} 4\pi \int_{\nu_0}^{\infty} \alpha_{ik}(\nu) (h\nu)^{-1} J(\nu) e^{-h\nu/kT} d\nu

    Spontaneous radiative recombination rate [1]_ Equation(9.45)

        .. math:: R_{ki}^{stim} = \frac{n_i^{LTE}}{n_k^{LTE}} 4\pi \int_{\nu_0}^{\infty} \alpha_{ik}(\nu) (h\nu)^{-1} (2h\nu^{3}/c^2) e^{-h\nu/kT} d\nu

    References
    -----------

    .. [1] Ivan Hubeny, Dimitri Mihalas, "Theory of Stellar Atmosphere:
        An Introduction to Astrophysical Non-equilibrium
        Quantitative Spectroscopic Analysis",
        Princeton University Press, pp. 273, 2015.
    """
    # factor : h\nu
    hv = Cst.h_ * Cst.c_ / wave

    expo = np.exp(-hv/(Cst.k_*Te))

    integrand_ik = alpha*J/hv
    Rik = 4*Cst.pi_ * Integrate.Trapze(integrand_ik, wave)

    # factor : [ni/nk]_{LTE}
    #factor_ = ne * (gi/gk) / expo[-1] * Te**(-1.5) * Cst.saha_**(-1)
    _factor = ni_lte / nk_lte

    integrand_ki_stim = integrand_ik * expo
    Rki_stim = _factor * 4*Cst.pi_ * Integrate.Trapze(integrand_ki_stim, wave)

    integrand_ki_spon = alpha/hv * (2.*Cst.h_*Cst.c_/wave**3) * expo
    Rki_spon = _factor * 4*Cst.pi_ * Integrate.Trapze(integrand_ki_spon, wave)

    return Rik, Rki_stim, Rki_spon

def bound_free_radiative_transition_coefficient_v0(wave,J,alpha,Te,gi,gk,ne):
    r"""
    Given wavelength mesh, mean intensity (as function of wavelength),
    photoionization cross section, compute
    radiative ionization rate,
    stimulated radiative recombination rate,
    spontaneous radiative recombination rate.

    Parameters
    ----------

    wave : array-like of np.double
        wavelength mesh, [:math:`cm`]

    J : array-like of np.double
        mean intensity as function of wavelength, [:math:`erg/cm^2/Sr/cm/s`]

    alpha : array-like of np.double
        photoionization cross section as function of wavelength, [:math:`cm^{2}`]

    Te : np.double
        Temperature, [:math:`K`]

    gi : np.uint8
        statistical weight of lower level i, [-]

    gj : np.uint8
        statistical weight of upper level j, [-]

    ne : np.double
        electron density, [:math:`cm^{-3}`]

    Returns
    -------

    Rik : np.double
        Radiative ionization rate, [:math:`s^{-1} \cdot cm^{-3}`]

    Rki_stim : np.double
        Stimulated Radiative ionization rate, [:math:`s^{-1} \cdot cm^{-3}`]

    Rki_spon : np.double
        Spontaneous Radiative ionization rate, [:math:`s^{-1} \cdot cm^{-3}`]


    Notes
    -----

    Radiative ionization rate [1]_ Equation(9.43),

        .. math:: R_{ik} = 4\pi \int_{\nu_0}^{\infty} \alpha_{ik}(\nu) (h\nu)^{-1} J(\nu) d\nu

    Stimulated radiative recombination rate [1]_ Equation(9.47),

        .. math:: R_{ki}^{stim} = \frac{n_i^{LTE}}{n_k^{LTE}} 4\pi \int_{\nu_0}^{\infty} \alpha_{ik}(\nu) (h\nu)^{-1} J(\nu) e^{-h\nu/kT} d\nu

    Spontaneous radiative recombination rate [1]_ Equation(9.45)

        .. math:: R_{ki}^{spon} = \frac{n_i^{LTE}}{n_k^{LTE}} 4\pi \int_{\nu_0}^{\infty} \alpha_{ik}(\nu) (h\nu)^{-1} (2h\nu^{3}/c^2) e^{-h\nu/kT} d\nu

    References
    -----------

    .. [1] Ivan Hubeny, Dimitri Mihalas, "Theory of Stellar Atmosphere:
        An Introduction to Astrophysical Non-equilibrium
        Quantitative Spectroscopic Analysis",
        Princeton University Press, pp. 273, 2015.
    """
    # factor : h\nu
    hv = Cst.h_ * Cst.c_ / wave

    expo = np.exp(-hv/(Cst.k_*Te))

    integrand_ik = alpha*J/hv
    Rik = 4*Cst.pi_ * Integrate.Trapze(integrand_ik, wave)

    # factor : [ni/nk]_{LTE}
    factor_ = ne * (gi/gk) / expo[-1] * Te**(-1.5) * Cst.saha_**(-1)

    integrand_ki_stim = integrand_ik * expo
    Rki_stim = factor_ * 4*Cst.pi_ * Integrate.Trapze(integrand_ki_stim, wave)

    integrand_ki_spon = alpha/hv * (2.*Cst.h_*Cst.c_/wave**3) * expo
    Rki_spon = factor_ * 4*Cst.pi_ * Integrate.Trapze(integrand_ki_spon, wave)

    return Rik, Rki_stim, Rki_spon
