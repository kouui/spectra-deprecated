################################################################################
# this file defines functions for
#     calculations of opacities of various atomic process
#
################################################################################

import numpy as np

from .. import Constants as Cst

#import pdb


################################################################################
def n_elements(a):
    a = np.asarray(a)
    na = a.size
    return na

################################################################################
# Thomson scattering (wavelength independent)
################################################################################

def Thomson_scattering(n_e):
    r"""
    Thomson scattering of free electrons (non-relativistic).
    Returns absorption coefficient instead of absorption cross section.

    Parameters
    ----------

    n_e : np.double or array-like
        electron density, [:math:`cm^{-3}`]

    Returns
    -------

    kappa : np.double or array-like
        absorption coefficient, [:math:`cm^{-1}`]

    Notes
    -----
    from [1]_.

    .. math:: \kappa = \sigma_{T} n_{e}

    where :math:`\sigma_{T}` is the absorption cross section for Thomson scattering
    .. math:: \sigma_{T} = \frac{8 \pi e^{4}}{3 m_{e}^{2} c^{4}} = 6.6524 \times 10^{-25} \quad cm^{2}

    References
    ----------

    .. [1] Ivan Hubeny, Dimitri Mihalas, "Theory of Stellar Atmosphere:
        An Introduction to Astrophysical Non-equilibrium
        Quantitative Spectroscopic Analysis",
        Princeton University Press, pp. 149, 2015.
    """

    kappa = 6.6524E-25 * n_e
    return kappa

################################################################################
# HI atom bound-free cross section for n=k level
################################################################################
def HIbf_CrossSec1(k,wl):
    r"""
    return bound-free cross section of a HI atom in n=k in unit of e-18 cm**2

    Parameters
    ----------
    k  : principle quantum number of HI atom
    wl : wavelength, [:math:`\AA`]

    Notes
    -----
    from [1]_, [2]_.
    - non-LTE ok
    - stimulated emission is not corrected
    - absorption coeff.:

            kbf = nk*shibf*(1.-exp(-hv/kt))*1.e-18 (/cm)

    Modification history:
    - k.ichimoto 15 jun.1987,	6 Jan.1992
    - k.ichimoto 19 Feb.1994
    - 2019.9.15   k.ichimoto from IDL ahic.pro

    References
    ----------
    .. [1] varnazza et al. (1976) ap.j.suppl. vol.30,1.
    .. [2] polynomial fitting for b-f gaunt factor is taken from Carbon & Gingerich (1969) "theory and observation of normal stellar atmosphere"  mit press

"""

    HIbf_cs_const = {
        1 : (0.9916, 9.068e-3, -0.2524),
        2 : (1.105, -7.922e-2, 4.536e-3),
        3 : (1.101, -3.290e-2, 1.152e-3),
        4 : (1.101, -1.923e-2, 5.110e-4),
        5 : (1.102, -0.01304,  2.638e-4),
        6 : (1.0986,-0.00902,  1.367e-4),
        7 : (1.,0., 0.),
    }
    wlk = 911.76 * k**2
    ak = 7.93 * k
    wl3 = wl/1000.
    gbf = HIbf_cs_const[k][0] + (HIbf_cs_const[k][1]+HIbf_cs_const[k][2]*wl3)*wl3
    nwl=len(wl)
    shibf1=np.zeros(nwl)

    ii=np.where(wl < wlk)[0]
    if ii.size != 0:
        shibf1[ii] = ak*(wl[ii]/wlk)**3 *gbf[ii]
    if nwl == 1:
        shibf1=shibf1[0]

    return shibf1

################################################################################
# HI bound-free CrossSection in LTE per 1 HI atom in unit of e-26 cm^2
################################################################################
def HIbf_CrossSection(T,wl):
    r"""
    return bound-free cross section of a HI atom in LTE in unit of e-26 cm^2

    Parameters
    ----------
    T  : Temperature [:math:`K`]
    wl : wavelength [:math:`\AA`]

    Notes
    -----
    from [1]_.

    HI bound-free opacity in LTE is

          k(T,wl) = n_HI * HIbf_CrossSection(T,wl) * 1e-26 (/cm)

    - n_HI: hydrogen number density in cm^(-3)
    - stimulated emission is corrected
    - partition function of HI is assumed to be 2.0, which is valid for T < 2.e4 K

    Modification history:
    - 2019.9.15  K.Ichimoto  from IDL ahic.pro


    References
    ----------
    .. [1] varnazza et al. (1976) ap.j.suppl. vol.30, 1.

"""

#    wl = np.asarray(wl)
#    T = np.asarray(T)

    nwl = n_elements(wl)
    nt = n_elements(T)
    if nwl == 1:
        wl = np.atleast_1d(wl)[0]
    if nt == 1:
        T = np.atleast_1d(T)[0]
    if (nwl > 1) and (nt > 1):
        print("both T & wl are array! (HIbf_Opacity)")
#        pdb.set_trace()

    l0 = np.floor(np.sqrt(wl/911.76)) + 1
    xl = 157779.*(1. - 1./l0**2)/T
    u = 2.0;
    a = (1.-np.exp(-1.438787e8/wl/T))/u

    if nwl > 1:
        abf = np.zeros(nwl)
        for k in range(1,7):
            ii = np.where(l0 == k)[0]
            if ii.size != 0:
                for l in range(k,min([k+3,7])+1):
                    abf[ii]=abf[ii]+(l**2)*np.exp(-xl[ii])*HIbf_CrossSec1(l,wl[ii])*2.e8
    else:
        abf = np.zeros(nt)
        for l in range(l0[0],min([l0[0]+3,7])+1):
            abf=abf+(l**2)*np.exp(-xl)*HIbf_CrossSec1(l,wl)*2.e8

    if max(l0) > 5:
        print("wl exceeds the limit in 'ahic' !")

    abf = a*abf

    return abf

################################################################################
# HI free-free CrossSection in LTE per 1 HI atom in unit of e-26 cm^2
################################################################################
def HIff_CrossSection(T,wl):
    r"""
    return free-free cross section of a HI atom in LTE in unit of e-26 cm^2

    Parameters
    ----------
    T  : Temperature, [:math:`K`]
    wl : wavelength, [:math:`\AA`]

    Notes
    -----
    from [1]_, [2]_.
    HI free-free opacity in LTE is

          kff = n_HI * HIff_CrossSection(T,wl) * 1e-26 (/cm)

    - n_HI: hydrogen number density in cm^(-3)
    - stimulated emission is corrected
    - partition function of HI is assumed to be 2.0, which is valid for T < 2.e4 K

    Modification history:
    - 2019.9.15  K.Ichimoto  from IDL ahic.pro

    References
    ----------
    .. [1] varnazza et al. (1976) ap.j.suppl. vol.30, 1.
    .. [2] porinomial fitting for f-f gaunt factor is taken from Carbon & Gingerich (1969) "theory and observation of
           normal stellar atmosphere"  MIT press
"""

    nwl = n_elements(wl)
    nt = n_elements(T)
    if nwl == 1:
        wl = np.atleast_1d(wl)[0]
    if nt == 1:
        T = np.atleast_1d(T)[0]
    if (nwl > 1) and (nt > 1):
        print("both T & wl are array! (HIff_Opacity)")
#    assert (nwl > 1) and (nt > 1), "both T & wl are array!"

    r"""
    ;+/*******************************************************************/
    function ahiff,t,wl
    ;/*******************************************************************/
    ;  HI free-free absorption coefficient per 1 proton and unit pe (dyne/cm**2)
        in unit of e-26
    ;  stimulated emission is not corrected
    ;  absorption coeff.:   kff = np*pe*ahiff*(1.-exp(-hv/kt))*1.e-26 (/cm)
    ;  ref.:  varnazza et al. (1976) ap.j.suppl. vol.30,1.
    ;         porinomial fitting for f-f gaunt factor is taken from
    ;         Carbon & Gingerich (1969) "theory and observation of
    ;           normal stellar atmosphere"  MIT press
    ;                           k.ichimoto 15 jun.1987,	6 Jan.1992
    ;                           k.ichimoto 19 Feb.1994
    """
    a = 1.0828 + 3.865e-6 * T
    b = 7.564e-7 + (4.920e-10 - 2.482e-15 *T)*T
    c = 5.326e-12 + (-3.904e-15 + 1.8790e-20 *T)*T
    gff = a+(b+c*wl)*wl
    ahiff = 9.9264e-6* T**(-1.5) * wl**3 *gff

    xi = 157779./T
    u = 2.0;
    a0 = (1.-np.exp(-1.438787e8/wl/T))/u
    aff = 0.66699*T**2.5 *np.exp(-xi)*ahiff

    aff = a0*aff

    return aff

################################################################################
# H-minus (negative hidrogen) CrossSection per HI atom in unit of e-26 cm^2 (*lte*)
################################################################################
def Hminus_CrossSection(T,wl,n_e):
    r"""
    return free-free & bound-free cross section of a H-(negative hidrogen) in LTE in unit of e-26 cm^2

    Parameters
    ----------
    T  : Temperature, [:math:`K`]
    wl : wavelength, [:math:`\AA`]
    n_e: electron number density, [:math:`cm^{-3}`]

    Notes
    -----
    from [1]_, [2]_.

    H- opacity in LTE is

          khm = n_HI * Hminus_CrossSection(T,wl,n_e) * 1.e-26 (/cm)

    - n_HI: hydrogen number density in cm^(-3)
    - stimulated emission is corrected
    - partition function of HI is assumed to be 2.0, which is valid for T < 2.e4 K

    Modification history:
    - k.ichimoto 18 jun. 1987, 	6 Jan. 1992
    - k.ichimoto  19 Feb.1994
    - 2019.9.15  K.Ichimoto  from IDL ahic.pro

    References
    ----------
    .. [1] varnazza et al. (1976) ap.j.suppl. vol.30, 1.
    .. [2] porinomial fitting for f-f gaunt factor is taken from Carbon & Gingerich (1969) "theory and observation of
           normal stellar atmosphere"  MIT press


"""

    nwl = n_elements(wl)
    nt = n_elements(T)
    if nwl == 1:
        wl = np.atleast_1d(wl)[0]
    if nt == 1:
        T = np.atleast_1d(T)[0]
    if (nwl > 1) and (nt > 1):
        print("both T & wl are array! (HIff_Opacity)")


    th = 5039.778/T
    wl3 = wl/1000.
#;/*  ---   bound-free   ---  */
    sigm = np.zeros(nwl)
    ii0 = np.where(wl3 > 16.419)[0] ; count = ii0.size
    if count != 0:
        sigm[ii0] = 0.
    ii1 = np.where( (wl3 < 16.419) & (wl3 > 14.2) )[0] ; count = ii1.size
#    ii1 = np.where( 14.2 < wl3 < 16.419 )[0] ; count = ii1.size
    if count != 0:
        xl = 16.419 - wl3[ii1]
        sigm[ii1] = (0.269818+(0.220190+(-0.0411288+0.00273236)*xl)*xl)*xl
    ii2 = np.where(wl3 <= 14.2)[0] ; count = ii2.size
    if count != 0:
        sigm[ii2] = 0.00680133 + (0.178708+(0.16479 + (-0.0204842 + 5.95244e-4 * wl3[ii2]) * wl3[ii2]) *wl3[ii2])*wl3[ii2]
    if n_elements(sigm) == 1:
        sigm = np.atleast_1d(sigm)[0]
    kbf = 0.41590 * th**2.5 * np.exp(1.738*th) * (1.- np.exp(-28.5486*th/wl3)) * sigm

#;/*  ---   free-free   ---  */
    a = 0.005366+(-0.011493+0.027039*th)*th
    b = -3.2062+(11.924-5.939*th)*th
    c = -0.40192+(7.0355-0.34592*th)*th
    kff = a + (b+c*wl3)*wl3/1000.

    ahm = kbf + kff
    pe = 1.38066e-16 * n_e * T
    hmo = ahm*pe
    return hmo


################################################################################
# HI Rayleigh scattering CrossSection per 1 HI atom in unit of e-26 cm^2
################################################################################
def HIRayleigh_CrossSection(wl):
    r"""
    return Rayleigh scattering cross section of a HI atom in unit of e-26 cm^2

    Parameters
    ----------
    wl : wavelength, [:math:`A`]

    Notes
    -----
    from [1]_, [2]_.

    HI Rayleigh opacity in LTE is

         khray = n_HI * HIRayleigh_CrossSection(wl) * 1.e-26 (/cm)

    - n_HI: hydrogen number density in cm^(-3)
    - stimulated emission is corrected
    - partition function of HI is assumed to be 2.0, which is valid for T < 2.e4 K

    modification history:
    - k.ichimoto 18 jun. 1987, 	6 Jan. 1992
    - k.ichimoto  19 Feb.1994
    - 2019.9.10  K.Ichimoto  from IDL avray.pro

    References
    ----------
    .. [1] Gingerich SAO special report 167,21,1964.
    .. [2] porinomial fitting for f-f gaunt factor is taken from Carbon & Gingerich (1969) "theory and observation of
           normal stellar atmosphere"  MIT press
"""

    #w2 = (wl>1026.)**2
    w2 = (wl.clip(min=1026.))**2
    avray1 = 5.799e13/w2**2 + 1.422e20/w2**3 + 2.784/w2**4
    if n_elements(avray1) == 1:
          avray1 = np.atleast_1d(avray1)[0]
    return avray1

################################################################################
# H2+ CrossSection per 1 HI atom and per 1 H+ in unit of e-26 cm^5?
################################################################################
def avH2p(T,wl):
#  k.i.   2019.9.11
    r"""

    H2+ opacity per 1 HI atom and per 1 H+, scaled by e26,

    Parameters
    ----------
    T  : temperature, [:math:`K`]
    wl : wave length, [:math:`\AA`]

    Returns
    -------
    kh2 : absorption coeff,
        kh2 = nhi * np * avh2p * 1.e-26, [:math:`cm^{-1}`]

    Notes
    -----
    from [1]_, [2]_.

    - stimulated emission is corrected.

    Modification history:
    - k.ichimoto 18 jun. 1987, 	6 Jan. 1992
    - k.ichimoto  19 Feb.1994
    - 2019.9.11  K.Ichimoto  from IDL avh2p.pro


    References
    ----------
    .. [1] Gingerich SAO special report 167,21,1964.
    .. [2] Carbon & Gingerich (1969) "theory and observation of normal stellar atmosphere"  MIT press
    """
    nwl = n_elements(wl)
    nT = n_elements(T)
    if nwl == 1:
        wl = np.atleast_1d(wl)[0]
    if nT == 1:
        T = np.atleast_1d(T)[0]

#;/* -----  data from Carbon & Gingerich (1969)  ----- */
    e = np.array(
          [ 3.0,    2.852,  2.58,   2.294,  2.023,  1.774,
            1.547,  1.344,  1.165,  1.007,  0.8702, 0.7516,
            0.6493, 0.5610, 0.4848, 0.4198, 0.3620, 0.3128,
            0.2702, 0.2332, 0.2011, 0.1732, 0.1491, 0.1281,
            0.1100, 0.09426,0.0809, 0.06906,0.05874,0.04994,
            0.04265,0.03635,0.0308, 0.026,  0.02195,0.01864,
            0.01581,0.01332,0.01118,0.00938,0.00793,0.00669,
            0.00561,0.00469,0.00392,0.0033 ])
    up = np.array(
          [  85.,       9.99465,   4.97842,   3.28472,   2.41452,
             1.87038,   1.48945,   1.20442,   0.98279,   0.80665,
             0.66493,   0.54997,   0.45618,   0.37932,   0.31606,
             0.26382,   0.22057,   0.18446,   0.15473,   0.12977,
             1.08890e-1,9.14000e-2,7.67600e-2,6.44500e-2,5.41200e-2,
             4.54000e-2,3.81000e-2,3.19500e-2,2.67600e-2,2.23700e-2,
             1.86900e-2,1.56100e-2,1.30200e-2,1.08300e-2,8.99000e-3,
             7.45000e-3,6.15000e-3,5.08000e-3,4.16000e-3,3.42000e-3,
             2.77000e-3,2.21000e-3,1.78000e-3,1.45000e-3,1.24000e-3,
             1.14000e-3 ])
    us = np.array(
          [ -85.,      -7.1426,   -2.3984,   -0.99032,  -0.39105,
             -0.09644,  0.05794,   0.13996,   0.18186,   0.20052,
             0.20525,   0.20167,   0.19309,   0.18167,   0.16871,
             0.15511,   0.14147,   0.12815,   0.11542,   0.10340,
             0.09216,   8.18000e-2,7.22900e-2,6.36700e-2,5.58400e-2,
             4.88400e-2,4.25700e-2,3.69900e-2,3.20700e-2,2.77500e-2,
             2.39400e-2,2.06100e-2,1.77200e-2,1.52200e-2,1.30500e-2,
             1.11900e-2,9.58000e-3,8.21000e-3,7.01000e-3,6.00000e-3,
             5.11000e-3,4.35000e-3,3.72000e-3,3.22000e-3,2.86000e-3,
             2.63000e-3 ])
    fr = np.array(
           [ 0.,      4.30272e-18,1.51111e-17,4.02893e-17,8.89643e-17,
          1.70250e-16,2.94529e-16,4.77443e-16,7.25449e-16,1.06238e-15,
          1.50501e-15,2.08046e-15,2.82259e-15,3.76256e-15,4.93692e-15,
          6.38227e-15,8.17038e-15,1.02794e-14,1.28018e-14,1.57371e-14,
          1.91217e-14,2.30875e-14,2.75329e-14,3.27526e-14,3.85481e-14,
          4.52968e-14,5.18592e-14,5.99825e-14,6.92092e-14,7.94023e-14,
          9.01000e-14,1.01710e-13,1.14868e-13,1.29969e-13,1.46437e-13,
          1.63042e-13,1.81440e-13,2.02169e-13,2.25126e-13,2.49637e-13,
          2.73970e-13,3.00895e-13,3.30827e-13,3.64140e-13,3.99503e-13,
          4.34206e-13 ])

    ev = 911.3047/wl
    Tk = 6.3348e-6 * T

    n = np.zeros(nwl, dtype=int)
    for i in range(0,nwl):
        #n[i]=min(where(e le ev[i],count))
        ii = np.where(e <= ev[i])[0] ; count = ii.size
        if count != 0:
            n[i] = ii[0]
        else:
            n[i]=45
    d = (ev-e[n])/(e[n+1]-e[n])
    usq = us[n] + (us[n+1]-us[n])*d
    upq = up[n] + (up[n+1]-up[n])*d
    frq = fr[n] + (fr[n+1]-fr[n])*d
    if nwl == 1:
        usq=usq[0]
        upq=upq[0]
        frq=frq[0]
    avh2p1 = abs( frq * ( np.exp(usq/Tk) - np.exp(-upq/Tk) ) )

    if n_elements(avh2p1) == 1:
        avh2p1=avh2p1[0]

    return avh2p1

################################################################################
# H2+ CrossSection per 1 HI atom n unit of e-26 cm^2 in LTE
################################################################################
def H2p_CrossSection(T,wl,n_e,n_H):
    r"""
    return H2+ CrossSection per 1 HI atom n unit of e-26 cm^2 in LTE

    Parameters
    ----------
    T  : Temperature [:math:`K`]
    wl : wavelength [:math:`\AA`]
    n_e: electron number density [:math:`cm^{-3}`]
    n_H: hydrogen number density [:math:`cm^{-3}``]

    Notes
    -----
    H2+ opacity in LTE is

         kH2p = n_HI * H2p_CrossSection(T,wl,n_e,n_H) * 1.e-26 (/cm)

    use `avH2p(T,wl)`

    Modification history:
    - 2019.9.15  K.Ichimoto
    """

    pe = 1.38066e-16 * n_e*T

    #;/*  -----   solving LTE Saha's eq. for hydrogen   -----	*/
    q = 2.0*2.07e-16 * T**(-1.5) * 10.**(5040.*13.6/T) * n_e
    np  = 1.0/(q+1.0)*n_H

    return np*avH2p(T,wl)

################################################################################
# LTE continuum opacity in cm-1
################################################################################
def xclte(T,n_H,n_e,wl):
    r"""
    LTE continuum opacity in cm-1, incruding HI, H-, H2, rayleigh scat.

    Parameters
    ----------
    T  : Temperature, [:math:`K`]
    n_H: H atom number density, [:math:`cm^{-3}`]
    n_e: electron number density, [:math:`cm^{-3}`]
    wl : wavelength, [:math:`\AA`]

    Notes
    -----
    HI Rayleigh opacity in LTE is [1]_.

         khray = n_HI * HIRayleigh_CrossSection(wl) * 1.e-26 (/cm)

    - n_HI: hydrogen number density in cm^(-3)
    - stimulated emission is corrected
    - partition function of HI is assumed to be 2.0, which is valid for T < 2.e4 K

    Modification history:
    - k.ichimoto 18 jun. 1987, 	6 Jan. 1992
    - k.ichimoto  19 Feb.1994
    - 2019.9.15  K.Ichimoto  from IDL avray.pro

    References
    ----------
    .. [1] Gingerich SAO special report 167,21,1964. porinomial fitting for f-f gaunt factor is taken from Carbon & Gingerich (1969) "theory and observation of normal stellar atmosphere"  MIT press
    """
    nwl = n_elements(wl)
    nT  = n_elements(T)
    nnH = n_elements(n_H)
    nne = n_elements(n_e)
    if nwl == 1:
        wl = np.atleast_1d(wl)[0]
    if nT == 1:
        T = np.atleast_1d(T)[0]
    if nnH == 1:
        n_H = np.atleast_1d(n_H)[0]
    if nne == 1:
        n_e = np.atleast_1d(n_e)[0]

#;/*  -----   solving LTE Saha's eq. for hydrogen   -----	*/
    q = 2.0*2.07e-16 * T**(-1.5) * 10.**(5040.*13.6/T) * n_e
    nHI =  q /(q+1.0)*n_H

    xclte = nHI * ( HIbf_CrossSection(T,wl) + HIff_CrossSection(T,wl) + Hminus_CrossSection(T,wl,n_e)
              + H2p_CrossSection(T,wl,n_e,n_H) + HIRayleigh_CrossSection(wl) ) * 1.e-26

    return xclte
