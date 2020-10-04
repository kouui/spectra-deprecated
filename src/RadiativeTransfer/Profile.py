################################################################################
# this file defines functions for
#     calculations related to spectral profile
#
################################################################################

import numpy as np
import numba as nb

from .. import Constants as Cst
from ..Math import BasicM


################################################################################
# absorption profile
#    - Voigt
#    - Gaussian
################################################################################
@nb.vectorize( [nb.float64(nb.float_,nb.float_)],nopython=True)
def Voigt(a,x):
    r"""
    Calculate Doppler width normalized voigt function using polynomial fitting formula.

    `voigt(a,x)` function itself is not a vectorized function, only available to scalar operation.
    so we applied

    `nb.vectorize([nb.float64(nb.float64,nb.float64)],nopython=True)`.

    Parameters
    ----------

    a : np.double or array-like
        damping constant normalized by Doppler width, [-]
    x : np.double or array-like
        Doppler width normalized mesh, [-]

    Returns
    -------

    res : np.double or array-like
        voigt function, normalized to 1, [-]

    Notes
    -----

    The Voigt function is not normalized but has area :math:`\sqrt{\pi}` in `x` unit

    This is a combination of Humlicek(1982) [1]_ and Hui et al.(1978) [2]_ methods.

    When :math:`a > 0.1`, one can not ignore the wing component of Voigt function.
    That is, to guarantee its normalization, one has to take care of
    whether the mesh points are wide enough.

    References
    ----------

    .. [1] J.Humlicek, 'Optimized computation of the voigt and complex probability functions',
        Journal of Quantitative Spectroscopy and Radiative Transfer (JQSRT),
        Volume 27, Issue 4, April 1982, Pages 437-444.

    .. [2] A.K.Hui, B.H.Armstrong, A.A.Wray, 'Rapid computation of the Voigt and complex error functions',
        Journal of Quantitative Spectroscopy and Radiative Transfer (JQSRT),
        Volume 19, Issue 5, May 1978, Pages 509-516.
    """

    if a < 0.01:
        Z = a - 1j*x
        U = Z * Z
        A0 = 36183.31; B0 = 32066.6
        A1 = 3321.9905;B1 = 24322.84
        A2 = 1540.787; B2 = 9022.228
        A3 = 219.0313; B3 = 2186.181
        A4 = 35.76683; B4 = 364.2191
        A5 = 1.320522; B5 = 61.57037
        A6 = .56419  ; B6 = 1.841439
        F = (np.exp(U) - Z * ( A0 - U * ( A1 - U * ( A2 - U * ( A3 - U * ( A4 - U * ( A5 - U * A6 )))))) /
         ( B0 - U * ( B1 - U * ( B2 - U * ( B3 - U * ( B4 - U * ( B5 - U * ( B6 - U ))))))))
    else:
        A0 = 122.607931777; B0 = 122.607931774
        A1 = 214.382388695; B1 = 352.730625111
        A2 = 181.928533092; B2 = 457.334478784
        A3 = 93.1555804581; B3 = 348.703917719
        A4 = 30.1801421962; B4 = 170.354001821
        A5 = 5.91262620977; B5 = 53.9929069129
        A6 = .564189583563; B6 = 10.4798571143

        Z = a - 1j*abs(x)# * C.k[0]
        F = ( ( A0 + Z * ( A1 + Z * ( A2 + Z * ( A3 + Z * ( A4 + Z * ( A5 + Z * A6 ) ) ) ) ) ) /
             ( B0 + Z * ( B1 + Z * ( B2 + Z * ( B3 + Z * ( B4 + Z * ( B5 + Z * ( B6 + Z ) ) ) ) ) ) ) )

    res = F.real / Cst.sqrtPi_
    return res

def Gaussian(x):
    r"""
    Calculate Doppler width normalized gaussian profile.

    Parameters
    ----------

    x : np.double or array-like
        Doppler width normalized mesh, [-]

    Returns
    -------

    res : np.double or array-like
        gaussian profile, normalized to 1, [-]

    Notes
    -----

    This formula refers to [1]_.

    References
    ----------

    .. [1] Ivan Hubeny, Dimitri Mihalas, "Theory of Stellar Atmosphere:
        An Introduction to Astrophysical Non-equilibrium
        Quantitative Spectroscopic Analysis",
        Princeton University Press, pp. 203, Eq(8.24), 2015.
    """
    res = np.exp(-x*x) / Cst.sqrtPi_

    return res

################################################################################
# constructing mesh distribution
################################################################################
def makeLineMesh_Half(nLambda, qcore, qwing, q):
    r"""
    Construct half line mesh. Following RH's `getlambda.c`.
    called by it's wrapper function `makeLineMesh_Full`

    Parameters
    ----------

    nLambda : np.uint8 or any other integer type
        Number of points in full line mesh. Must be an odd number.

    qcore : np.double
        a parameter to control mesh density.
        there are always half of all points inside [-qcore,qcore].

    qwing : np.double
        a parameter to control how far away your mesh points reach.

    q : array-like, dtype of np.double
        a array to store output half line mesh distribution.

    Notes
    -------
    with q[0] = 0 and q[-1] = qwing, nLambda//2 points belonging to -qcore< q < +qcore. So,

    - change qwing, you change how far your Doppler width mesh reach
    - change qcore, you change how dense in line core.
    """
    #assert BasicM.is_odd(nLambda), "nLambda should be an odd number."
    nLhalf = nLambda//2 + 1

    if qwing <= 2*qcore:
        beta = 1.0
    else:
        beta = 0.5 * qwing / qcore

    y = beta + (beta*beta + (beta-1.)*nLhalf + 2. - 3.*beta)**(0.5)
    b = 2.0*np.log(y) / (nLhalf - 1)
    a = qwing / (nLhalf - 2. + y*y)

    for i in range(0, nLhalf):
        q[i] = a * (i + (np.exp(b*i)-1.))

def makeLineMesh_Full(nLambda, qcore=2.5, qwing=10.):
    r"""
    Construct Full line mesh.
    Calls inner function `makeLineMesh_Half` to construct half part
    and then make the use of symmetry.

    Parameters
    ----------

    nLambda : np.uint8 or any other integer type
        Number of points in full line mesh. Must be an odd number.

    qcore : np.double
        a parameter to control mesh density.
        there are always half of all points inside [-qcore,qcore].

    qwing : np.double
        a parameter to control how far away your mesh points reach.

    Returns
    --------
    x : array-like, dtype of np.double
        Full line mesh distribution.

    Notes
    -------
    with q[0] = 0 and q[-1] = qwing, nLambda//2 points belonging to -qcore< q < +qcore. So,

    - change qwing, you change how far your Doppler width mesh reach
    - change qcore, you change how dense in line core.
    """

    x = np.empty(nLambda, dtype=np.double)
    nLmid = nLambda // 2
    makeLineMesh_Half(nLambda, qcore, qwing, x[nLmid:])
    x[:nLmid] = -x[nLmid+1:][::-1]

    return x

def makeContinuumMesh(nLambda):
    r"""
    Given number of mesh points,
    compute a mesh distribution sampling most aroung wavelength edge.

    Parameters
    ----------

    nLambda : np.uint8 or any other integer type
        Number of points in Continuum mesh.

    Returns
    --------

    Mesh : array-like, dtype of np.double
        Output mesh distribution.

    """
    mesh = np.empty(nLambda, dtype=np.double)
    for j in range(1, nLambda+1):
        qj_ = (nLambda+1.-j) / nLambda
        mesh[j-1] = qj_**(0.5)

    return mesh

def half_to_full(_arr_half, _isMinus=False):
    r"""
    create full (anti-)symmetric full array according to half array

    Parameters
    ----------

    _arr_half : 1darray,
        half array.

    _isMinus : boolean
        True : anti-symmetric full array; False : symmetric full array

    Returns
    --------

    _arr_full : 1darray,
        full array.

    """
    _nLmid = _arr_half.shape[0]
    _nLfull = (_nLmid-1) * 2 + 1
    _arr_full = np.zeros(_nLfull, dtype=_arr_half.dtype)

    if _isMinus:
        _fac = -1
    else:
        _fac = 1

    _arr_full[_nLmid:] = _arr_half[1:]
    _arr_full[:_nLmid] = _fac * _arr_half[::-1]

    return _arr_full


################################################################################
# whether to compile them using numba's LLVM
################################################################################
if Cst.isJIT == True:
    makeLineMesh_Half = nb.njit( ['(int64,float64,float64,float64[:])','(uint16,float64,float64,float64[:])'] )( makeLineMesh_Half )
    makeLineMesh_Full = nb.njit( ['float64[:](int64,float64,float64)', 'float64[:](uint16,float64,float64)'] )( makeLineMesh_Full )
