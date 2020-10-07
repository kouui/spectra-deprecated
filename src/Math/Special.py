################################################################################
# this file defines functions for
#     calculating Mathematical special functions
################################################################################

import numba as nb
import numpy as np

from .. import Constants as Cst
from ..Config import isJIT_


################################################################################
# Exponential integral E1(x) and E2(x)
#
#       En(x) = \int_{1}^{\infty} t^{1/n}e^{-xt}dt, x>0, n=0,1,...
#
#       with relation: En+1 = 1/n (exp(-x) - xEn(x))
################################################################################

a53 = np.array([-0.57721566,  0.99999193, -0.24991055,
                0.05519968, -0.00976004,  0.00107857], dtype=np.double)
a56 = np.array([8.5733287401, 18.0590169730, 8.6347608925,  0.2677737343],dtype=np.double)
b56 = np.array([9.5733223454, 25.6329561486,21.0996530827,  3.9584969228],dtype=np.double)

def E0(x):
    return np.exp(-x) / x

def E1(x):
    r"""
    Approximated formula for Exponential integral :math:`E_1(x)`.

    `E1(x)` function itself is not a vectorized function, only available to scalar operation.
    so we applied

    `@nb.vectorize([nb.float64(nb.float64), nb.float64(nb.int_)], nopython=True)`.

    Parameters
    ----------

    x : np.double or array-like
        independent variable x

    Returns
    -------

    E1_ : np.double or array-like
         1st order Exponential integral of x

    Notes
    -----
    Refer to [1]_.

    References
    ----------

    .. [1] D.ABarry, J.-Y.Parlange, "Approximation for the exponential integral (Theis well function)",
           Journal of Hydrology, Volume 227, Issues 1â€“4, 31 January 2000, Pages 287-291

    """
    assert 0 < x <= 80.0, "argument x should be a positive number smaller than 80.0"
    E1_ = 0.0

    if x <= 1.0:
        E1_ = -np.log(x) + a53[0] + x*(a53[1] + x*(a53[2] + x*(a53[3] + x*(a53[4] + x*a53[5]))))
    else:
        E1_  = a56[3]/x +  a56[2] + x*(a56[1] + x*(a56[0] + x))
        E1_ /= b56[3] + x*(b56[2] + x*(b56[1] + x*(b56[0] + x)))
        E1_ *= np.exp(-x)

    return E1_


def E2(x):
    r"""
    Calculate Exponential integral :math:`E_2(x)` from :math:`E_1(x)`.

    Parameters
    ----------

    x : np.double or array-like
        independent variable x

    Returns
    -------

    E2_ : np.double or array-like
         2nd order Exponential integral of x

    Notes
    ------
    Using relation [1]_ Equation(11.127):

    .. math:: E_{n+1}(x) = [e^{-x} - xE_{n}(x)] / n

    References
    ----------
    .. [1] Ivan Hubeny, Dimitri Mihalas, "Theory of Stellar Atmosphere:
        An Introduction to Astrophysical Non-equilibrium
        Quantitative Spectroscopic Analysis",
        Princeton University Press, pp. 365, 2015.
    """
    E2_ = np.exp(-x) - x*E1(x)
    return E2_

def E3(x):
    return 1/2 * ( np.exp(-x) - x*E2(x) )

################################################################################
# whether to compile them using numba's LLVM
################################################################################

if isJIT_ == True :
    E0 = nb.jit( [nb.float64(nb.float64), nb.float64(nb.int_)], nopython=True)( E0 )
    E1 = nb.vectorize( [nb.float64(nb.float64), nb.float64(nb.int_)], nopython=True)( E1 )
    E2 = nb.vectorize( [nb.float64(nb.float64), nb.float64(nb.int_)], nopython=True)( E2 )
    E3 = nb.vectorize( [nb.float64(nb.float64), nb.float64(nb.int_)], nopython=True)( E3 )
