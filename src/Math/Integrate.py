################################################################################
# this file defines functions for
#     integration method
################################################################################

import numpy as np
import numba as nb

from .. import Constants as Cst
from ..Config import isJIT_



################################################################################
# integration with using Tapzodial method
################################################################################

def Trapze(integrand, x):
    r"""
    Integration using Trapzoidal rule

    Parameters
    ----------
    integrand : array-like of np.double
        integrand as a function of variable x

    x : array-like of np.double
        independent variable x

    Returns
    -------

    sum : np.double
         result of integration

    Notes
    ------
    Refer to [1]_.

    References
    ----------
    .. [1] Trapzoidal rule, wikipedia, https://en.wikipedia.org/wiki/Trapezoidal_rule
    """

    n = x.size
    dx = np.empty(n,dtype=np.double)
    for i in range(1, n-1):
        dx[i] = 0.5 * (x[i+1]-x[i-1])
    dx[0] = 0.5 * (x[1]-x[0])
    dx[n-1] = 0.5 * (x[n-1] - x[n-2])

    sum = 0.0
    for i in range(n):
        sum += dx[i] * integrand[i]
    return sum

################################################################################
# whether to compile them using numba's LLVM
################################################################################

if isJIT_ == True:
    Trapze = nb.jit([nb.float64(nb.float64[:],nb.float64[:]),nb.int_(nb.int_[:],nb.int_[:])],nopython=True)( Trapze )
