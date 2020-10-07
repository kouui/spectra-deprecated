################################################################################
# this file defines functions for
#     calculating Gaunt factors used in calculating collisoinal and
#     radiative transition rates in atomic hydrogen
#
# ref :
#   L. C. Johnson, "Approximations for collisional and
#          radiative transition rates in atomic hydrogen",
#          Astrophysical Journal, vol. 174, p.227, May 1972.
#          1972ApJ...174..227J
################################################################################

import numba as nb
from .. import Constants as Cst
from .. import Config


def g0(_n):
    r"""

    Notes
    -----

    Refer to [1]_.

    References
    ----------

    .. [1] L. C. Johnson, "Approximations for collisional and
           radiative transition rates in atomic hydrogen",
           Astrophysical Journal, vol. 174, p.227, May 1972.
           1972ApJ...174..227J

    """

    if _n==1:
        _g = 1.133
    elif _n==2:
        _g = 1.0785
    else:
        _g = 0.9935 + (0.2328 - 0.1296/_n)/_n
    return _g


def g1(_n):
    r"""

    Notes
    -----

    Refer to [1]_.

    References
    ----------

    .. [1] L. C. Johnson, "Approximations for collisional and
           radiative transition rates in atomic hydrogen",
           Astrophysical Journal, vol. 174, p.227, May 1972.
           1972ApJ...174..227J

    """

    if _n==1:
        _g = -0.4059
    elif _n==2:
        _g = -0.2319
    else:
        _g = -(0.6282 - (0.5598 - 0.5299/_n)/_n) / _n
    return _g

def g2(_n):
    r"""

    Notes
    -----

    Refer to [1]_.

    References
    ----------

    .. [1] L. C. Johnson, "Approximations for collisional and
           radiative transition rates in atomic hydrogen",
           Astrophysical Journal, vol. 174, p.227, May 1972.
           1972ApJ...174..227J

    """

    if _n==1:
        g = 0.07014
    elif _n==2:
        _g = 0.02947
    else:
        _g = (0.3887 - (1.181 - 1.4700/_n)/_n)/(_n*_n)
    return g


################################################################################
# whether to compile them using numba's LLVM
################################################################################

if Config.isJIT == True :
    g0 = nb.vectorize( [nb.float64(nb.int64),nb.float64(nb.uint8)],nopython=True)( g0 )
    g1 = nb.vectorize( [nb.float64(nb.int64),nb.float64(nb.uint8)],nopython=True)( g1 )
    g2 = nb.vectorize( [nb.float64(nb.int64),nb.float64(nb.uint8)],nopython=True)( g2 )
