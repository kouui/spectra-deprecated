import numpy as np
import numba as nb

from .. import Constants as Cst

@nb.vectorize( [nb.float64(nb.int_,nb.int_)] , nopython=True)
def Einstein_A_coefficient(ni,nj):
    r"""
    Einstein coefficient for spontaneous emission in the hydrogen atom

    Parameters:
    ------------
    ni : int,  [-]
        principal quantum number of lower level

    nj : int,  [-]
        principal quantum number of upper level

    Returns:
    ---------
    Aji : np.float64, [:math:`s^{-1}`]

    Notes
    ------

    L. C. Johnson, 1972

    """

    coef = 32. / (3. * Cst.sqrt3_ * Cst.pi_ )
    x = 1. - (ni / nj)**2
    xm1 = 1. / x

    if ni == 1:
        g0 = 1.1330
        g1 = -0.4059
        g2 = 0.07014
    elif ni == 2:
        g0 = 1.0785
        g1 = -0.2319
        g2 = 0.02947
    else:
        g0 = 0.9935 + ( 0.2328 - 0.1296 / ni ) / ni
        g1 = - ( 0.6282 - ( 0.5598 - 0.5299 / ni ) / ni ) / ni
        g2 = ( 0.3887 - ( 1.181 - 1.470 / ni ) / ni ) / ni / ni

    fij = coef * ni * ( g0 + ( g1 + g2 * xm1 ) * xm1 ) / ( x * nj )**3
    gf = 2. * ni**2 * fij
    Eij = Cst.E_Rydberg_ * (1./ni**2 - 1./nj**2)
    ambda = Cst.h_ * 3E+14 / Eij

    Aji = 0.6670E+08 * gf / ( 2. * nj**2 * ambda * ambda )

    return Aji
