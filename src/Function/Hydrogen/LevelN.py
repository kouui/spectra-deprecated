
import numpy as np
import numba as nb

from ... import Constants as Cst

@nb.vectorize(['uint8(uint8)', 'int64(int64)'], nopython=True)
def get_level_n( g ):
    r"""
    compute the quantum number n given statistical weight g

    Parameters
    -----------
    g : int, [:math:`-`]
        statistical weight

    Returns
    --------
    n : int [:math:`-`]
        quantum number n
    """
    if g == 1:
        assert False
    else:
        return int( np.sqrt(g//2) )
#get_level_n = np.frompyfunc( get_level_n, 1, 1 )


@nb.njit( ['float64[:,:](uint8[:], float64[:,:])', 'float64[:,:](int64[:], float64[:,:])'] )
def ratio_Etran_to_Eionize( ni_arr, wave_arr ):
    r"""

    """

    nCont = wave_arr.shape[0]
    ratio = np.ones(wave_arr.shape, np.double)

    for k in range(nCont):
        E_ionize = Cst.E_Rydberg_ / ni_arr[k]**2
        E_tran =  Cst.h_ * Cst.c_ / wave_arr[k,:]
        E_tran[:] +=  E_ionize - E_tran[0]

        ratio[k,:] = E_tran[:] / E_ionize

    return ratio
