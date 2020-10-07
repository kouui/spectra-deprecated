import numpy as np
import numba as nb

from .. import Constants as Cst
from .. import Config

#@nb.njit
def setMatrixC(_Cmat, _Cji, _Cij, _idxI, _idxJ, _Ne):
    r"""
    Compute the collisional rate matrix.

    Parameters
    ----------

    _Cmat : numpy.2darray of np.double
        collisional rate matrix, a 2D Array to store computed results, [:math:`s^{-1}`]

    _Cji : numpy.1darray of np.double
        downward collisional transition rate, [:math:`s^{-1} \cdot cm^{3}`]

    _Cij : numpy.1darray of np.double
        upward collisional transition rate, [:math:`s^{-1} \cdot cm^{3}`]

    _idxI : numpy.1darray of np.uint16
        level index of lower level i, [-]

    _idxJ : numpy.1darray of np.uint16
        level index of upper level j, [-]

    _Ne: np.double
        electron density, [:math:`cm^{-3}`]

    Notes
    -----
    Refer to [1]_ Equation(9.80).

    .. math:: \sum_{j \neq i} n_j (R_{ji}+n_{e} C_{ji}) - n_i \sum_{j \neq i}(R_{ij}+n_{e} C_{ij}) = 0


    References
    ----------

    .. [1] Ivan Hubeny, Dimitri Mihalas, "Theory of Stellar Atmosphere:
        An Introduction to Astrophysical Non-equilibrium
        Quantitative Spectroscopic Analysis",
        Princeton University Press, pp. 282, 2015.

    """

    _n_row, _n_col = _Cmat.shape
    assert _n_row == _n_col, '_Cmat should be a squared matrix.'

    _nTran = _Cji.size
    for k in range(_nTran):
        i, j = _idxI[k], _idxJ[k]
        _Cmat[i,j] += _Ne * _Cji[k]
        _Cmat[j,i] += _Ne * _Cij[k]

#@nb.njit
def setMatrixR(_Rmat, _Rji_spon, _Rji_stim, _Rij, _idxI, _idxJ):
    r"""
    Compute the radiative rate matrix.

    Parameters
    ----------

    _Rmat : numpy.2darray of np.double
        radiative rate matrix, a 2D Array to store computed results, [:math:`s^{-1}`]

    _Rji_spon : numpy.1darray of np.double
        spontaneous radiative transition rate, [:math:`s^{-1}`]

    _Rji_stim : numpy.1darray of np.double
        spontaneous radiative transition rate, [:math:`s^{-1}`]

    _Rij : numpy.1darray of np.double
        upward radiative transition rate, [:math:`s^{-1}`]

    _idxI : numpy.1darray of np.uint16
        level index of lower level i, [-]

    _idxJ : numpy.1darray of np.uint16
        level index of upper level j, [-]

    Notes
    -----
    Refer to [1]_ Equation(9.80).

    .. math:: \sum_{j \neq i} n_j (R_{ji}+C_{ji}) - n_i \sum_{j \neq i}(R_{ij}+C_{ij}) = 0


    References
    ----------

    .. [1] Ivan Hubeny, Dimitri Mihalas, "Theory of Stellar Atmosphere:
        An Introduction to Astrophysical Non-equilibrium
        Quantitative Spectroscopic Analysis",
        Princeton University Press, pp. 282, 2015.

    """
    _n_row, _n_col = _Rmat.shape
    assert _n_row == _n_col, '_Rmat should be a squared matrix.'

    _nTran = _Rji_spon.size
    for k in range(_nTran):
        i, j = _idxI[k], _idxJ[k]
        _Rmat[i,j] += _Rji_spon[k] + _Rji_stim[k]
        _Rmat[j,i] += _Rij[k]


def solveSE(_Rmat, _Cmat):
    r"""
    solve the linear equation system of statistical equilibrium.

    Parameters
    ----------

    _Rmat : np.double, np.array, (nLevel,nLevel)
        radiative transition rate matrix, [:math:`s^{-1}`]

    _Cmat : np.double, np.array, (nLevel,nLevel)
        collisional transition rate matrix, [:math:`s^{-1}`]

    Returns
    -------

    _nArr : np.double, np.array, (nLevel,)
        normalized level population. [:math:`cm^{-3}`]

    """

    _nLevel = _Rmat.shape[0]
    _A = _Cmat[:,:] + _Rmat[:,:]
    _b = np.zeros(_nLevel, dtype=np.double)

    #-------------------------------------------------------------
    # diagnal components
    #-------------------------------------------------------------
    for k in range(_nLevel):
        _A[k,k] = -_A[:,k].sum()

    #-------------------------------------------------------------
    # abundance definition equation
    #-------------------------------------------------------------
    _A[-1,:] = 1.
    _b[-1] = 1.

    _nArr = np.linalg.solve(_A,_b)

    return _nArr

#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------

if Config.isJIT == True :
    solveSE = nb.njit( ['float64[:](float64[:,:],float64[:,:])'] )( solveSE )
    setMatrixC = nb.njit( setMatrixC )
    setMatrixR = nb.njit( setMatrixR )
