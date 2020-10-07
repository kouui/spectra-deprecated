################################################################################
# this file defines functions for
#     calculations related to basic/naive mathematical formula
################################################################################

from .. import Constants as Cst
from ..Config import isJIT_

import numba as nb


################################################################################
# If is an odd number, return True
################################################################################
def is_odd(num):
    r"""
    A fast method to check whether a number is odd.

    Parameters
    ----------

    num : int
        input x[-]

    Returns
    -------

    res : int
         1 if input num is an odd number, otherwise 0.
    """
    return num & 0x1

if isJIT_:
    pass
    #is_odd = nb.njit( ['bool(int64)','bool(uint16)'] )( is_odd )
