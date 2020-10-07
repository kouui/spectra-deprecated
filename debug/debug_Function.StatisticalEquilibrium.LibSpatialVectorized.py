
import sys
sys.path.append("../")

from src.Structure import AtomCls
from src.Function.StatisticalEquilibrium import LibSpatialVectorized as SELib

import numpy

if __name__ == "__main__":

    atom_, _ = AtomCls.InitAtom("../data/conf/H.conf", isHydrogen=True)

    #Te_ = 1E4
    #Ne_ = 1E11
    arr_ = numpy.ones((10,10),dtype=numpy.double)
    Te_ = arr_ * 1.E4
    Ne_ = arr_ * 1.E11
    Tr_ = 6E3

    n_LTE_ , nj_by_ni_Line_, nj_by_ni_Cont_ = SELib.ni_nj_LTE(atom_, Te_, Ne_)
