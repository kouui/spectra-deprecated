import numpy as np
from .. import Constants as Cst
from . import AtomIO

from ..RadiativeTransfer import Profile

class RadiativeLine:

    def __init__(self, _parent, _folder):
        r"""
        """
        self.I_cm = [] # List()
        self.I_hz = [] # List()
        self.lineIndex = np.zeros(_parent.nRadiativeLine, np.uint32)


        for k in range(_parent.nRadiativeLine):
            filename = _parent.Mesh.RadLine_filenames[k]
            arr = AtomIO.read_half_intensity_hz(_folder+'/'+filename)
            lineIndex = _parent.Mesh.Coe.lineIndex[k]
            self.lineIndex[k] = lineIndex
            f0 = _parent.Line.f0[lineIndex]
            w0 = _parent.Line.w0[lineIndex]

            #arr[:,0] *= fac1 # d\nu -> d\lambda
            #arr[:,1] *= fac2 # intensity_hz -> intensity_cm
            nLmid = arr.shape[0]
            nLfull = (nLmid-1) * 2 + 1
            arr_full_hz = np.zeros((2,nLfull), dtype=np.double)
            arr_full_hz[0,:] = Profile.half_to_full(_arr_half=arr[:,0], _isMinus=True)
            arr_full_hz[1,:] = Profile.half_to_full(_arr_half=arr[:,1], _isMinus=False)
            self.I_hz.append( arr_full_hz )

            fac1 = Cst.c_ / (f0 * f0)
            fac2 = Cst.c_ / (w0 * w0)
            arr_full_cm = np.zeros((2,nLfull), dtype=np.double)
            arr_full_cm[0,:] = arr_full_hz[0,:] * fac1 # # d\nu -> d\lambda, symmetric
            arr_full_cm[1,:] = arr_full_hz[1,:] * fac2 # intensity_hz -> intensity_cm
            self.I_cm.append( arr_full_cm )
