import numpy as np
from .. import Constants as Cst
from . import AtomIO

from ..RadiativeTransfer import Profile

class WavelengthMesh:

    def __init__(self, _parent, _isPrint=False):
        r"""
        """
        self.isPrint = _isPrint
        self._parent = _parent

    def make_Line_Mesh(self, _path):
        r"""
        """

        _parent = self._parent

        dtype  = np.dtype([
                          ('idxI',np.uint8),      #: level index, the Level index of lower level
                          ('idxJ',np.uint8),      #: level index, the Level index of upper level
                          ('lineIndex',np.uint16),#: line index
                          ('ProfileType',np.uint8),#: 0: Voigt; 1:...
                          ('qcore', np.double),
                          ('qwing', np.double),
                          ('nLambda', np.uint16), #: number of meaningful wavelength mesh point
                          ])
        if _path is None:
            self.nRadiativeLine = 0
            self.Coe = np.recarray(self.nRadiativeLine, dtype=dtype)
        else:
            with open(_path, 'r') as file:
                fLines = file.readlines()

            self.nRadiativeLine, rs = AtomIO.read_Radiative_Line_Number(_lns=fLines)

            self.Coe = np.recarray(self.nRadiativeLine, dtype=dtype)

            self.RadLine_filenames = []
            AtomIO.read_Mesh_Info(_rs=rs, _lns=fLines,
                            _Mesh_coe = self.Coe,
                            _filename = self.RadLine_filenames,
                            _level_info_table=_parent._Level_info_table,
                            _line_ctj_table=_parent.Line_ctj_table)


        _parent.nRadiativeLine = self.nRadiativeLine


        #--- make line mesh
        ##self.RadLine = [] # List()
        ##for k in range(self.nRadiativeLine):
        ##    nLambda = self.Coe.nLambda[k]
        ##    qcore = self.Coe.qcore[k]
        ##    qwing = self.Coe.qwing[k]
        ##    mesh = Profile.makeLineMesh_Full(nLambda, qcore, qwing) # in Doppler width unit
        ##    self.RadLine.append( mesh )

        if self.isPrint:
            print("line mesh prepared.")

    def make_Cont_Mesh(self):
        r"""
        """

        #--- make continuum mesh
        if self._parent.hasContinuum:
            self.Cont = [] # List()
            for k in range(self._parent.nCont):
                mesh = Profile.makeContinuumMesh(41) # in limit wavelength unit
                w0 = self._parent.Cont.w0[k]
                self.Cont.append( mesh * w0 )
            self.Cont = np.array( self.Cont, dtype=np.double )

            if self.isPrint:
                print("continuum mesh prepared.")
