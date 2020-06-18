
import numpy as np
from .. import Constants as Cst
from . import AtomIO
from ..RadiativeTransfer import Profile

class Atom:

    def __init__(self, _filepath, _file_Aji=None, _file_CEe=None, _file_CEp=None):
        r"""
        initial method of class Atom.

        Parameters
        ----------

        _filepath : str
            path (and filename) to config file *.Level

        _file_Aji : str
            path (and filename) to Aji data file *.Aji, default: None

        _file_CEe : str
            path (and filename) to Electron impact Effective Collisional Strength data file *.Electron, default: None

        _file_CEp : str
            path (and filename) to Proton impact Effective Collisional Strength data file *.Proton, default: None
        """
        self.filepath_dict = {
            "config" : _filepath,

        }
        self.__read_Level()
        self.__make_line_idx_ctj_table()

        # whether to read *.Aji file at __init__
        if _file_Aji is not None:
            self.read_Aji(_file_Aji)

        # whether to read *.Electron and *.Proton files at __init__
        if _file_CEe is not None:
            self.read_CE(_file_CEe, _file_CEp)


    def __read_Level(self):
        r"""
        read the Level information from config file *.Level
        """

        with open(self.filepath_dict["config"], 'r') as file:
            fLines = file.readlines()

        #--- read general info
        rs, self.Title, self.Z, self.Element, self.nLevel = AtomIO.read_general_info(_rs=0, _lns=fLines)
        self.nLine = self.nLevel * (self.nLevel-1) // 2
        self.nCont = 0

        #--- read Level info
        dtype  = np.dtype([
                          ('erg',np.double),            #: level energy, erg
                          ('g',np.uint8),               #: g=2J+1, statistical weight
                          ('stage',np.uint8),           #: ionization stage
                          ])
        self.Level = np.recarray(self.nLevel, dtype=dtype)
        self.Level_info = {"configuration" : [], "term" : [], "J": [], "2S+1": []}
        rs = AtomIO.read_level_info(rs, _lns=fLines, _Level_info=self.Level_info,
                            _erg=self.Level.erg[:], _g=self.Level.g[:], _stage=self.Level.stage[:])
        self.Level.erg[:] *= Cst.eV2erg_

        #--- make tuple of tuple (configuration, term, J)
        self.Level_info_table = []
        for k in range(self.nLevel):
            self.Level_info_table.append((self.Level_info["configuration"][k],
                                          self.Level_info["term"][k],
                                          self.Level_info["J"][k]))
        self.Level_info_table = tuple(self.Level_info_table)

    def __make_line_idx_ctj_table(self):
        r"""
        make a hash dictionary for mapping

        (ctj_i, ctj_j) --> (idxI, idxJ)
        """

        Line_idx_table = []
        Line_ctj_table = []
        for i in range(0, self.nLevel):
            for j in range(i+1, self.nLevel):
                # i : lower level
                # j : upper level
                Line_idx_table.append( ( i, j ) )
                Line_ctj_table.append( ( self.Level_info_table[i], self.Level_info_table[j] ) )

        self.Line_idx_table = tuple( Line_idx_table )
        self.Line_ctj_table = tuple( Line_ctj_table )

    def read_Aji(self, _path):
        r"""
        read Aji information from *.Aji

        Parameters
        ----------

        _path : str
            path to *.Aji data file
        """

        print("Reading Einstein Aji coefficient from : \n", _path)
        print("...")

        self.filepath_dict["Aji"] = _path
        with open(_path, 'r') as file:
            fLines = file.readlines()

        #--- read line info
        dtype = np.dtype([('idxI',np.uint16),           #: level index, the Level index of lower level
                           ('idxJ',np.uint16),          #: level index, the Level index of lower level
                           ('AJI',np.double),           #: Einstein Aji coefficient
                           ('f0',np.double),            #: central frequency
                           ('w0',np.double),            #: central wavelength in cm
                           ('w0_AA',np.double),         #: central wavelength in Angstrom
                           ("isContinuum",np.uint8),    #: continuum tansition identifier, 0: same stage, 1: continuum transition, 2: others
                           ])
        self.Line = np.recarray(self.nLine, dtype=dtype)
        # idxI and idxJ
        idx = 0
        for i in range(0, self.nLevel):
            for j in range(i+1, self.nLevel):
                self.Line.idxI[idx], self.Line.idxJ[idx] = i, j

                if self.Level.stage[i] == self.Level.stage[j]:
                    self.Line.isContinuum[idx] = 0
                elif self.Level.stage[i] == (self.Level.stage[j]-1):
                    self.Line.isContinuum[idx] = 1
                else:
                    self.Line.isContinuum[idx] = 2
                idx += 1
        del idx    # for safety
        self.Line.AJI[:] = 0

        AtomIO.read_line_info(_lns=fLines, _Aji=self.Line.AJI[:], _line_ctj_table=self.Line_ctj_table)

        # calculate f0, w0, w0_AA
        for k in range(self.nLine):
            i = self.Line.idxI[k]
            j = self.Line.idxJ[k]
            self.Line.f0[k] = (self.Level.erg[j]-self.Level.erg[i]) / Cst.h_
        self.Line.w0[:] = Cst.c_ / self.Line.f0[:]
        self.Line.w0_AA[:] = self.Line.w0[:] * 1E+8

        print("Finished.")
        print()

    def read_CE(self, _path_electron, _path_proton=None):
        r"""
        read Collisional Excitation table from *.Electron and *.Proton
        in most case the data is Effective Collisional Strength (ECS)

        Parameters
        ----------

        _path_electron : str
            path to *.Electron data file

        _path_proton : str
            path to *.proton
        """
        #---------------------------------------------------------------------
        # read Electron impact data
        #---------------------------------------------------------------------
        print("Reading Electron impact Effective Collisional Strength from : \n", _path_electron)
        print("...")

        self.filepath_dict["CE_electron"] = _path_electron
        with open(_path_electron, 'r') as file:
            fLines = file.readlines()

        # read Temperature grid for interpolation
        rs, nTe, Te, self.CE_type = AtomIO.read_CE_Temperature(_lns=fLines)
        self.CE_Te_table = np.array(Te, dtype=np.double)
        self.CE_table = np.zeros((self.nLine, nTe), dtype=np.double)
        dtype  = np.dtype([
                          ('idxI',np.uint8),      #: level index, the Level index of lower level
                          ('idxJ',np.uint8),      #: level index, the Level index of upper level
                          ('f1',np.uint8),        #: a factor for ESC calculation due to fine structure, \Omega * f1 / f2
                          ('f2',np.uint8),        #: a factor for ESC calculation due to fine structure, \Omega * f1 / f2
                          ('gi',np.uint8),        #: statistical weight of lower level
                          ('gj',np.uint8),        #: statistical weight of upper level
                          ('dEij',np.double)      #: excitation energy, [:math:`erg`]
                          ])

        self.CE_coe = np.recarray(self.nLine, dtype=dtype)

        # idxI and idxJ
        idx = 0
        for i in range(0, self.nLevel):
            for j in range(i+1, self.nLevel):
                self.CE_coe.idxI[idx], self.CE_coe.idxJ[idx] = i, j
                idx += 1
        del idx    # for safety

        # read CE_table
        AtomIO.read_CE_table(_rs=rs, _lns=fLines, _CE_table=self.CE_table,
                _f1=self.CE_coe.f1[:], _f2=self.CE_coe.f2[:], _line_ctj_table=self.Line_ctj_table)

        for k in range(self.nLine):
            self.CE_coe.gi[k] = self.Level.g[self.CE_coe.idxI[k]]
            self.CE_coe.gj[k] = self.Level.g[self.CE_coe.idxJ[k]]
            self.CE_coe.dEij[k] = self.Level.erg[self.CE_coe.idxJ[k]] - self.Level.erg[self.CE_coe.idxI[k]]

        print("Finished.")
        print()

        #---------------------------------------------------------------------
        # read Proton impact data (not yet imported)
        #---------------------------------------------------------------------
        if _path_proton is not None:

            print("Reading Proton impact Effective Collisional Strength from : \n", _path_proton)
            print("...")
            print("Finished.")
            print()

            self.filepath_dict["CE_proton"] = _path_proton
            with open(_path_proton, 'r') as file:
                fLines = file.readlines()

            pass

    def read_CI(self, _path_electron, _path_proton=None):
        r"""
        read Collisional Ionization table from *.Electron and *.Proton

        Parameters
        ----------

        _path_electron : str
            path to *.Electron data file

        _path_proton : str
            path to *.proton
        """
        #---------------------------------------------------------------------
        # read Electron impact data
        #---------------------------------------------------------------------
        print("Reading Electron impact Collisional Ionization coefficient from : \n", _path_electron)
        print("...")

        self.filepath_dict["CI_electron"] = _path_electron
        with open(_path_electron, 'r') as file:
            fLines = file.readlines()

        # read Temperature grid for interpolation
        self.nCont, rs, nTe, Te = AtomIO.read_CI_Temperature(_lns=fLines)
        self.CI_Te_table = np.array(Te, dtype=np.double)
        self.CI_table = np.zeros((self.nCont, nTe), dtype=np.double)
        dtype  = np.dtype([
                          ('idxI',np.uint8),      #: level index, the Level index of lower level
                          ('idxJ',np.uint8),      #: level index, the Level index of upper level
                          ('lineIndex',np.uint16),#: line index
                          #('f1',np.uint8),        #: a factor for ESC calculation due to fine structure, \Omega * f1 / f2
                          ('f2',np.uint8),        #: a factor for ESC calculation due to fine structure, \Omega * f1 / f2
                          ('gi',np.uint8),        #: statistical weight of lower level
                          ('gj',np.uint8),        #: statistical weight of upper level
                          ('dEij',np.double)      #: ionization limit, [:math:`erg`]
                          ])

        self.CI_coe = np.recarray(self.nCont, dtype=dtype)

        # read CE_table
        AtomIO.read_CI_table(_rs=rs, _lns=fLines,
                        _CI_table=self.CI_table[:,:], _f2=self.CI_coe.f2[:],
                        _idxI=self.CI_coe.idxI[:],_idxJ=self.CI_coe.idxJ[:],
                        _lineIndex = self.CI_coe.lineIndex[:],
                        _level_info_table=self.Level_info_table,
                        _line_ctj_table=self.Line_ctj_table)

        for k in range(self.nCont):
            self.CI_coe.gi[k] = self.Level.g[self.CI_coe.idxI[k]]
            self.CI_coe.gj[k] = self.Level.g[self.CI_coe.idxJ[k]]
            self.CI_coe.dEij[k] = self.Level.erg[self.CI_coe.idxJ[k]] - self.Level.erg[self.CI_coe.idxI[k]]

        print("Finished.")
        print()

        #---------------------------------------------------------------------
        # read Proton impact data (not yet imported)
        #---------------------------------------------------------------------
        pass

    def read_PI(self, _path_alpha):
        r"""
        read Collisional Ionization table from *.Alpha

        Parameters
        ----------

        _path_alpha: str
            path to *.Alpha data file

        """

        print("Reading Photoionization cross section from : \n", _path_alpha)
        print("...")

        self.filepath_dict["Photoionization"] = _path_alpha
        with open(_path_alpha, 'r') as file:
            fLines = file.readlines()

        nCont, rs, nMesh = AtomIO.read_PI_Info(_lns=fLines)
        #assert nCont == self.nCont
        self.PI_table = np.zeros((nCont, nMesh, 2), dtype=np.double)
        # change this to numba.types.List in the future for inhomogenious array

        dtype  = np.dtype([
                          ('idxI',np.uint8),      #: level index, the Level index of lower level
                          ('idxJ',np.uint8),      #: level index, the Level index of upper level
                          ('lineIndex',np.uint16),#: line index
                          ('nLambda', np.uint16), #: number of meaningful wavelength mesh point
                          ('alpha0', np.double),  #: Photoionization cross section at frequency edge
                          ('gi',np.uint8),        #: statistical weight of lower level
                          ('gj',np.uint8),        #: statistical weight of upper level
                          ('dEij',np.double)      #: ionization limit, [:math:`erg`]
                          ])
        self.PI_coe = np.recarray(nCont, dtype=dtype)

        AtomIO.read_PI_table(_rs=rs, _lns=fLines,
                        _PI_table = self.PI_table[:,:,:],
                        _PI_coe = self.PI_coe,
                        _level_info_table=self.Level_info_table,
                        _line_ctj_table=self.Line_ctj_table)
        self.PI_table[:,:,0] *= 1E-7 # nm --> cm   (cm, cm^2)


        for k in range(nCont):
            self.PI_coe.gi[k]   = self.Level.g[self.PI_coe.idxI[k]]
            self.PI_coe.gj[k]   = self.Level.g[self.PI_coe.idxJ[k]]
            self.PI_coe.dEij[k] = self.Level.erg[self.PI_coe.idxJ[k]] - self.Level.erg[self.PI_coe.idxI[k]]

            # shift edge wavelength to the computed wavelength w0
            self.PI_table[k,:self.PI_coe.nLambda[k],0] += self.Line.w0[self.PI_coe.lineIndex[k]] - self.PI_table[k,0,0]

        print("Finished.")
        print()

    def read_Mesh(self, _path):
        r"""
        read wavelength mesh information from *.RadiativeLine

        Parameters
        ----------

        _path: str
            path to *.RadiativeLine data file

        """

        print("Reading wavelength mesh information from : \n", _path)
        print("...")

        self.filepath_dict["RadiativeLine"] = _path
        with open(_path, 'r') as file:
            fLines = file.readlines()

        self.nRadiativeLine, rs = AtomIO.read_Radiative_Line_Number(_lns=fLines)

        dtype  = np.dtype([
                          ('idxI',np.uint8),      #: level index, the Level index of lower level
                          ('idxJ',np.uint8),      #: level index, the Level index of upper level
                          ('lineIndex',np.uint16),#: line index
                          ('ProfileType',np.uint8),#: 0: Voigt; 1:...
                          ('qcore', np.double),
                          ('qwing', np.double),
                          ('nLambda', np.uint16), #: number of meaningful wavelength mesh point
                          ])
        self.Mesh_coe = np.recarray(self.nRadiativeLine, dtype=dtype)
        self.RadiativeLine_filename = []

        AtomIO.read_Mesh_Info(_rs=rs, _lns=fLines,
                        _Mesh_coe = self.Mesh_coe,
                        _filename = self.RadiativeLine_filename,
                        _level_info_table=self.Level_info_table,
                        _line_ctj_table=self.Line_ctj_table)


    def make_Mesh(self):

        #--- make line mesh
        self.line_mesh_list = []

        for k in range(self.nRadiativeLine):
            nLambda = self.Mesh_coe.nLambda[k]
            qcore = self.Mesh_coe.qcore[k]
            qwing = self.Mesh_coe.qwing[k]
            mesh = Profile.makeLineMesh_Full(nLambda, qcore, qwing) # in Doppler width unit

            self.line_mesh_list.append( mesh )

        print("line mesh prepared.")

        #--- make continuum mesh
        self.continuum_mesh_list = []
        for k in range(self.nCont):
            mesh = Profile.makeContinuumMesh(21) # in limit wavelength unit
            w0 = self.Line.w0[self.CI_coe.lineIndex[k]]
            self.continuum_mesh_list.append( mesh * w0 )

        print("continuum mesh prepared.")

    def read_Radiative_Line_intensity(self, _folder):

        self.radiative_line_intensity_list = []

        for k in range(self.nRadiativeLine):
            filename = self.RadiativeLine_filename[k]
            arr = AtomIO.read_half_intensity_hz(_folder+'/'+filename)
            f0 = self.Line.f0[self.Mesh_coe.lineIndex[k]]
            w0 = self.Line.w0[self.Mesh_coe.lineIndex[k]]
            fac1 = Cst.c_ / (f0 * f0)
            fac2 = Cst.c_ / (w0 * w0)
            arr[:,0] *= fac1 # d\nu -> d\lambda
            arr[:,1] *= fac2 # intensity_hz -> intensity_cm
            nLmid = arr.shape[0]
            nLfull = (nLmid-1) * 2 + 1
            arr_full = np.zeros((2,nLfull), dtype=np.double)
            arr_full[0, nLmid:] = arr[1:,0]
            arr_full[0,:nLmid] = -arr[::-1,0]
            arr_full[1, nLmid:] = arr[1:,1]
            arr_full[1,:nLmid] = arr[::-1,1]

            self.radiative_line_intensity_list.append( arr_full )




    def ctj_to_level_idx(self, ctj):
        r"""
        ctj --> idx
        """

        return self.Level_info_table.index(ctj)

    def level_idx_to_ctj(self, idx):
        r"""
        idx --> ctj
        """

        return self.Level_info_table[idx]

    def line_ctj_to_line_idx(self, line_ctj):
        r"""
        (ctj_i, ctj_j) --> (idxI, idxJ)
        """

        return self.Line_idx_table[ self.Line_ctj_table.index( line_ctj ) ]

    def line_idx_to_line_ctj(self, line_idx):
        r"""
        (idxI, idxJ) --> (ctj_i, ctj_j)
        """

        return self.Line_ctj_table[ self.Line_idx_table.index( line_idx ) ]

    def line_index_to_line_ctj(self, _index):
        r"""
        line index (line No.) --> (ctj_i, ctj_j)
        """

        return self.Line_ctj_table[_index]

    def line_ctj_to_line_index(self, line_ctj):
        r"""
        (ctj_i, ctj_j) --> line index (line No.)
        """

        return self.Line_ctj_table.index(line_ctj)

    def line_index_to_line_idx(self, _index):
        r"""
        line index (line No.) --> (idxI, idxJ)
        """

        return self.Line_idx_table[_index]

    def line_idx_to_line_index(self, line_idx):
        r"""
        (idxI, idxJ) --> line index (line No.)
        """

        return self.Line_idx_table.index(line_idx)

    def conf_to_line_idx(self, conf_lower, conf_upper):
        r"""
        given the configuration tuple of lower and upper level, respectively,
        return the index of that transition.

        Warning :
        this method is left from the Atom() class before the merge
        of combining with AtomicQuery.
        use `self.line_ctj_to_line_index()` instead
        """
        _line_ctj = ( conf_lower, conf_upper )
        return self.line_ctj_to_line_index( _line_ctj )
