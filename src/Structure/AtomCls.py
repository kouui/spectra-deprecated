import numpy as np
from .. import Constants as Cst
from .. import Element
from . import AtomIO
from . import MeshCls
from . import RadLineCls

from collections import OrderedDict

#from numba.typed import List


class Atom:


    def __init__(self, _filepath, _file_Aji=None, _file_CEe=None, _file_CEp=None, _isPrint=False):
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

        self.isPrint = _isPrint

        self.filepath_dict = {
            "config" : _filepath,

        }
        self.read_Level()
        self.get_nTran_nLine_nCont()
        self._prepare_idx_ctj_mapping()

        if self.hasContinuum:
            self.make_Cont()

        # whether to read *.Aji file at __init__
        if _file_Aji is not None:
            self.read_Aji(_file_Aji)

        # whether to read *.Electron and *.Proton files at __init__
        if _file_CEe is not None:
            self.read_CE(_file_CEe, _file_CEp)

    def read_Level(self):
        r"""
        read the Level information from config file *.Level

        """

        with open(self.filepath_dict["config"], 'r') as file:
            fLines = file.readlines()

        #--- read general info
        rs, self.Title, self.Z, self.Element, self.nLevel = AtomIO.read_general_info(_rs=0, _lns=fLines)
        self.Z = int(self.Z)
        self.Mass = Element.Element_dict[self.Element]["Mass"]
        self.Abun = 10**(Element.Element_dict[self.Element]["Abundance"]-12.0)

        #--- read Level info
        dtype  = np.dtype([
                          ('erg',np.double),            #: level energy, erg
                          ('g',np.uint8),               #: g=2J+1, statistical weight
                          ('stage',np.uint8),           #: ionization stage
                          ('gamma',np.double),          #: radiative damping constant of Level
                          ("isGround",np.bool),         #: whether a level is ground level
                          ])
        self.Level = np.recarray(self.nLevel, dtype=dtype)
        self._Level_info = {"configuration" : [], "term" : [], "J": [], "2S+1": []}
        rs = AtomIO.read_level_info(rs, _lns=fLines, _Level_info=self._Level_info,
                            _erg=self.Level.erg[:], _g=self.Level.g[:], _stage=self.Level.stage[:])
        self.Level.erg[:] *= Cst.eV2erg_

        self.Level.isGround[:] = 1
        for k in range(1,self.nLevel):
            if self.Level.stage[k] == self.Level.stage[k-1]:
                self.Level.isGround[k] = 0

        self.Level.gamma[:] = 0

        #--- make tuple of tuple (configuration, term, J)
        self._Level_info_table = []
        for k in range(self.nLevel):
            self._Level_info_table.append((self._Level_info["configuration"][k],
                                          self._Level_info["term"][k],
                                          self._Level_info["J"][k]))
        self._Level_info_table = tuple(self._Level_info_table)
        self.Level_ctj_table = self._Level_info_table

    def get_nTran_nLine_nCont(self):
        r"""
        based on self.Level,
        compute self.nTran, self.nLine, self.nCont

        """
        _stage = self.Level.stage
        count = 0
        current_stage = _stage[0]
        counter = OrderedDict()
        for k,s in enumerate(_stage):
            if s == current_stage:
                count += 1
                if k == self.nLevel-1:
                    counter[current_stage] = count
                    break
            elif s > current_stage:
                counter[current_stage] = count
                count = 1
                current_stage = s
                counter[current_stage] = count

        nLine, nCont = 0, 0
        for k, v in counter.items():
            nLine += v * (v-1) // 2
            if k+1 in counter.keys():
                nCont += v
        nTran = nLine + nCont

        self.nTran = nTran
        self.nLine = nLine
        self.nCont = nCont

        if self.nCont > 0:
            self.hasContinuum = True

    def _idx_ctj_into_dict(self, _dict, _i, _j):
        r"""
        """
        _dict["idxI"].append( _i )
        _dict["idxJ"].append( _j )
        _dict["ctj_i"].append( self._Level_info_table[_i] )
        _dict["ctj_j"].append( self._Level_info_table[_j] )

    def _prepare_idx_ctj_mapping(self):
        r"""
        make tuples for mapping

        lineIndex <--> (ctj_i, ctj_j)
        lineIndex <--> (idxI, idxJ)
        contIndex <--> (ctj_i, ctj_j)
        contIndex <--> (idxI, idxJ)

        """
        _Tran_dict = {}
        _Line_dict = {}
        _Cont_dict = {}
        for key in ("idxI", "idxJ", "ctj_i", "ctj_j"):
            _Tran_dict[key] = []
            _Line_dict[key] = []
            _Cont_dict[key] = []

        for i in range(0, self.nLevel):
            for j in range(i+1, self.nLevel):
                # i : lower level
                # j : upper level
                if self.Level.stage[i] == self.Level.stage[j]:
                    self._idx_ctj_into_dict(_Tran_dict, i, j)
                    self._idx_ctj_into_dict(_Line_dict, i, j)
                elif self.Level.stage[i] == self.Level.stage[j]-1 and self.Level.isGround[j]:
                    self._idx_ctj_into_dict(_Tran_dict, i, j)
                    self._idx_ctj_into_dict(_Cont_dict, i, j)

        assert self.nLine == len(_Line_dict["idxI"])
        assert self.nCont == len(_Cont_dict["idxI"])

        self._Tran_dict = _Tran_dict
        self._Line_dict = _Line_dict
        self._Cont_dict = _Cont_dict

        Line_idx_table = []
        Line_ctj_table = []
        for k in range(self.nLine):
            Line_idx_table.append( ( _Line_dict["idxI"][k], _Line_dict["idxJ"][k] ) )
            Line_ctj_table.append( ( _Line_dict["ctj_i"][k], _Line_dict["ctj_j"][k] ) )
        self.Line_idx_table = tuple( Line_idx_table )
        self.Line_ctj_table = tuple( Line_ctj_table )

        Cont_idx_table = []
        Cont_ctj_table = []
        for k in range(self.nCont):
            Cont_idx_table.append( ( _Cont_dict["idxI"][k], _Cont_dict["idxJ"][k] ) )
            Cont_ctj_table.append( ( _Cont_dict["ctj_i"][k], _Cont_dict["ctj_j"][k] ) )
        self.Cont_idx_table = tuple( Cont_idx_table )
        self.Cont_ctj_table = tuple( Cont_ctj_table )

    def make_Cont(self):
        r"""
        """
        dtype = np.dtype([('idxI',np.uint16),           #: level index, the Level index of lower level
                           ('idxJ',np.uint16),          #: level index, the Level index of lower level
                           ('f0',np.double),            #: central frequency
                           ('w0',np.double),            #: central wavelength in cm
                           ('w0_AA',np.double),         #: central wavelength in Angstrom
                           ])
        self.Cont = np.recarray(self.nCont, dtype=dtype)


        for k in range(self.nCont):
            i, j = self.Cont_idx_table[k]
            self.Cont.idxI[k], self.Cont.idxJ[k] = i, j
            self.Cont.f0[k] = (self.Level.erg[j]-self.Level.erg[i]) / Cst.h_
        self.Cont.w0[:] = Cst.c_ / self.Cont.f0[:]
        self.Cont.w0_AA[:] = self.Cont.w0[:] * 1E+8


    def read_Aji(self, _path):
        r"""
        read Aji information from *.Aji

        Parameters
        ----------

        _path : str
            path to *.Aji data file
        """

        if self.isPrint:
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
                           #("isContinuum",np.uint8),    #: continuum tansition identifier, 0: same stage, 1: continuum transition, 2: others
                           ('Gamma',np.double),         #: radiative damping constant of Line
                           ])
        self.Line = np.recarray(self.nLine, dtype=dtype)

        # idxI and idxJ
        for k in range(self.nLine):
            self.Line.idxI[k], self.Line.idxJ[k] = self.Line_idx_table[k]

        self.Line.AJI[:] = 0

        AtomIO.read_line_info(_lns=fLines, _Aji=self.Line.AJI[:], _line_ctj_table=self.Line_ctj_table)

        # calculate f0, w0, w0_AA
        for k in range(self.nLine):
            i = self.Line.idxI[k]
            j = self.Line.idxJ[k]
            self.Line.f0[k] = (self.Level.erg[j]-self.Level.erg[i]) / Cst.h_
        self.Line.w0[:] = Cst.c_ / self.Line.f0[:]
        self.Line.w0_AA[:] = self.Line.w0[:] * 1E+8

        self.Line.Gamma[:] = 0

        if self.isPrint:
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

        self.CE = Collisional_Transition(_parent=self, _path_electron=_path_electron, _type="CE", _isPrint=self.isPrint)

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

        self.CI = Collisional_Transition(_parent=self, _path_electron=_path_electron, _type="CI", _isPrint=self.isPrint)

    def read_PI(self, _path_alpha):
        r"""
        read Collisional Ionization table from *.Alpha

        Parameters
        ----------

        _path_alpha: str
            path to *.Alpha data file

        """

        if self.isPrint:
            print("Reading Photoionization cross section from : \n", _path_alpha)

        self.filepath_dict["Photoionization"] = _path_alpha
        self.PI = Photoionization(_parent=self, _path_alpha=_path_alpha, _isPrint=self.isPrint)

    def read_RadiativeLine_and_make_Line_Mesh(self, _path):
        r"""
        read wavelength mesh information from *.RadiativeLine
        and make Mesh for RadiativeLine and Cont

        Parameters
        ----------

        _path: str
            path to *.RadiativeLine data file

        """

        if self.isPrint:
            print("Reading Radiative Line information from : \n", _path)

        self.filepath_dict["RadiativeLine"] = _path
        self.Mesh = MeshCls.WavelengthMesh(_parent=self, _isPrint=self.isPrint)
        self.Mesh.make_Line_Mesh(_path=_path)

    def make_Cont_Mesh(self):
        r"""
        """
        if not hasattr(self, 'Mesh'):
            self.Mesh = MeshCls.WavelengthMesh(_parent=self)
        self.Mesh.make_Cont_Mesh()

    def read_RadLine_intensity(self, _folder):
        r"""
        read line profiles of incident radiation for
        specific line transition

        """
        self.I_Rad = RadLineCls.RadiativeLine(_parent=self, _folder=_folder)






    def ctj_to_level_idx(self, ctj):
        r"""
        ctj --> idx

        """

        return self._Level_info_table.index(ctj)

    def level_idx_to_ctj(self, idx):
        r"""
        idx --> ctj

        """

        return self._Level_info_table[idx]

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





    def cont_ctj_to_cont_idx(self, cont_ctj):
        r"""
        (ctj_i, ctj_j) --> (idxI, idxJ)

        """

        return self.Cont_idx_table[ self.Cont_ctj_table.index( cont_ctj ) ]

    def cont_idx_to_cont_ctj(self, cont_idx):
        r"""
        (idxI, idxJ) --> (ctj_i, ctj_j)

        """

        return self.Cont_ctj_table[ self.Cont_idx_table.index( cont_idx ) ]

    def cont_index_to_cont_ctj(self, _index):
        r"""
        line index (line No.) --> (ctj_i, ctj_j)

        """

        return self.Cont_ctj_table[_index]

    def cont_ctj_to_cont_index(self, cont_ctj):
        r"""
        (ctj_i, ctj_j) --> cont index (cont No.)

        """

        return self.Cont_ctj_table.index(cont_ctj)

    def cont_index_to_cont_idx(self, _index):
        r"""
        cont index (cont No.) --> (idxI, idxJ)

        """

        return self.Cont_idx_table[_index]

    def cont_idx_to_cont_index(self, cont_idx):
        r"""
        (idxI, idxJ) --> cont index (cont No.)

        """

        return self.Cont_idx_table.index(cont_idx)





class Collisional_Transition:


    def __init__(self, _parent, _path_electron, _path_proton=None, _type="", _isPrint=False):

        self.isPrint = _isPrint

        assert _type in ("CE", "CI")
        self._parent = _parent

        #---------------------------------------------------------------------
        # read Electron impact data
        #---------------------------------------------------------------------
        if _type == "CE":
            if self.isPrint:
                print("Reading Electron impact Effective Collisional Strength from : \n", _path_electron)
            _parent.filepath_dict["CE_electron"] = _path_electron
            N = _parent.nLine
        elif _type == "CI":
            if self.isPrint:
                print("Reading Electron impact Collisional Ionization coefficient from : \n", _path_electron)
            _parent.filepath_dict["CI_electron"] = _path_electron
            N = _parent.nCont

        with open(_path_electron, 'r') as file:
            fLines = file.readlines()

        # read Temperature grid for interpolation
        if _type == "CE":
            rs, nTe, Te, self.CE_type = AtomIO.read_CE_Temperature(_lns=fLines)
        elif _type == "CI":
            rs, nTe, Te = AtomIO.read_CI_Temperature(_lns=fLines)

        self.Te_table = np.array(Te, dtype=np.double)
        self.Omega_table = np.zeros((N, nTe), dtype=np.double)
        dtype  = np.dtype([
                          ('idxI',np.uint8),      #: level index, the Level index of lower level
                          ('idxJ',np.uint8),      #: level index, the Level index of upper level
                          ('f1',np.uint8),        #: a factor for ESC calculation due to fine structure, \Omega * f1 / f2
                          ('f2',np.uint8),        #: a factor for ESC calculation due to fine structure, \Omega * f1 / f2
                          ('gi',np.uint8),        #: statistical weight of lower level
                          ('gj',np.uint8),        #: statistical weight of upper level
                          ('dEij',np.double)      #: excitation energy, [:math:`erg`]
                          ])
        self.Coe = np.recarray(N, dtype=dtype)

        if _type == "CE":
            AtomIO.read_CE_table(_rs=rs, _lns=fLines, _CE_table=self.Omega_table,
                    _idxI=self.Coe.idxI[:],_idxJ=self.Coe.idxJ[:],
                    _f1=self.Coe.f1[:], _f2=self.Coe.f2[:],
                    _level_info_table=_parent._Level_info_table,
                    _line_ctj_table=_parent.Line_ctj_table)

        elif _type == "CI":
            AtomIO.read_CI_table(_rs=rs, _lns=fLines, _CI_table=self.Omega_table[:,:],
                            _f2=self.Coe.f2[:],
                            _idxI=self.Coe.idxI[:],_idxJ=self.Coe.idxJ[:],
                            _level_info_table=_parent._Level_info_table,
                            _cont_ctj_table=_parent.Cont_ctj_table)
            self.Coe.f1[:] = 0

        for k in range(N):
            self.Coe.gi[k] = _parent.Level.g[self.Coe.idxI[k]]
            self.Coe.gj[k] = _parent.Level.g[self.Coe.idxJ[k]]
            self.Coe.dEij[k] = _parent.Level.erg[self.Coe.idxJ[k]] - _parent.Level.erg[self.Coe.idxI[k]]

        if self.isPrint:
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

class Photoionization:

    def __init__(self, _parent, _path_alpha, _isPrint=False):
        r"""
        """

        self.isPrint = _isPrint

        with open(_path_alpha, 'r') as file:
            fLines = file.readlines()

        N = _parent.nCont

        nCont, rs = AtomIO.read_PI_Info(_lns=fLines)
        alpha_table_dict = OrderedDict()

        dtype  = np.dtype([
                          ('idxI',np.uint8),      #: level index, the Level index of lower level
                          ('idxJ',np.uint8),      #: level index, the Level index of upper level
                          #('lineIndex',np.uint16),#: line index
                          ('nLambda', np.uint16), #: number of meaningful wavelength mesh point
                          ('alpha0', np.double),  #: Photoionization cross section at frequency edge
                          ('gi',np.uint8),        #: statistical weight of lower level
                          ('gj',np.uint8),        #: statistical weight of upper level
                          ('dEij',np.double)      #: ionization limit, [:math:`erg`]
                          ])
        self.Coe = np.recarray(N, dtype=dtype)

        AtomIO.read_PI_table(_rs=rs, _lns=fLines,
                        _PI_table_dict = alpha_table_dict,
                        _PI_coe = self.Coe,
                        _level_info_table=_parent._Level_info_table,
                        _cont_ctj_table=_parent.Cont_ctj_table)

        self.alpha_table = []
        sorted_keys = sorted( list(alpha_table_dict.keys()) )
        for key in sorted_keys:
            self.alpha_table.append(  alpha_table_dict[key]  )


        #self.PI_table[:,:,0] *= 1E-7 # nm --> cm   (cm, cm^2)
        for mesh_arr in self.alpha_table:
            mesh_arr[0,:] *= 1E-7 # nm --> cm   (cm, cm^2)


        for k in range(N):
            self.Coe.gi[k]   = _parent.Level.g[self.Coe.idxI[k]]
            self.Coe.gj[k]   = _parent.Level.g[self.Coe.idxJ[k]]
            self.Coe.dEij[k] = _parent.Level.erg[self.Coe.idxJ[k]] - _parent.Level.erg[self.Coe.idxI[k]]

            # shift edge wavelength to the computed wavelength w0
            #self.PI_table[k,:self.PI_coe.nLambda[k],0] += self.Line.w0[self.PI_coe.lineIndex[k]] - self.PI_table[k,0,0]
            self.alpha_table[k][0,:] += _parent.Cont.w0[k] - self.alpha_table[k][0,0]

        if self.isPrint:
            print("Finished.")
            print()
