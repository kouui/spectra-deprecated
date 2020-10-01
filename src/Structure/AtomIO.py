import numpy as np

def skip_line(_ln):
    r"""
    skip
        1. comment line. start with '#'
        2. empty line. _ln.strip() is ''
    """
    if _ln[0] == "#" or _ln.strip() == '':
        return True

    return False

def check_end(_ln):
    r"""
    end data reading
        1. _ln starts with "END" end data reading
    """
    if _ln[:3].upper() == "END":
        return True

    return False


def read_general_info(_rs, _lns):
    r"""
    read general information in *.Level config file
        1. Title
        2. Z
        3. Element
    """

    for _i, _ln in enumerate(_lns[_rs:]):

        if skip_line(_ln):
            continue
        if check_end(_ln):
            break

        if _ln.split()[0].strip().lower() == "title:":
            _Title = ' '.join( _ln.split()[1:] )

        if _ln.split()[0].strip().lower() == "z":
            _Z = _ln.split()[-1].strip()

        if _ln.split()[0].strip().lower() == "element":
            _Element = _ln.split()[-1].strip()

        if _ln.split()[0].strip().lower() == "nlevel":
            _nLevel = int( _ln.split()[-1].strip() )

    _re = _rs + _i + 1

    return _re, _Title, _Z, _Element, _nLevel

def read_level_info(_rs, _lns, _Level_info, _erg, _g, _stage):
    r"""
    read level information to
        1. _Level_info
            - ["configuration"]
            - ["term"]
            - ["J"]
            - ["2S+1"]
        2. _erg
        3. _g
        4. _stage
    """

    _idx = 0
    #_prefix = ''
    for _i, _ln in enumerate(_lns[_rs:]):

        if skip_line(_ln):
            continue
        elif check_end(_ln):
            break

        _words = _ln.split()
        _words = [_v.strip() for _v in _words]

        if _words[0] == "prefix":
            _prefix = _words[1] if _words[1] != '-' else ''
            continue

        if _words[0] == '-' and _prefix[-1] == '.':
            _Level_info["configuration"].append( _prefix[:-1] )
        else:
            _Level_info["configuration"].append( _prefix+_words[0] )

        _Level_info["term"].append( _words[1] )
        _Level_info["J"].append( _words[2] )
        _Level_info["2S+1"].append( _words[5] )

        _erg[_idx] = float( _words[8] )
        _g[_idx] = int( _words[6] )
        _stage[_idx] = int( _words[7] )

        _idx += 1

    _re = _rs + _i + 1

    return _re

def read_line_info(_lns, _Aji, _line_ctj_table):
    r"""
    read line information
    """
    #_count = 0
    #_prefix = ''
    for _i, _ln in enumerate(_lns[:]):

        if skip_line(_ln):
            continue
        elif check_end(_ln):
            break

        _words = _ln.split()
        _words = [_v.strip() for _v in _words]

        if _words[0] == "prefix":
            _prefix = _words[1] if _words[1] != '-' else ''
            continue

        # get ctj pair
        ctj_ij = ( (_prefix+_words[0],_words[1],_words[2]), (_prefix+_words[3],_words[4],_words[5]) )

        if ctj_ij in _line_ctj_table:
            line_index = _line_ctj_table.index( ctj_ij )
            _Aji[line_index] += float( _words[6] )
            #_count += 1

    return None

def read_CE_Temperature(_lns):
    r"""
    read Temperature grid for interpolation
    """
    for _i, _ln in enumerate(_lns[:]):

        if skip_line(_ln):
            continue
        elif check_end(_ln):
            break

        _words = _ln.split()
        _words = [_v.strip() for _v in _words]

        if _words[0].lower() == "type":
            _type = _words[1]

        if _words[0].lower() == "temperature":
            _Te = [float(v) for v in _words[1:]]

    _nTe = len( _Te )
    _re = _i + 1

    return _re, _nTe, _Te, _type

def read_CE_table(_rs, _lns, _CE_table, _idxI, _idxJ, _f1, _f2, _level_info_table, _line_ctj_table):
    r"""
    read CE table for interpolation
    """
    #_count = 0
    #_prefix = ''
    for _i, _ln in enumerate(_lns[_rs:]):

        if skip_line(_ln):
            continue
        elif check_end(_ln):
            break

        _words = _ln.split()
        _words = [_v.strip() for _v in _words]

        if _words[0] == "prefix":
            _prefix = _words[1] if _words[1] != '-' else ''
            continue

        # get ctj pair
        _ctj_ij = ( (_prefix+_words[0],_words[1],_words[2]), (_prefix+_words[3],_words[4],_words[5]) )

        if _ctj_ij in _line_ctj_table:
            line_index = _line_ctj_table.index( _ctj_ij )
            _idxI[line_index] = _level_info_table.index( _ctj_ij[0] )
            _idxJ[line_index] = _level_info_table.index( _ctj_ij[1] )
            _CE_table[line_index,:] += [float(v) for v in _words[6:-2]]
            _f1[line_index] = float(_words[-2])
            _f2[line_index] = float(_words[-1])

            #_count += 1

    return None

def read_CI_Temperature(_lns):
    r"""
    read Temperature grid for interpolation
    """
    for _i, _ln in enumerate(_lns[:]):

        if skip_line(_ln):
            continue
        elif check_end(_ln):
            break

        _words = _ln.split()
        _words = [_v.strip() for _v in _words]

        #if _words[0].lower() == "ncont":
        #    _nCont = int( _words[1] )

        if _words[0].lower() == "temperature":
            _Te = [float(_v) for _v in _words[1:]]

    _nTe = len( _Te )
    _re = _i + 1

    return _re, _nTe, _Te

def read_CI_table(_rs, _lns, _CI_table, _f2, _idxI, _idxJ, _level_info_table, _cont_ctj_table):
    r"""
    read CI table for interpolation
    """
    #_count = 0
    #_prefix = ''
    for _i, _ln in enumerate(_lns[_rs:]):

        if skip_line(_ln):
            continue
        elif check_end(_ln):
            break

        _words = _ln.split()
        _words = [_v.strip() for _v in _words]

        if _words[0] == "prefix":
            _prefix = _words[1] if _words[1] != '-' else ''
            continue

        # get ctj pair
        if _words[3] == '-' and _prefix[-1] == '.':
            _ctj_ij = ( (_prefix+_words[0],_words[1],_words[2]), (_prefix[:-1],_words[4],_words[5]) )
        else:
            _ctj_ij = ( (_prefix+_words[0],_words[1],_words[2]), (_prefix+_words[3],_words[4],_words[5]) )

        if _ctj_ij in _cont_ctj_table:
            contIndex = _cont_ctj_table.index( _ctj_ij )
            _idxI[contIndex] = _level_info_table.index( _ctj_ij[0] )
            _idxJ[contIndex] = _level_info_table.index( _ctj_ij[1] )
            _CI_table[contIndex,:] += [float(v) for v in _words[6:-1]]
            _f2[contIndex] = float(_words[-1])

        #_count += 1

    return None

def read_PI_Info(_lns):
    r"""
    read nCont and nMesh for Photoionization
    """
    for _i, _ln in enumerate(_lns[:]):

        if skip_line(_ln):
            continue
        elif check_end(_ln):
            break

        _words = _ln.split()
        _words = [_v.strip() for _v in _words]

        if _words[0].lower() == "ncont":
            _nCont = int( _words[1] )

        #if _words[0].lower() == "nmesh":
        #    _nMesh = int( _words[1] )

    _re = _i + 1

    return _nCont, _re#, _nMesh

def read_PI_table(_rs, _lns, _PI_table_dict, _PI_coe, _level_info_table, _cont_ctj_table):
    r"""
    read PI table for interpolation
    """
    _countMesh = 0
    _readMesh = False

    #_prefix = ''
    for _i, _ln in enumerate(_lns[_rs:]):

        if skip_line(_ln):
            continue
        elif check_end(_ln):
            break

        _words = _ln.split()
        _words = [_v.strip() for _v in _words]

        if _words[0] == "prefix":
            _prefix = _words[1] if _words[1] != '-' else ''
            continue

        # get ctj pair
        if len(_words) > 2:

            if _words[3] == '-' and _prefix[-1] == '.':
                _ctj_ij = ( (_prefix+_words[0],_words[1],_words[2]), (_prefix[:-1],_words[4],_words[5]) )
            else:
                _ctj_ij = ( (_prefix+_words[0],_words[1],_words[2]), (_prefix+_words[3],_words[4],_words[5]) )

            if _ctj_ij in _cont_ctj_table:
                contIndex = _cont_ctj_table.index( _ctj_ij )
                _PI_coe.idxI[contIndex] = _level_info_table.index( _ctj_ij[0] )
                _PI_coe.idxJ[contIndex] = _level_info_table.index( _ctj_ij[1] )

                nLambda = int(_words[6])
                _PI_coe.nLambda[contIndex] = nLambda
                _PI_coe.alpha0[contIndex] = float(_words[8])

                _readMesh = True
                _countMesh = 0
                mesh_array = np.zeros((2,nLambda),np.double)
            else:
                _readMesh = False

        else:
            if not _readMesh:
                continue
            #_PI_table[_countLine,_countMesh,:] += [float(v) for v in _words]
            mesh_array[:,_countMesh] = [float(v) for v in _words]

            if _countMesh == (nLambda-1):
                _PI_table_dict[contIndex] = mesh_array

            _countMesh += 1

    return None

def read_Radiative_Line_Number(_lns):
    r"""
    read nRadiativeLine
    """
    for _i, _ln in enumerate(_lns[:]):

        if skip_line(_ln):
            continue
        elif check_end(_ln):
            break

        _words = _ln.split()
        _words = [_v.strip() for _v in _words]

        if _words[0].lower() == "nradiative":
            _nRadiativeLine = int( _words[1] )

    _re = _i + 1

    return _nRadiativeLine, _re

def read_Mesh_Info(_rs, _lns, _Mesh_coe, _filename, _level_info_table, _line_ctj_table):
    r"""
    read CI table for interpolation
    """
    _count = 0
    #_prefix = ''
    for _i, _ln in enumerate(_lns[_rs:]):

        if skip_line(_ln):
            continue
        elif check_end(_ln):
            break

        _words = _ln.split()
        _words = [_v.strip() for _v in _words]

        if _words[0] == "prefix":
            _prefix = _words[1] if _words[1] != '-' else ''
            continue

        # get ctj pair
        _ctj_ij = ( (_prefix+_words[0],_words[1],_words[2]), (_prefix+_words[3],_words[4],_words[5]) )

        assert _ctj_ij in _line_ctj_table

        _Mesh_coe.lineIndex[_count] = _line_ctj_table.index( _ctj_ij )
        _Mesh_coe.idxI[_count] = _level_info_table.index( _ctj_ij[0] )
        _Mesh_coe.idxJ[_count] = _level_info_table.index( _ctj_ij[1] )

        if _words[6] == "Voigt":
            _Mesh_coe.ProfileType[_count] = 0
        else:
            assert False

        _Mesh_coe.nLambda[_count] = int(_words[7])
        _Mesh_coe.qcore[_count] = float(_words[10])
        _Mesh_coe.qwing[_count] = float(_words[11])

        _filename.append( _words[12] )

        _count += 1

    return None

def read_half_intensity_hz(_path):

    _arr = np.loadtxt(_path, skiprows=3)

    return _arr
