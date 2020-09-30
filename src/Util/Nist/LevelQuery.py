
import os, time, re

import requests
from bs4 import BeautifulSoup

from collections import OrderedDict

import pandas as pd

#-----------------------------------------------------------------------------

from .. import Logger
from ..External import roman, mendeleev
from ..System import myIO

#-----------------------------------------------------------------------------

K2eV_ = 8.6173324 * 1.E-5               #: compute electron temperature in unit eV, [:math:`eV / K`]

#-----------------------------------------------------------------------------
def is_useless(line):

    if line[0] == '-':
        return True
    else:
        split_set = set( [item.strip() for item in line.split('|')] )
        if len( split_set )==1 and '' in split_set:
            return True
        else:
            return False

def is_title(line):

    if line.startswith("Configuration"):
        return True
    else:
        return False

def is_term_start(line):

    if line[0].isnumeric():
        return True
    else:
        return False

def is_continuum(line):

    if 'Limit' in line:
        return True
    else:
        return False

def is_term_head(item_list):

    if item_list[0] == '' :#and item_list[1] == '':
        return False
    else:
        return True

def is_bad_line(item_list):

    conf, term, j, g = item_list[:4]

    words = [item for item in conf.split('.')]
    is_match = re.match(r'([0-9]+)\[[a-z]\-[a-z]\]', words[-1])
    if is_match is not None:
        status = True
    elif term != '' and j=='' and g=='':
        status = True
    else:
        status = False

    return status

def read_line(line):
    return [item.strip() for item in line.split('|')]
read_title = read_line

def assert_title(title_list):
    t = title_list
    assert t[0]=="Configuration" and t[1]=="Term" and t[2]=="J" and t[3]=="g" and t[4]=="Level (eV)"

def _remove_text_in_pair(s, c1, c2):

    idx1 = 0
    while idx1 > -1:
        if idx1 == 0:
            idx1 = s.find(c1)
        idx2 = s.find(c2)
        s = s[:idx1] + s[idx2+1:]
        idx1 = s.find(c1)

    return s

def _preprocess_configuration(s):

    if '(' in s:
        s = _remove_text_in_pair(s, '(', ')')

    s = s.replace('..', '.')
    if s[-1] == '.':
        s = s[:-1]
    return s

def _preprocess_term(s):
    s = s.replace('*', '')
    s = s.replace('?', '')
    if ' ' in s:
        s = s.split(' ')[-1]
    return s

def _preprocess_energy(s):

    if s[-2] == '+':
        s = s[:-2]
    elif s[-1] == '†':
        s = s[:-1]
    elif s[-1] ==  '?':
        if s[-3] == '+':
            s = s[:-3]
        else:
            s = s[:-1]

    try:
        return float(s)
    except ValueError:
        if '(' in s:
            s = s.replace('(', '').replace(')', '')
            return float(s)
        elif '[' in s:
            s =  s.replace('[', '').replace(']', '')
            return float(s)
        else:
            assert False, f"bad level energy value : {s}"

def _preprocess_J(s):

    if s == '':
        return [s,]
    if 'or' in s:
        return [item.strip() for item in s.split('or')]
    if ',' in s:
        return [item.strip() for item in s.split(',')]
    if s[-1] == '?':
        return [s[:-1],]
    else:
        return [s,]



_preprocess = {
    "configuration" : _preprocess_configuration,
    "term" : _preprocess_term,
    "J" : _preprocess_J,
    "g" : lambda _x : int(_x),
    "E[eV]" : _preprocess_energy,
}

def write_data(data_dict, item_list):

    out_J = _preprocess['J']( item_list[2] )

    for j in out_J:
        data_dict["configuration"].append( _preprocess["configuration"]( item_list[0] ) )
        data_dict["term"].append( _preprocess["term"]( item_list[1] ) )
        data_dict["J"].append( j )
        data_dict["g"].append( _preprocess["g"]( item_list[3] ) )
        data_dict["E[eV]"].append( _preprocess["E[eV]"]( item_list[4] ) )

        data_dict["origin_configuration"].append( item_list[0] )
        data_dict["origin_term"].append( item_list[1] )
        data_dict["origin_J"].append( item_list[2] )


def read_text_table(text_table):

    lines = text_table.split('\n')

    data_dict = OrderedDict()
    for key in ("configuration", "term", "J", "g", "E[eV]",
                "origin_configuration", "origin_term", "origin_J"):
        data_dict[key] = []


    term_conf = None
    term_term = None

    is_read = False
    is_block_read = False

    for line in lines:

        if is_block_read and line[0]== '-':
            break

        elif is_useless(line):
            term_conf = None
            term_term = None
            is_read = False
            continue

        elif is_title(line):
            title_list = read_title(line)
            assert_title(title_list)
            continue

        #elif is_continuum(line):
        #    # read ionization energy
        #
        #    # end reading
        #    print( line )
        #    break

        else:
            item_list = read_line(line)

            if is_term_head(item_list):

                if is_bad_line(item_list):
                    continue

                is_read = True
                is_block_read = True

                term_conf = item_list[0]
                term_term = item_list[1]

            else:
                item_list[0] = term_conf
                item_list[1] = term_term

            if is_read:
                try:
                    write_data(data_dict, item_list)
                except:
                    assert False, line


    return data_dict





#-----------------------------------------------------------------------------
class LevelQuery:


    def __init__(self, verbose=True, waiting=0, logger=None):

        self.logger = Logger.MyLogger(verbose) if logger is None else logger

        self.waiting = waiting


        self.data_dict = None


    def query(self, spec, temp_K=None):

        time.sleep(self.waiting)

        temp = 1 if temp_K is None else temp * K2eV_
        self.logger.info( f"querying energy levels for '{spec}'" )

         # build the web request
        nist_URL = 'http://physics.nist.gov/cgi-bin/ASD/energy1.pl'
        post_data = ('biblio=on' + '&'
                     + 'conf_out=on' + '&'
                     + 'encodedlist=XXT2' + '&'
                     + 'page_size=15' + '&'
                     + 'format=1' + '&'
                     + 'j_out=on' + '&'
                     + 'g_out=on' + '&'
                     + 'lande_out=on' + '&'
                     + 'level_out=on' + '&'
                     + 'multiplet_ordered=1' + '&'            # 0 : in term order; 1 : in energy order
                     + 'output=0' + '&'
                     + 'perc_out=on' + '&'
                     + 'spectrum=' + str(spec).replace(' ', '+') + '&'
                     + 'splitting=1' + '&'
                     + 'submit=Retrieve+Data' + '&'
                     + 'temp=' + str(temp) + '&'
                     + 'term_out=on' + '&'
                     + 'unc_out=1' + '&'
                     + 'units=1'
                     )

        full_URL = nist_URL + '?' + post_data  # This issues as a GET instead of POST, but it works ok anyway
        self.logger.info( f'{full_URL}' )

        response = requests.get(full_URL)
        root = BeautifulSoup(response.content, 'html.parser')
        text_table = root.find('pre').getText().strip()

        data_dict = read_text_table( text_table )

        self.data_dict = data_dict

        return data_dict

def get_ground_configuration(data_frame : pd.DataFrame):

    return data_frame.loc[0, "configuration"]

def get_configuration_prefix(conf_ground : str, conf_full : str):

    idx = conf_full.rfind( conf_ground )

    return conf_full[:idx]

def get_full_configuration(spec : str):

    element, stage = spec.split()

    return mendeleev.element(element).ec.ionize(n=roman.fromRoman(stage)-1).to_nist_str()

def get_ionization_energy(spec : str):

    element, stage = [item.strip() for item in spec.split()]
    return mendeleev.element(element).get_ionization_energy( roman.fromRoman(stage) )

#-----------------------------------------------------------------------------

def is_good_row(row):

    term = row["term"]
    conf = row["configuration"]
    J    = row["J"]

    status = True

    if '[' in term and ']' in term:
        status = False
    #elif term.find(' ') > -1:
    #    status = False

    if conf == '':
        status = False
    elif conf[-1] == '?':
        status = False

    if term=='' and J!='' and conf!='':
        status = False

    return status

def _postprocess_dataframe(df : pd.DataFrame):

    df["status"] = df.apply( is_good_row, axis=1 )

def _add_final_continuum_row(data : dict, ionization_energy : float, stage : str):

    data["configuration"].append( '-' )
    data["term"].append('-')
    data["J"].append('-')
    data["g"].append(1)
    data["E[eV]"].append( ionization_energy )
    data["stage"].append( stage )

    data["origin_configuration"].append( '' )
    data["origin_term"].append( '' )
    data["origin_J"].append( '' )
    #data["status"].append( True )

def _remove_space(term):
    if ' ' in term:
        return term.split()[-1]
    else:
        return term

class ElementQuery:

    def __init__(self, verbose=True, waiting=0, data_directory=None):

        self.logger = Logger.MyLogger(verbose)

        self.levelQ = LevelQuery(waiting=waiting, logger=self.logger)

        self.data_directory = data_directory
        if data_directory is not None:
            self._prepare_data_directory(data_directory)




    def query(self, element : str, to_csv = False):

        self.element = element

        self.logger.info( f"Querying for all ionization stage in element '{element}'" )

        mendeleev_info = mendeleev.element(element)
        n_electrons = mendeleev_info.electrons
        unique_notations = [element + ' ' + roman.toRoman(i) for i in range(1, n_electrons+1)]

        self.logger.info(f"Found unique notations = {unique_notations}")

        file_path = os.path.join( self.data_directory, f"{element}.pickle" ) if self.data_directory is not None else None

        if file_path is None:

            dataframe = self._query_all_ions(unique_notations)

        else:
            if os.path.exists( file_path ):
                dataframe = myIO._load_pickle( file_path )
                self.logger.info(f"dataframe for '{element}' has been loaded from : {file_path}")

                if False: # modify pickle data
                    dataframe["term"] = dataframe["term"].map( _remove_space )
                    _postprocess_dataframe( dataframe )
                    myIO._dump_pickle(_data=dataframe, _fname=file_path)

            else:
                dataframe = self._query_all_ions(unique_notations)
                myIO._dump_pickle(_data=dataframe, _fname=file_path)
                self.logger.info(f"dataframe for '{element}' has been saved as : {file_path}")

                _postprocess_dataframe( dataframe )

        self.dataframe = dataframe
        return dataframe

    def _query_all_ions(self, unique_notations : list):

        energy_bias = 0
        nIon = len(unique_notations)
        for i, spectrum in enumerate( unique_notations ):

            data_dict_spectrum = self.levelQ.query( spectrum )

            nL = len(data_dict_spectrum["configuration"])
            data_dict_spectrum["E[eV]"] = [ val+energy_bias for val in data_dict_spectrum["E[eV]"] ]
            stage = spectrum.split()[-1].strip()
            data_dict_spectrum['stage'] = [stage,] * nL
            self.logger.info(f"stored {nL} Levels for '{spectrum}'")

            if i == 0:
                data_dict_element = data_dict_spectrum
            else:
                for key in data_dict_element.keys():
                    data_dict_element[key].extend( data_dict_spectrum[key] )

            ionization_energy = get_ionization_energy( spectrum )
            energy_bias += ionization_energy

            del data_dict_spectrum
        _add_final_continuum_row(data_dict_element, energy_bias, roman.nextRoman(stage))

        dataframe = pd.DataFrame( data_dict_element )
        del data_dict_element

        return dataframe

    def _prepare_data_directory(self, data_directory : str):

        self.logger.info(f"will load/save data from/into {data_directory}")
        if not os.path.exists( data_directory ):
            os.makedirs( data_directory )
            self.logger.info(f"{data_directory} didn't exist --> created")

    def to_csv(self):
        dataframe = self.dataframe
        file_path = os.path.join( self.data_directory, f"{self.element}.csv" ) if self.data_directory is not None else None
        if file_path is not None:
            columns=("configuration", "term", "J", "g", "E[eV]", "stage", "origin_configuration","origin_term","origin_J")
            df1 = dataframe[dataframe['status']==True].reset_index()
            df1.to_csv(file_path, columns=columns)
            self.logger.info(f"dataframe for '{self.element}' has been saved as : {file_path}")



#-----------------------------------------------------------------------------
if __name__ == "__main__":
    pass

    #_levelQ = LevelQuery()

    #_element = "H"
    #_stage = "I"
    #_spectrum = ' '.join([_element, _stage])

    #_df = _levelQ.query(_spectrum, 1)
    #for _col in _df.columns:
    #    print(_col, ': ',_df[_col].dtype)

    #_conf_ground = get_ground_configuration(_df)
    #_conf_full = get_full_configuration(_spectrum)
    #_conf_prefix = get_configuration_prefix(_conf_ground, _conf_full)
    #print(mendeleev.element(_element).get_ionization_energy( roman.fromRoman(_stage) ))
