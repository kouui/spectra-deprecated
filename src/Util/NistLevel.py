
import os, time

import requests
from bs4 import BeautifulSoup

from collections import OrderedDict

import pandas as pd

#-----------------------------------------------------------------------------

_DEBUG_ = True

if _DEBUG_:
    import Logger
    from external import roman, mendeleev
else:
    from . import Logger
    from .external import roman, mendeleev

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

    if item_list[0] == '' and item_list[1] == '':
        return False
    else:
        return True

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
    return s

def _preprocess_term(s):
    return s.replace('*', '')

def _preprocess_energy(s):

    if s[-2:] == '+x':
        s = s[:-2]

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

    if 'or' in s:
        return [item.strip() for item in s.split('or')]
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
        for i, key in enumerate( data_dict.keys() ):
            if key == 'J':
                data_dict[key].append(j)
            else:
                data_dict[key].append(_preprocess[key]( item_list[i] ))


def read_text_table(text_table):

    lines = text_table.split('\n')

    data_dict = OrderedDict()
    for key in ("configuration", "term", "J", "g", "E[eV]"):
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
                    print(line)

    return data_dict





#-----------------------------------------------------------------------------
class LevelQuery:


    def __init__(self, verbose=True, waiting=0):

        self.logger = Logger.MyLogger(verbose)
        self.waiting = waiting


    def query(self, spec, temp_K):

        time.sleep(self.waiting)

        temp = temp_K * K2eV_
        self.logger.info( f'querying energy levels for {spec}' )

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

        data_frame = pd.DataFrame( data_dict )

        return data_frame#, configuration_prefix

def get_ground_configuration(data_frame : pd.DataFrame):

    return data_frame.loc[0, "configuration"]

def get_configuration_prefix(conf_ground : str, conf_full : str):

    idx = conf_full.rfind( conf_ground )

    return conf_full[:idx]

def get_full_configuration(spec : str):

    element, stage = spec.split()

    return mendeleev.element(element).ec.ionize(n=roman.fromRoman(stage)-1).to_nist_str()

#-----------------------------------------------------------------------------

class ElementQuery:


    pass
#-----------------------------------------------------------------------------
if __name__ == "__main__":

    _levelQ = LevelQuery()

    _element = "H"
    _stage = "I"
    _spectrum = ' '.join([_element, _stage])

    _df = _levelQ.query(_spectrum, 1)
    #for _col in _df.columns:
    #    print(_col, ': ',_df[_col].dtype)

    _conf_ground = get_ground_configuration(_df)
    _conf_full = get_full_configuration(_spectrum)
    _conf_prefix = get_configuration_prefix(_conf_ground, _conf_full)

    print(mendeleev.element(_element).get_ionization_energy( roman.fromRoman(_stage) ))
