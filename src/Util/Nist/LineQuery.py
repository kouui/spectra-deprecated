
import os, time, re

import requests
from bs4 import BeautifulSoup

from collections import OrderedDict

import pandas as pd

#-----------------------------------------------------------------------------

from .. import Logger
from ..External import roman, mendeleev
from ..System import myIO

from . import LevelQuery

#-----------------------------------------------------------------------------
def is_useless(line):

    item_list = [item.strip() for item in line.split('|')]
    split_set = set( item_list )

    if line[0] == '-':
        return True
    else:
        if len( split_set )==1 and '' in split_set:
            return True
        elif item_list[6]=='' or item_list[9]=='':
            return True
        else:
            return False

def is_bad_line(item_list):

    status = False
    if item_list[1] == '':
        status = True
    elif 'Observed' in item_list[0] or 'Wavelength' in item_list[0] or "Vac" in item_list[0]:
        status = True
    else:
        pass

    return status



def read_line(line):
    return [item.strip() for item in line.split('|')]

def _preprocess_Aji(s):
    return float(s)

def _preprocess_ritzw(s):

    s = s[:-1] if s[-1] in ('+', '*') else s

    return float(s)

def _preprocess_J(s):

    if s == '':
        return s
    if 'or' in s:
        assert False, "bad J, has 'or'"
    if ',' in s:
        assert False, "bad J, has ','"
    if s[-1] == '?':
        return s[:-1]
    else:
        return s

def write_data(data_dict, item_list):

    data_dict["configuration_upper"].append( LevelQuery._preprocess_configuration( item_list[6] ) )
    data_dict["term_upper"].append( LevelQuery._preprocess_term( item_list[7] ) )
    data_dict["J_upper"].append( _preprocess_J( item_list[8] ) )
    data_dict["configuration_lower"].append( LevelQuery._preprocess_configuration( item_list[9] ) )
    data_dict["term_lower"].append( LevelQuery._preprocess_term( item_list[10] ) )
    data_dict["J_lower"].append( _preprocess_J( item_list[11] ) )

    data_dict["Aji[/s]"].append( _preprocess_Aji( item_list[3] ) )
    data_dict["ritz_wavelength[A]"].append( _preprocess_ritzw( item_list[1] ) )
    data_dict["type"].append( item_list[12] )

    data_dict["origin_configuration_upper"].append(  item_list[6] )
    data_dict["origin_term_upper"].append(  item_list[7] )
    data_dict["origin_J_upper"].append(  item_list[8] )
    data_dict["origin_configuration_lower"].append(  item_list[9] )
    data_dict["origin_term_lower"].append(  item_list[10] )
    data_dict["origin_J_lower"].append(  item_list[11] )



def read_text_table(text_table):

    lines = text_table.split('\n')

    data_dict = OrderedDict()
    for key in ("configuration_upper", "term_upper", "J_upper",
                "configuration_lower", "term_lower", "J_lower",
                "ritz_wavelength[A]", "Aji[/s]", "type",
                "origin_configuration_upper", "origin_term_upper", "origin_J_upper",
                "origin_configuration_lower", "origin_term_lower", "origin_J_lower"):
        data_dict[key] = []


    is_read = False
    is_block_read = False

    for line in lines:

        if is_block_read and line[0]== '-':
            break

        elif is_useless(line):
            continue

        else:
            item_list = read_line(line)

            if is_bad_line(item_list):
                continue

            is_block_read = True

            try:
                write_data(data_dict, item_list)
            except:
                assert False, line

    return data_dict

#-----------------------------------------------------------------------------

class LineQuery:


    def __init__(self, verbose=True, waiting=0, logger=None):

        self.logger = Logger.MyLogger(verbose) if logger is None else logger

        self.waiting = waiting


        self.data_dict = None


    def query(self, spec, wavelength_low, wavelength_up, unit="AA"):

        assert unit in ("AA", "nm", "um"), 'unit argument should be in ("AA", "nm", "um")'
        unit_int = self._unit_to_code(unit)



        time.sleep(self.waiting)

        self.logger.info( f"querying line transitions for '{spec}'" )

        nist_URL = 'http://physics.nist.gov/cgi-bin/ASD/lines1.pl'

        post_data = ('encodedlist=XXT1XXR0q0qVqVIII' + '&' # some key to make it work?
                        + 'spectra=' + str(spec).replace(' ', '+') + '&' # eg 'He' or 'He+I' or 'He+II', no spaces
                        + 'limits_type=0' + '&'
                        + 'low_wl=' + f"{wavelength_low}" + '&'
                        + 'upp_wl=' + f"{wavelength_up}" + '&'
                        + 'unit=' + f"{unit_int}" '&' # wl unit 0=Angstroms, 1=nm, 2=um
                        + 'en_unit=1' + '&' # energy unit 0 cm^-1, 1 eV, 2 Rydberg
                        + 'de=0' + '&'
                        + 'low_wn=' + '&'
                        + 'upp_wn=' + '&'
                        + 'temp=' + '&'
                        + 'doppler=' + '&'
                        + 'eden=' + '&'
                        + 'iontemp=' + '&'
                        + 'java_window=3' + '&'
                        + 'java_mult=' + '&'
                        + 'tsb_value=0' + '&'
                        + 'format=1' + '&' # 0 HTML output, 1 ascii output
                        + 'remove_js=on' + '&' # cleans up output for easier parsing
                        + 'output=0' + '&' # 0 return all output, 1 return output in pages
                        + 'page_size=15' + '&'
                        + 'bibrefs=1' + '&'
                        + 'line_out=1' + '&' # 0 return all lines, 1 only w/trans probs, 2 only w/egy levl, 3 only w/obs wls
                        + 'unc_out=0' + '&'
                        + 'order_out=0' + '&' # output ordering: 0 wavelength, 1 multiplet
                        + 'show_av=3' + '&' # show wl in Vacuum
                        + 'max_low_enrg=' + '&' # maximum lower level energy
                        + 'max_upp_enrg=' + '&' # maximum upper level energy
                        + 'min_str=' + '&' # minimum transition strength
                        + 'max_str=' + '&' # maximum transition strength
                        + 'min_accur=' + '&' # minimum line accuracy, eg AAA AA A B C
                        + 'min_intens=' + '&' # minimum relative intensity to return
                        + 'show_obs_wl=1' + '&' # show observed wavelength
                        + 'show_calc_wl=1' + '&' # show calculated (Ritz) wavelength
                        + 'A_out=0' + '&' # show $
                        + 'intens_out=on' + '&' # show relative intensity
                        + 'allowed_out=1' + '&' # show allowed transitions
                        + 'forbid_out=1' + '&' # show forbidden transitions
                        + 'conf_out=on' + '&' # show electron configuration
                        + 'term_out=on' + '&' # show terms
                        + 'enrg_out=on' + '&' # show transition energies
                        + 'J_out=on' + '&' # show J (total angular momentum)
                        # 'g_out=on' + '&' # show g (statistical weight?)
                        + 'submit=Retrieve+Data'
                    )

        full_URL = nist_URL + '?' + post_data  # This issues as a GET instead of POST, but it works ok anyway
        self.logger.info( f'{full_URL}' )

        response = requests.get(full_URL)
        root = BeautifulSoup(response.content, 'html.parser')
        content = root.find('pre')
        if content is None:
            data_dict = None
        else:
            text_table = content.getText().strip()
            data_dict = read_text_table(text_table)

        self.data_dict = data_dict

        return data_dict

    def _unit_to_code(self, unit):

        unit_dict = {
            "AA" : 0,
            "nm" : 1,
            "um" : 2,
        }

        return unit_dict[unit]

#-----------------------------------------------------------------------------

def is_good_row(row):

    status = True

    return status

def _postprocess_dataframe(df : pd.DataFrame):

    df["status"] = df.apply( is_good_row, axis=1 )

class ElementQuery(LevelQuery.ElementQuery):

    def __init__(self, verbose=True, waiting=0, data_directory=None):

        self.logger = Logger.MyLogger(verbose)

        self.lineQ = LineQuery(waiting=waiting, logger=self.logger)

        self.data_directory = data_directory
        if data_directory is not None:
            self._prepare_data_directory(data_directory)

    def set_wavelength_range(self, wavelength_low, wavelength_up, unit="AA"):

        self.wavelength_low = wavelength_low
        self.wavelength_up = wavelength_up
        self.unit = unit

    def _query_all_ions(self, unique_notations : list):

        data_dict_element = None

        for i, spectrum in enumerate( unique_notations ):

            data_dict_spectrum = self.lineQ.query( spectrum,
                                self.wavelength_low, self.wavelength_up, unit=self.unit )
            if data_dict_spectrum is None:
                self.logger.info(f"stored 0 Line transitions for '{spectrum}'")
                continue

            nL = len(data_dict_spectrum["type"])
            stage = spectrum.split()[-1].strip()
            data_dict_spectrum['stage'] = [stage,] * nL
            self.logger.info(f"stored {nL} Line transitions for '{spectrum}'")

            if data_dict_element is None:
                data_dict_element = data_dict_spectrum
            else:
                for key in data_dict_element.keys():
                    data_dict_element[key].extend( data_dict_spectrum[key] )

            del data_dict_spectrum


        dataframe = pd.DataFrame( data_dict_element )
        del data_dict_element

        return dataframe

    def _postprocess_dataframe(self, dataframe : pd.DataFrame ):

        _postprocess_dataframe(dataframe)

    def _set_csv_columns(self):

        self.csv_columns = (
             "configuration_lower", "term_lower", "J_lower",
             "configuration_upper", "term_upper", "J_upper",
             "Aji[/s]", "ritz_wavelength[A]", "type", "stage",
             "origin_configuration_lower", "origin_term_lower", "origin_J_lower",
             "origin_configuration_upper", "origin_term_upper", "origin_J_upper",
             )





#-----------------------------------------------------------------------------
