
import sys, os
ROOT_DIR = "../"
sys.path.append( ROOT_DIR )

from pprint import pprint


from src.Util.DataFormatter import LevelFormatter




if __name__ == "__main__":

    _template_path = os.path.join( ROOT_DIR, 'data/atom/templates/spectrum.Level' )
    _context = {
        "csv_path" : "/Users/liu/kouui/data/NIST/Level/H.csv",
        "title" : "Hydrogen",
        'element' : 'H',
        "level_info" : {
            "I" : {
                "n_level" : 8,            # simultaneously include levels with same configuration and term
                'has_continuum' : True,   # only for final stage
            }
        },
        "out_folder" : "../data/atom/H",
        "out_file" : "H.Level"
    }

    _context = {
        "csv_path" : "/Users/liu/kouui/data/NIST/Level/Ca.csv",
        "title" : "Ca II H/K infrared",
        'element' : 'Ca',
        "level_info" : {
            "II" : {
                "n_level" : 5,            # simultaneously include levels with same configuration and term
                'has_continuum' : True,   # only for final stage
            }
        },
        "out_folder" : "../data/atom/Ca_II",
        "out_file" : "Ca_II.new.Level"
    }

    LevelFormatter.from_nist_csv( _template_path, _context )
