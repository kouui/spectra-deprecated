
import sys, os
ROOT_DIR = "../"
sys.path.append( ROOT_DIR )

from pprint import pprint


from src.Util.DataFormatter import AjiFormatter




if __name__ == "__main__":

    _template_path = os.path.join( ROOT_DIR, 'data/atom/templates/spectrum.Aji' )
    _context = {
        "csv_path" : "/Users/liu/kouui/data/NIST/Level/H.csv",
        "line_csv_path" : "/Users/liu/kouui/data/NIST/Line/H.csv",
        "title" : "Hydrogen",
        'element' : 'H',
        "level_info" : {
            "I" : {
                "n_level" : 8,            # simultaneously include levels with same configuration and term
                'has_continuum' : True,   # only for final stage
            }
        },
        "out_folder" : "../data/atom/H/Einstein_A",
        "out_file" : "Nist.Aji"
    }

    AjiFormatter.from_nist_csv( _template_path, _context )
