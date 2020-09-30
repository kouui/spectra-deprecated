
import sys
sys.path.append( "../" )


from src.Util.Nist import LineQuery
from src.Util.External import mendeleev

if __name__ == "__main__":

    _elementQ = LineQuery.ElementQuery( waiting=1, data_directory="/Users/liu/kouui/data/NIST/Line/" )
    for i in range(1, 27):
        mendeleev_info = mendeleev.element( i )
        _element = mendeleev_info.symbol
        _elementQ.set_wavelength_range(0, 1000000, unit="AA") # 0-100um
        _elementQ.query( _element )
        _elementQ.to_csv()
