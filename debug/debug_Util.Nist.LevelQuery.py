
import sys
sys.path.append( "../" )


from src.Util.Nist import LevelQuery
from src.Util.External import mendeleev

if __name__ == "__main__":

    _elementQ = LevelQuery.ElementQuery( waiting=1, data_directory="/Users/liu/kouui/data/NIST/Level/" )
    #for i in range(1, 27):
    for i in range(1, 27):
        mendeleev_info = mendeleev.element( i )
        _element = mendeleev_info.symbol
        _elementQ.query( _element )
        _elementQ.to_csv()
