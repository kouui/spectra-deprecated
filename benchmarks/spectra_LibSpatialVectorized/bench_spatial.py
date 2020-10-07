import numpy
import numba
import sys
import timeit, functools, collections

import matplotlib.pyplot as plt

sys.path.append('../../')

from src.Util import Benchmark


from src.Structure import AtomCls
from src.Function.StatisticalEquilibrium import LibSpatialVectorized as SELib
from src.Atomic import Collision

def func(_atom, _Te, _Ne, _Tr):

    _n_LTE , _nj_by_ni_Line, _nj_by_ni_Cont = SELib.ni_nj_LTE(_atom, _Te, _Ne)

    _Rik, _Rki_stim, _Rki_spon = SELib.bf_R_rate(_atom, _Te, _nj_by_ni_Cont, _Tr=_Tr)

    _Bij_Jbar, _Bji_Jbar = SELib.B_Jbar(_atom, _Te, _Tr=_Tr)

    _Cij = SELib.get_Cij(_atom, _Te)
    _nj_by_ni = numpy.append(_nj_by_ni_Line,_nj_by_ni_Cont,axis=0)
    _Cji = Collision.Cij_to_Cji( _Cij, _nj_by_ni )

    _n_SE = SELib.solve_SE(_atom, _Ne, _Cji, _Cij, _Bji_Jbar, _Bij_Jbar, _Rki_spon, _Rki_stim, _Rik)

    return _n_SE

if __name__ == "__main__":

    nLoop_ = 10

    atom_, _ = AtomCls.InitAtom("../../data/conf/H.conf", isHydrogen=True)

    Tr_ = 6E3

    ns_ = [100, 900, 10000, 90000]#, 1000000]

    functions_ = collections.OrderedDict({
        "ni_nj_LTE" : None,
        "bf_R_rate" : None,
        "B_Jbar"    : None,
        'get_Cij'   : None,
        'solve_SE'  : None,
        "total"     : None
    })
    result_ = collections.OrderedDict()
    for key_ in functions_.keys():
        result_[key_] = []

    for n_ in ns_:

        nd_ = int( numpy.sqrt(n_) )
        arr_ = numpy.ones((nd_,nd_),dtype=numpy.double)
        Te_ = arr_ * 1.E4
        Ne_ = arr_ * 1.E11

        print(f"n = {n_}")

        #-- fun 1
        n_LTE_ , nj_by_ni_Line_, nj_by_ni_Cont_ = SELib.ni_nj_LTE(atom_, Te_, Ne_)

        name_, fun_ = 'ni_nj_LTE', SELib.ni_nj_LTE
        t_ = timeit.Timer( functools.partial( fun_, atom_, Te_, Ne_ ) )
        result_[name_].append( t_.timeit(nLoop_)/nLoop_ )

        #-- fun 2
        Rik_, Rki_stim_, Rki_spon_ = SELib.bf_R_rate(atom_, Te_, nj_by_ni_Cont_, _Tr=Tr_)

        name_, fun_ = 'bf_R_rate', SELib.bf_R_rate
        t_ = timeit.Timer( functools.partial( fun_, atom_, Te_, nj_by_ni_Cont_, _Tr=Tr_ ) )
        result_[name_].append( t_.timeit(nLoop_)/nLoop_ )

        #-- fun 3
        Bij_Jbar_, Bji_Jbar_ = SELib.B_Jbar(atom_, Te_, _Tr=Tr_)

        name_, fun_ = 'B_Jbar', SELib.B_Jbar
        t_ = timeit.Timer( functools.partial( fun_, atom_, Te_, _Tr=Tr_ ) )
        result_[name_].append( t_.timeit(nLoop_)/nLoop_ )

        #-- fun 4
        Cij_ = SELib.get_Cij(atom_, Te_)
        nj_by_ni_ = numpy.append(nj_by_ni_Line_,nj_by_ni_Cont_,axis=0)
        Cji_ = Collision.Cij_to_Cji( Cij_, nj_by_ni_ )

        name_, fun_ = 'get_Cij', SELib.get_Cij
        t_ = timeit.Timer( functools.partial( fun_, atom_, Te_ ) )
        result_[name_].append( t_.timeit(nLoop_)/nLoop_ )

        #-- fun 5
        n_SE_ = SELib.solve_SE(atom_, Ne_, Cji_, Cij_, Bji_Jbar_, Bij_Jbar_, Rki_spon_, Rki_stim_, Rik_)

        name_, fun_ = 'solve_SE', SELib.solve_SE
        t_ = timeit.Timer( functools.partial( fun_, atom_, Ne_, Cji_, Cij_, Bji_Jbar_, Bij_Jbar_, Rki_spon_, Rki_stim_, Rik_ ) )
        result_[name_].append( t_.timeit(nLoop_)/nLoop_ )

        #-- func 6
        name_, fun_ = 'total', func
        t_ = timeit.Timer( functools.partial( fun_, atom_, Te_, Ne_, Tr_ ) )
        result_[name_].append( t_.timeit(nLoop_)/nLoop_ )

    Benchmark.print_table(ns_, result_, outFile='./result.txt')
    Benchmark.make_figure(ns_, result_, './result.png', title='scaling with spatial size')
