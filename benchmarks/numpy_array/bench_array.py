
import numpy
import numba
import sys
import timeit, functools, collections

import matplotlib.pyplot as plt

sys.path.append('../../')

from src.Util import Benchmark


def f_empty(_n):
    _x = numpy.empty( _n, dtype=numpy.float64 )
    return _x

def f_ones(_n):
    _x = numpy.ones( _n, dtype=numpy.float64 )
    return _x

def f_zeros(_n):
    _x = numpy.zeros( _n, dtype=numpy.float64 )
    return _x


if __name__ == "__main__":

    nLoop_ = 10


    ns_ = [10, 100, 1000, 10000, 100000, 1000000]

    functions_ = collections.OrderedDict({
        "numpy.empty" : f_empty,
        "numpy.zeros" : f_zeros,
        "numpy.ones" : f_ones,
    })
    result_ = collections.OrderedDict()
    for key_ in functions_.keys():
        result_[key_] = []

    for n_ in ns_:

        print(f"n = {n_}")

        for name_, fun_ in functions_.items():

            _ = fun_( n_ )
            t_ = timeit.Timer( functools.partial( fun_,n_ ) )
            result_[name_].append( t_.timeit(nLoop_)/nLoop_ )

    Benchmark.print_table(ns_, result_, outFile='./result.txt')
    Benchmark.make_figure(ns_, result_, './result.png', title='numpy array initialization')
