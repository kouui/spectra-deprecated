import numpy as np
import numba as nb
import sys
import timeit, functools, collections

import matplotlib.pyplot as plt

sys.path.append('../../')

from src.Util import Benchmark



def makeLineMesh_Half(nLambda, qcore, qwing, q):

    nLhalf = nLambda//2 + 1

    if qwing <= 2*qcore:
        beta = 1.0
    else:
        beta = 0.5 * qwing / qcore

    y = beta + (beta*beta + (beta-1.)*nLhalf + 2. - 3.*beta)**(0.5)
    b = 2.0*np.log(y) / (nLhalf - 1)
    a = qwing / (nLhalf - 2. + y*y)

    for i in range(0, nLhalf):
        q[i] = a * (i + (np.exp(b*i)-1.))


def makeLineMesh_Full(nLambda, qcore=2.5, qwing=10.):


    x = np.empty(nLambda, dtype=np.double)
    nLmid = nLambda // 2
    makeLineMesh_Half(nLambda, qcore, qwing, x[nLmid:])
    x[:nLmid] = -x[nLmid+1:][::-1]

    return x

@nb.njit('void(int64,float64,float64,float64[:])')
def makeLineMesh_Half_njit(nLambda, qcore, qwing, q):

    nLhalf = nLambda//2 + 1

    if qwing <= 2*qcore:
        beta = 1.0
    else:
        beta = 0.5 * qwing / qcore

    y = beta + (beta*beta + (beta-1.)*nLhalf + 2. - 3.*beta)**(0.5)
    b = 2.0*np.log(y) / (nLhalf - 1)
    a = qwing / (nLhalf - 2. + y*y)

    for i in range(0, nLhalf):
        q[i] = a * (i + (np.exp(b*i)-1.))

@nb.njit('float64[:](int64,float64,float64)')
def makeLineMesh_Full_njit(nLambda, qcore, qwing):


    x = np.empty(nLambda, dtype=np.double)
    nLmid = nLambda // 2
    makeLineMesh_Half_njit(nLambda, qcore, qwing, x[nLmid:])
    x[:nLmid] = -x[nLmid+1:][::-1]

    return x




if __name__ == "__main__":

    _nLoop = 10


    _ns = [11, 51, 101, 1001]

    _functions = collections.OrderedDict({
        "np" : makeLineMesh_Full,
        "np+njit" : makeLineMesh_Full_njit,
    })
    _result = collections.OrderedDict()
    for _key in _functions.keys():
        _result[_key] = []

    for _n in _ns:

        print(f"n = {_n}")

        #_x = np.random.rand(_n)[:]
        #_y = np.random.rand(_n)[:]

        for _name, _fun in _functions.items():

            _ = _fun( _n, 2.5, 10)
            _t = timeit.Timer( functools.partial( _fun,_n, 2.5, 10) )
            _result[_name].append( _t.timeit(_nLoop)/_nLoop )

    Benchmark.print_table(_ns, _result, outFile='./result.txt')
    Benchmark.make_figure(_ns, _result, './result.png', title='makeLineMesh_Full, numpy vs njit',)
