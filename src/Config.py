
import numba
#-----------------------------------------------------------------------------
# for numba configuratoin
#-----------------------------------------------------------------------------

isJIT_ = True
"""whether to turn on the JIT compilation in all *.py files,
since sphinx does no understand numba thus can not generate documentation for numba jitted functions.

set to
    True  : before simulation; before pushing to github

    False : before generating documentation using sphinx
"""

def set_numba_thread(threading_layer='threadsafe', n_thread=2):
    r""" """

    nb.config.THREADING_LAYER = threading_layer
    """ set to thread safe library : tbb, omp, workqueue, default : workqueue. tbb if possible """

    nb.set_num_threads(n_thread)
    """ limiting the number of threads """



#-----------------------------------------------------------------------------
# global datatype definition
#-----------------------------------------------------------------------------

dtFLOAT_   = 'float32'
dtDOUBLE_  = 'float64'
dtUINT_    = 'uint16'
dtINT_     = 'int64'
