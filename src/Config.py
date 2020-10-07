
import numba as nb
import numpy as np
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



nb.config.THREADING_LAYER = 'threadsafe'
""" set to thread safe library : tbb, omp, workqueue, default : workqueue. tbb if possible """

nb.set_num_threads(2)
""" limiting the number of threads """

#-----------------------------------------------------------------------------
# global datatype definition
#-----------------------------------------------------------------------------

dtFLOAT_   = np.float32
dtDOUBLE_  = np.float64
dtUINT_    = np.uint16
dtINT_     = np.int64
