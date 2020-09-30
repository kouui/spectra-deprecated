

from logzero import logger

#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------

class MyLogger:

    def __init__(self, verbose):

        self.verbose = verbose

    def info(self, *args, **kargs):
        if self.verbose:
            logger.info(*args, **kargs)

    def error(self, *args, **kargs):
        if self.verbose:
            logger.error(*args, **kargs)

    def warning(self, *args, **kargs):
        if self.verbose:
            logger.warning(*args, **kargs)

    def logfile(self, *args, **kargs):
        logger.logfile(*args, **kargs)
