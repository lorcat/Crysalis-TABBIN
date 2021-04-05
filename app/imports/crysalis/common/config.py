# sets the defaul debugging level
try:
    from crysalis.imports import *
except ImportError:
    from ..imports import *

__all__ = ["get_debugging_level", "set_debugging_level", "is_tabbin_version_allowed"]

# configuring the default debug level
# can be overwritten by the controller
DEBUGGING_LEVEL = INFO

# ALLOWED TABBIN versions
TABBIN_VERSION_MIN = 2
TABBIN_VERSION_MAX = 5

def get_debugging_level():
    global  DEBUGGING_LEVEL
    return DEBUGGING_LEVEL

def set_debugging_level(level=DEBUG):
    global  DEBUGGING_LEVEL
    DEBUGGING_LEVEL = level

def is_tabbin_version_allowed(version):
    """
    Performs a test for the validity of the tabbin version
    :param version:
    :return:
    """
    global TABBIN_VERSION_MIN, TABBIN_VERSION_MAX
    res = False
    if TABBIN_VERSION_MIN < version < TABBIN_VERSION_MAX:
        res = True
    return res