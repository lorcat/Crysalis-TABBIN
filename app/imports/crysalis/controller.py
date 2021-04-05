try:
    from crysalis.imports import *
except ImportError:
    from .imports import *

try:
    from crysalis.common.logger import *
except ImportError:
    from .common.logger import *

try:
    from crysalis.tabbin import *
except ImportError:
    from .tabbin.reader import *

class CrysalisController(Tester):
    """
    Class controlling the access to different functionality of the Crysalis library
    """
    def __init__(self, debug_mode=None):
        Tester.__init__(self, debug_mode=debug_mode)

        self.debug("Starting up")

    def getTabbin(self, debug_mode=None):
        res = CrysalisTabbinController(debug_mode=debug_mode)
        return res

