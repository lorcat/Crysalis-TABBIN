__author__ = 'glazyrin'

from ..imports import *
from .config import *

__all__ = ["Tester"]

logging.basicConfig(level=get_debugging_level(),
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filemode='w')

# default class for logging information to the STDERR
class Logger(object):

    DEFAULTLEVEL = INFO

    def __init__(self, debug_mode=None):
        self._logger = logging.getLogger(self.__class__.__name__)

        dl = self.DEFAULTLEVEL

        try:
            test = get_debugging_level()
            if self.test_debug_mode(test):
                dl = test
        except AttributeError:
            pass

        # class definition has preference over the config file
        if self.test_debug_mode(debug_mode):
            dl = debug_mode

        # sets the debugging mode for the session
        self.debug_mode = dl
        self._logger.setLevel(dl)

    @property
    def logger(self):
        return self._logger

    def test_debug_mode(self, debug_mode):
        """
        Test for the validity of the debug mode
        :param debug_mode:
        :return:
        """
        res = False
        if debug_mode is not None and debug_mode in (DEBUG, ERROR, WARNING, INFO):
            res = True
        return res

    def info(self, msg):
        msg = self._check_msg(msg)
        self._logger.info(msg)

    def makeinfo(self, msg):
        msg = self._check_msg(msg)
        self._logger.info(msg)

    def debug(self, msg):
        msg = self._check_msg(msg)
        self._logger.debug(msg)

    def error(self, msg):
        msg = self._check_msg(msg)
        self._logger.error(msg)

    def warning(self, msg):
        msg = self._check_msg(msg)
        self._logger.error(msg)

    def _check_msg(self, msg):
        if msg is not None:
            if isinstance(msg, str):
                pass
            else:
                msg = str(msg)
        else:
            msg = str(msg)
        return msg

    def get_logger_level(self):
        return self._logger.level

    def _is_debug_level(self, level):
        res = False
        if self.get_logger_level() == level:
            res = True
        return res

    def is_debug_level(self):
        return self._is_debug_level(DEBUG)

    def is_info_level(self):
        return self._is_debug_level(INFO)

    def is_error_level(self):
        return self._is_debug_level(ERROR)

    def is_warning_level(self):
        return self._is_debug_level(WARNING)


# wrapper to test specific value types
class Tester(Logger):
    def __init__(self, debug_mode=None, show_init=True):
        Logger.__init__(self, debug_mode=debug_mode)
        if show_init is not None and show_init:
            self.debug("Class initialization.")

    def test(self, value=None, type=None):
        """
        Main function testing values
        :param value:
        :param type:
        :return:
        """
        res = False

        if value is not None:
            if type is not None and isinstance(value, type):
                res = True
            elif type is None:
                res = True

        return res

    def testString(self, value):
        """
        Tests value for being a string
        :param value:
        :return:
        """
        res = False
        if self.test(value, str) :
            res = True
        return res

    def testFloat(self, value):
        """
        Tests value for being a float
        :param value:
        :return:
        """
        return self.test(value, float)

    def testInt(self, value):
        """
        Tests value for being an integer
        :param value:
        :return:
        """
        return self.test(value, int)

    def test_existing_file(self, path):
        """
        Tests if the path is a file
        :param path:
        :return:
        """
        res = False
        if os.path.isfile(path):
            res = True
        return res

def create_path(file_path, *args):
    """
    Creates and returns the path based on file name and additional attributes
    :return:
    """
    return os.path.join(os.path.dirname(file_path), *args)
