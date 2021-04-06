"""
Provides functionality for reading and changing the tabbin files of Rigaku Crysalis (former Oxford)
"""

__author__ = "Konstantin Glazyrin"
__copyright__ = "Creative Commons"

__all__ = ["CrysalisTabbinController"]

try:
    from crysalis.imports import *
except ImportError:
    from ..imports import *

try:
    from crysalis.common.logger import *
except ImportError:
    from ..common.logger import *

try:
    from crysalis.common.config import *
except ImportError:
    from ..common.config import *

from .keys import *

class TabbinAbstract(Tester):
    """
    Class controlling abstract functionality, like data conversion
    """

    intlist = (np.int, np.int8, np.int16, np.int32, np.int64)
    uintlist = (np.uint, np.uint8, np.uint16, np.uint32, np.uint64)
    floatlist = (np.float, np.float32, np.double)

    def __init__(self, debug_mode=DEBUG, show_init=None):
        Tester.__init__(self, debug_mode=debug_mode, show_init=show_init)

    def _get_value_from_binary(self, nptype, value):
        """
        Returns a numpy type value from a binary string
        :param nptype:
        :param value:
        :return:
        """
        res = None
        try:
            if nptype in self.intlist:
                value = nptype(int.from_bytes(value, byteorder='little', signed=True))
            elif nptype in self.uintlist:
                value = nptype(int.from_bytes(value, byteorder='little', signed=False))
            elif nptype in self.floatlist:
                if len(value) == 8:
                    value = nptype(struct.unpack("d", value)[0])
                else:
                    value = nptype(struct.unpack("f", value)[0])
            res = value
        except ValueError:
            self.error("Cannot convert data ({}) to np.type ({})".format(value, nptype))
        return res

    def _get_pxdistance(self, x1, y1, x2, y2):
        """
        Returns the distance between the coordinates
        :param x1:
        :param y1:
        :param x2:
        :param y2:
        :return:
        """
        x1, x2, y1, y2 = float(x1), float(x2), float(y1), float(y2)
        return math.sqrt(math.pow(x2-x1, 2)+math.pow(y2-y1, 2))

    def _get_binary_from_value(self, nptype, value):
        """
        Returns a binary representation of the value
        :param ntype:
        :param value:
        :return:
        """
        res = None

        value_size = nptype(0.).itemsize

        if nptype in self.intlist or nptype in self.uintlist:
            res = int(value).to_bytes(length=value_size, byteorder="little")
        elif nptype in self.floatlist:
            if value_size == 8:
                res = bytearray(struct.pack("d", value))
            elif value_size == 4:
                res = bytearray(struct.pack("f", value))
            else:
                self.error("Could not pack the value ({}) of type ({})".format(value, nptype))
        return res


class TabbinPoint(TabbinAbstract):
    """
    Class controlling the information for the points of the tabbin file
    Reading and writing is conducted directly into the file data reference
    """
    GROUP_CHANGED = 0

    STRUCT_SIZE = 168

    # offsets going after the header in decimal, first number - start, numpy value - size
    OFFSETS_LOCAL = {
        OFFSET_DX: [0, np.float64],
        OFFSET_DY: [8, np.float64],
        OFFSET_DZ: [16, np.float64],
        OFFSET_DLENGTH: [24, np.float64],
        OFFSET_LINTENSITY: [32, np.uint32],
                                                    # OFFSET_IFILTER_OBSOLETE: [36, np.uint16],
                                                    # OFFSET_ISYMSETTING_OBSOLETE: [38, np.uint16],
                                                    # OFFSET_ISWSMODE_OBSOLETE: [40, np.uint16],
                                                    # OFFSET_IPHTSCANTYPE: [42, np.uint16],
        OFFSET_IXPIX: [44, np.uint16],
        OFFSET_IYPIX: [46, np.uint16],
        OFFSET_FCENTROIDX: [48, np.float32],
        OFFSET_FCENTROIDY: [52, np.float32],
                                                    # OFFSET_DAPHTSTARTANGLESRAD: [56, [np.float32, np.float32, np.float32, np.float32, np.float32, np.float32,]],
                                                    # OFFSET_DAPHTENDANGLESRAD: [104, [np.float32, np.float32, np.float32, np.float32, np.float32, np.float32,]]
                                                    # OFFSET_LCALCULATIONSTATUS: [152, np.uint32],
        OFFSET_UTWINGROUPFLAGS: [156, np.uint32],
        OFFSET_DWRUNFRAME1BASED: [160, np.int16],
                                                    #OFFSET_DWFRAMESTAMP_OR_LO_RINGNUMBER_HI_FRAMEID: [164, np.uint32],
        OFFSET_GROUPKEY: [None, np.int8]            # dummy, no real sence, but convinient
    }

    def __init__(self, npoint, file_contents, offset_start, offset_group, debug_mode=None, show_init=False):
        TabbinAbstract.__init__(self, debug_mode=debug_mode, show_init=show_init)

        self.init_variables(npoint, file_contents, offset_start, offset_group)

    def init_variables(self, npoint, file_contents, offset_start, offset_group):
        """
        Initializes the variables
        :return:
        """
        self.file_contents = file_contents

        self.offset_start = offset_start
        self.offset_group = offset_group

        # current point number
        self._number = npoint

        # initialize offsets
        self.fill_offsets()

    def fill_offsets(self):
        """
        Initializes the offsets for the given file and point
        :return:
        """

        self.offsets = {}

        for k in self.OFFSETS_LOCAL.keys():
            if k != OFFSET_GROUPKEY:
                offset = self.offset_start + self.OFFSETS_LOCAL[k][0]
                self.offsets.setdefault(k, offset)
            else:
                self.offsets.setdefault(k, self.offset_group)

        if self.is_debug_level() and self.number <2:
            self.debug(self.offsets)

    def _read_values(self, key):
        """
        Reads a value from the data and returns it in a given type
        :param key:
        :return:
        """
        res = None

        # get type of the value
        offset = self.offsets[key]
        value_type = self.OFFSETS_LOCAL[key][1]
        value_size = value_type(0).itemsize

        raw_value = self.file_contents[offset:offset + value_size]
        res = self._get_value_from_binary(value_type, raw_value)

        if self.is_debug_level():
            self.debug("Reading key ({}); type ({}); raw ({}) value ({})".format(key, value_type, raw_value, res))

        return res

    def _write_values(self, key, value):
        """
        Reads a value from the data and returns it in a given type
        :param key:
        :return:
        """
        res = None

        # get type of the value
        offset = self.offsets[key]
        value_type = self.OFFSETS_LOCAL[key][1]
        value_size = value_type(0).itemsize

        if self.is_debug_level():
            self.debug("Preparing a value key ({}) type ({}) value ({}) offset ({}) raw ({})".format(key, value_type, value, offset, self._get_binary_from_value(value_type, value)))

        raw_value = self._get_binary_from_value(value_type, value)

        self.debug(raw_value)
        self.file_contents[offset:offset+value_size] = raw_value
        res = raw_value


        if self.is_debug_level():
            self.debug("Writing key ({}); type ({}); raw ({}) value ({})".format(key, value_type, raw_value, value))
            tdata = [" {}".format(hex(int(raw_value[i])).upper()) for i in range(value_type(0).itemsize)]
            self.debug(tdata)

        return res

    @property
    def number(self):
        return self._number

    @number.setter
    def number(self, value):
        self._number = value

    @property
    def dx(self):
        key = OFFSET_DX
        v = self._read_values(key)
        if v is not None:
            v = v
        return v

    @property
    def dy(self):
        key = OFFSET_DY
        v = self._read_values(key)
        if v is not None:
            v = v
        return v

    @property
    def dz(self):
        key = OFFSET_DZ
        v = self._read_values(key)
        if v is not None:
            v = v
        return v

    @property
    def dlength(self):
        key = OFFSET_DLENGTH
        v = self._read_values(key)
        if v is not None:
            v = v
        return v

    @property
    def px(self):
        key = OFFSET_IXPIX
        v = self._read_values(key)
        if v is not None:
            v = v+1
        return v

    @property
    def py(self):
        key = OFFSET_IYPIX
        v = self._read_values(key)
        if v is not None:
            v = v + 1
        return v

    @property
    def intensity(self):
        key = OFFSET_LINTENSITY
        v = self._read_values(key)
        if v is not None:
            v = v + 1
        return v

    @property
    def frame(self):
        key = OFFSET_DWRUNFRAME1BASED
        v = self._read_values(key)
        if v is not None:
            v = v
        return v

    @property
    def group(self):
        key = OFFSET_GROUPKEY
        v = self._read_values(key)
        if v is not None:
            v = v + 1
        return v

    @group.setter
    def group(self, value):
        key = OFFSET_GROUPKEY
        try:
            # values start at 0
            value = value - 1

            if value < 1:
                value = 0
            elif value > 16:
                value = 15
            else:
                self._write_values(key, value)
                self.GROUP_CHANGED += 1
        except ValueError:
            self.error("Error on conversion the group value ({})".format(value))

    def setGroup(self, value):
        self.debug("Setting the {} {} {} ".format(self.number, self.px, self.py))
        self.group = value
        return value

    def resetGroupChanged(self):
        """
        Resets the class related counter
        :return:
        """
        self.GROUP_CHANGED = 0

    def getGroupChanged(self):
        """
        Returns the group change counter
        :return:
        """
        return self.GROUP_CHANGED

    def __str__(self):
        """
        Returns the text representation for the class
        :return:
        """
        return "{}(number={}, intensity={}, group={}, px={}, py={})".format(self.__class__.__name__, self.number, self.intensity, self.group, self.px, self.py)


class CrysalisTabbinController(TabbinAbstract):
    """
        Class controlling the tabbin file functionality of the Crysalis library
    """

    # global, known offsets
    OFFSETS = {
        OFFSET_POINT_NUM: [0, np.uint32],
        OFFSET_IVERSION_TABBIN_HEADER : [10, np.uint16],

        OFFSET_GROUP_LIST: None,   # variable

        # fixed offsets
        OFFSET_POINT_LIST: 312,  # 0x138 - fixed
        OFFSET_POINT_LISTNEXT: 168,
        OFFSET_GROUPSTART: 806,  # fixed with respect to the OFFSET_GROUP_LIST start of the group value for a point
        OFFSET_GROUPNEXT: 32,  # repetition of the group value for the point
    }

    # key to look for in order to find the groups
    OFFSET_GROUPSEARCHKEY = b'\x2B\x58\x47\x47'

    # VERSION test
    MIN_VERSION = 2
    MAX_VERSION = 16

    def __init__(self, debug_mode=None, show_init=None):
        Tester.__init__(self, debug_mode=debug_mode, show_init=show_init)

        # variables - globals
        self.num_points = 0
        self.version = None
        self.points = []
        self.file_contents = None

        # points xy
        self.points_xy = {}
        self.points_xy_ref = {}

        # points for the x and y direction
        self.points_framey = {}
        self.points_framey_ref = {}
        self.points_framex = {}
        self.points_framex_ref = {}

        # 2D representation of the detector
        self.nparray = None

        # array to use for testing neighboring points
        self.points_addskip = []

    def __str__(self):
        """
        Returns the text representation for the class
        :return:
        """
        return "{}(num_points={}, version={}, points={})".format(self.__class__.__name__,  self.num_points,self.version,  self.points)

    def read_file(self, path, binning=1.):
        """
        Reads the TABBIN
        :param path:
        :return:
        """
        test = self.test_existing_file(path)
        self.debug("Trying to read the file ({}). File exists ({})".format(path, test))

        # test for the file
        if not test:
            self.error("File ({}) does not exist".format(path))
            return

        # test the file format
        test_format = self.test_tabbin_format(path)
        if not test_format:
            return

        # read all the points
        self.read_points(binning=binning)

    def read_points(self, binning=1.):
        """
        Fill the data points with values
        :return:
        """
        # read the data - piece by piece
        maxx, maxy = 0, 0

        if self.num_points > 0:
            po = self.OFFSETS[OFFSET_POINT_LIST]
            pi = self.OFFSETS[OFFSET_POINT_LISTNEXT]
            of_group = self.OFFSETS[OFFSET_GROUP_LIST]
            of_group_pi = self.OFFSETS[OFFSET_GROUPNEXT]

            # get dimensions - this is fast
            for npoint in range(self.num_points):
                # get point data 168 byte - tabbin_header_tag
                offset_start = po + npoint * pi
                offset_stop = po + (npoint + 1) * pi
                offset_group = of_group + npoint * of_group_pi

                npoint_data = self.file_contents[offset_start:offset_stop]
                point = TabbinPoint(npoint, self.file_contents, offset_start, offset_group)

                # calculate statistics
                tpx = point.px
                tpy = point.py

                if tpx > maxx:
                    maxx = tpx
                if tpy > maxy:
                    maxy = tpy

            self.info("Max image dimensions are: ({}:{})".format(maxx, maxy))

            # represent the detector
            self.nparray = np.empty([maxx+1, maxy+1], dtype=list)

            # get real data
            for npoint in range(self.num_points):
                # get point data 168 byte - tabbin_header_tag
                offset_start = po + npoint * pi
                offset_stop = po + (npoint + 1) * pi
                offset_group = of_group + npoint * of_group_pi

                npoint_data = self.file_contents[offset_start:offset_stop]

                tdata = [" {}".format(hex(int(npoint_data[i])).upper()) for i in range(8)]

                # pass the reference to the full data, save the new object
                point = TabbinPoint(npoint, self.file_contents, offset_start, offset_group)

                # test - for the header + data
                if self.is_debug_level():
                    if npoint < 5:
                        self.debug("\t ({:02}) Data is ({})".format(npoint, npoint_data))
                        self.debug("\t ({:02}) First element is ({})".format(npoint, tdata))

                        tgroup = hex(int(self.file_contents[offset_group])).upper()
                        self.debug("\t ({:02}) Group number is ({}:{})".format(
                            npoint, offset_group, tgroup
                        ))

                        # read data from the tabbin class
                        self.debug(point)

                # append data
                self.points.append(point)
                self.debug("Offset stop is ({})".format(offset_stop))

                # calculate statistics
                tpx = point.px
                tpy = point.py
                tframe = point.frame

                # binning if needed
                tbpx = int(binning*round(float(tpx)/binning))
                tbpy = int(binning*round(float(tpy)/binning))

                if tbpx > self.nparray.shape[0] or tbpy > self.nparray.shape[1]:
                    self.error("Some size is wrong ( x: {}) or ( y: {})".format(tbpx, tbpy))
                    continue

                txy = "{} {}".format(tbpx, tbpy)
                if not txy in self.points_xy.keys():
                    self.points_xy.setdefault(txy, 1)
                else:
                    self.points_xy[txy] += 1

                # Y projection
                tframey = "{}+{}".format(tframe, tbpy)
                if not tframey in self.points_framey.keys():
                    self.points_framey.setdefault(tframey, 1)
                    self.points_framey_ref.setdefault(tframey, [])
                    self.points_framey_ref[tframey].append(point)
                else:
                    self.points_framey[tframey] += 1
                    self.points_framey_ref[tframey].append(point)

                # X projection
                tframex = "{}+{}".format(tframe, tbpx)
                if not tframex in self.points_framex.keys():
                    self.points_framex.setdefault(tframex, 1)
                    self.points_framex_ref.setdefault(tframex, [])
                    self.points_framex_ref[tframex].append(point)
                else:
                    self.points_framex[tframex] += 1
                    self.points_framex_ref[tframex].append(point)

                # creating a reference layout for the pixels
                if self.nparray[tbpx, tbpy] is None:
                    self.nparray[tbpx, tbpy] = list([point])
                    self.debug("Adding # {} {} {} ".format(point.number, tpx, tpy))
                else:
                    self.nparray[tbpx, tbpy].append(point)
                    self.debug("Adding multiple {} {} {} ".format(point.number, tpx, tpy))

                # add a reference for stats
                self.points_xy_ref.setdefault(point, [tpx, tpy, tbpx, tbpy])

    def mod_list_pixelmultiframe(self, path, group=None, radius=20., frame_threshold=7):
        """
        Returns a list of references where frames have pixels at the same spot
        :return:
        """
        # filter the list and flatten it
        tstart = time.time()
        radius = int(radius)

        # list of list with non zero values
        tlist_main = list(filter(lambda l: l is not None and len(l)> 1, list(self.nparray.flatten())))
        tlist_short = [sublist[0] for sublist in tlist_main]
        if len(tlist_short)>0:
            tlist_short[0].resetGroupChanged()

        tprep = time.time()
        self.info("Short list of the points with the same pixel coordinates withing binning ({})".format(len(tlist_short)))

        times = []
        tellist = []

        # filter diamonds appearing in different frames
        for el in tlist_short:
            tprep = time.time()
            elposx, elposy = self.points_xy_ref[el][0], self.points_xy_ref[el][1]
            el.setGroup(group)

            nxi, nxa = elposx-radius, elposx+radius
            nyi, nya = elposy-radius, elposy+radius

            if nxi < 0:
                nxi = 0

            if nyi < 0:
                nyi = 0

            self.debug("Origin X {} {} ({}:{}) Y {} ({}:{})".format(el.number, elposx, nxi, nxa, elposy, nyi, nya))

            tellist = list(filter(lambda l: l is not None, list(self.nparray[nxi:nxa, nyi:nya].flatten())))
            [item.setGroup(group) for sublist in tellist for item in sublist]

            times.append(time.time()-tprep)

        # filter diamonds creating stripes either in horizontal or vertical direction
        for vs in self.points_framey_ref.values():
            if( len(vs) > frame_threshold):
                for el in vs:
                    el.setGroup(group)

        for vs in self.points_framex_ref.values():
            if( len(vs) > frame_threshold):
                for el in vs:
                    el.setGroup(group)

        if len(tlist_short) > 0:
            self.info("Group was changed ({}) times".format(tlist_short[0].getGroupChanged()))

        self.debug("Converted {}".format(["--- {} {} {}".format(item.number, item.px, item.py) for sublist in tellist for item in sublist]))

        tstop = time.time()
        # write the file with the group
        with open(path, "wb") as fh:
            fh.write(self.file_contents)
        tfull = time.time()

        self.info("Times stop-start ({}) final-start ({}) write-stop ({})".format(tstop - tstart, tfull - tstart,
                                                                                  tfull - tstop))

    def test_tabbin_format(self, path):
        """
        Tests the file for the format (first 4 bytes should provide the number of points, version header should have meaningful info)
        :param path:
        :return:
        """
        res = False

        self.debug("Testing the file ({}) format".format(path))

        with open(path, "rb") as fh:
            # get full file contents, find the mentioning of the group header, starting from the end of the file // typical file size is small
            file_contents = fh.read()

            # select the data piece by piece
            key = OFFSET_POINT_NUM
            offset = self.OFFSETS[key][0]
            num_files_type = self.OFFSETS[key][1]
            size = num_files_type(0).itemsize
            num_points = file_contents[offset:offset+size]

            # read version number
            key = OFFSET_IVERSION_TABBIN_HEADER
            offset = self.OFFSETS[key][0]
            version_type = self.OFFSETS[key][1]
            size = version_type(0).itemsize
            version = file_contents[offset:offset+size]

            try:
                if len(version) == 0 or len(num_points) == 0:
                    raise ValueError

                num_points = self._get_value_from_binary(num_files_type, num_points)
                version = self._get_value_from_binary(version_type, version)

                if version is None or num_points is None:
                    raise ValueError

                if not is_tabbin_version_allowed(version):
                    raise ValueError

                self.num_points = num_points
                self.version = version
                key = OFFSET_GROUP_LIST
                self.OFFSETS[key] = self.OFFSETS[OFFSET_POINT_LIST]+self.OFFSETS[OFFSET_POINT_LISTNEXT]*num_points+self.OFFSETS[OFFSET_GROUPSTART]

                self.info(
                    "The tabbin file ({}) has version ({}) and contains ({}) points. Group position reference is ({})".format(
                        path, version, num_points, self.OFFSETS[key]))

                # store a copy of the file contents - to modify and write
                self.file_contents = bytearray(file_contents)

                res = True
            except ValueError:
                self.error("The file format is wrong. Num. files ({}); Version ({}); Group position ({})".format(num_points, version, self.OFFSETS[OFFSET_GROUP_LIST]))

        return res

    def _get_value_from_binary(self, nptype, value):
        """
        Returns a numpy type value from a binary string
        :param nptype: 
        :param value: 
        :return: 
        """
        res = None
        try:
            if nptype in self.intlist:
                value = nptype(int.from_bytes(value, byteorder='little', signed=True))
            elif nptype in self.uintlist:
                value = nptype(int.from_bytes(value, byteorder='little', signed=False))
            elif nptype in self.floatlist:
                value = nptype(float.fromhex(value))
            res = value
        except ValueError:
            self.error("Cannot convert data ({}) to np.type ({})".format(value, nptype))
        return res