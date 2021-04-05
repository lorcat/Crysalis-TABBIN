from app.imports import *


class Starter:
    def __init__(self):
        """
        Initialization
        """
        super(Starter, self).__init__()

        # file upload
        self.lbl_filename = None
        self.btn_filename = None
        self.lbl_filedate = None

        self.lbl_statdata = None
        self.lbl_cell = None
        self.lbl_const_cell = None
        self.lbl_ub = None
        self.lbl_ubline = None
        self.lbl_m = None
        self.lbl_b = None

        self.last_data = None
        self.last_fn = None

        # matrix
        self.matrix_ub = None
        self.matrix_u = None
        self.matrix_b = None
        self.matrix_m = None

        self.input_gui()

    def on_cifod_upload(self, filedata):
        """
        Process information on upload if necessary
        """
        pass

    def input_gui(self):
        """
        Initializes the graphical design
        """
        # file upload
        self._init_fileupload()

        asyncio.ensure_future(self._fn())

    def _init_fileupload(self):
        """
        Initializes file upload
        """
        # file upload button + label
        w1 = FileUpload(
            accept='.cif_od',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
            multiple=False,  # True to accept multiple files upload else False
            description='Load (*.cif_od) file:',
            layout=Layout(flex='0 1 auto', min_height='40px', width='auto'),
        )

        w1.observe(self.on_cifod_upload)
        self.btn_filename = w1

        self.lbl_filename = w2 = HTML("")

        display(HBox([w1, Label(value=" " * 5), w2]))

        self.lbl_statdata = w3 = HTML("")

        self.lbl_const_cell = w4 = HTML("")
        self.lbl_cell = w5 = HTML("")

        self.lbl_ubline = w6 = HTML("")

        # UB matrix
        self.lbl_ub = w7 = HTML("")
        self.lbl_b = w8 = HTML("")
        self.lbl_m = w9 = HTML("")

        # stats
        display(w3)

        # cell - constrained, unconstrained, UM S
        display(GridBox([w6, w4, w5], layout=Layout(grid_template_columns="repeat(1, auto)")))

        display(GridBox([w7, w8, w9], layout=Layout(grid_template_columns="repeat(3, auto)")))

    def wait_for_filename_change(self, widget, value):
        """
        Processes asynch data change based on a widget
        """
        future = asyncio.Future()

        def getvalue(change):
            # make the new value available
            future.set_result(change.new)
            widget.unobserve(getvalue, value)

        widget.observe(getvalue, value)
        return future

    async def _fn(self):
        """
        Async handling of the uploaded file change
        """
        while True:
            x = await self.wait_for_filename_change(self.btn_filename, 'value')

            # processing data
            if x is not None:
                self.last_fn = fn = tuple(x.keys())[0]

                # start processing in a separate thread
                th = threading.Thread(target=self._process_new_data, args=[fn, x[fn]])
                th.setDaemon(True)
                th.start()

    def _process_new_data(self, filename, data):
        """
        Processes new data in a separate thread
        """
        meta = data[METADATA]
        content = data[CONTENT]

        # filling last data
        self.last_data = {}

        if isinstance(content, bytes):
            content = content.decode()

        # fill parameters with float values
        tflst = {
            PARAM_R1: float,
            PARAM_COMPLETENESS: float,
            PARAM_WAVELENGTH: float,

            PARAM_UB11: float,
            PARAM_UB12: float,
            PARAM_UB13: float,
            PARAM_UB21: float,
            PARAM_UB22: float,
            PARAM_UB23: float,
            PARAM_UB31: float,
            PARAM_UB32: float,
            PARAM_UB33: float,

            PARAM_SPACEGROUP: None,
            PARAM_SPACEGROUP_NUM: int,

            # unconstrained cell
            PARAM_CELLA: None,
            PARAM_CELLB: None,
            PARAM_CELLC: None,
            PARAM_AL: None,
            PARAM_BE: None,
            PARAM_GA: None,
            PARAM_VOL: None,

            PARAM_REFLECTIONS: int,

            # constrained cell
            PARAM_CONST_CELLA: None,
            PARAM_CONST_CELLB: None,
            PARAM_CONST_CELLC: None,
            PARAM_CONST_AL: None,
            PARAM_CONST_BE: None,
            PARAM_CONST_GA: None,
            PARAM_CONST_VOL: None,

            # supplementary
            PARAM_CREATION_DATE: None,
            PARAM_2THETA_MIN: float,
            PARAM_2THETA_MAX: float,
        }

        for k in tflst.keys():
            conv = tflst[k]

            if conv is None:
                conv = str

            p = re.compile(f"{k}\s+([^\s]+)", re.I + re.MULTILINE)
            if k == PARAM_SPACEGROUP:
                p = re.compile(f"{k}\s+'([^']+)'", re.I + re.MULTILINE)
            m = p.findall(content)

            self.last_data.setdefault(k, None)
            if len(m) > 0:
                try:
                    self.last_data[k] = conv(m[0])
                except ValueError:
                    pass

        # shows data about the filename used and the internal work date
        self.lbl_filename.value = self._get_fileinfo(filename, self.last_data[PARAM_CREATION_DATE])

        # shows data about the integration
        self.lbl_statdata.value = self._get_stats()

        # get information on the cell
        self.lbl_const_cell.value = self._get_lattice_info_const()
        self.lbl_cell.value = self._get_lattice_info()

        # get information on UB
        self.lbl_ub.value = self._get_ub()

    def _get_ub(self):
        """
        Returns HTML content for UB
        :return:
        """
        tlist = (
        PARAM_UB11, PARAM_UB12, PARAM_UB13,
        PARAM_UB21, PARAM_UB22, PARAM_UB23,
        PARAM_UB31, PARAM_UB32, PARAM_UB33,
        )

        ub = np.zeros([3,3], dtype=np.float)

        wl = 1.
        k = PARAM_WAVELENGTH
        if k in self.last_data and isinstance(self.last_data[k], float):
            wl = self.last_data[k]

        res = []

        tdata = []

        for k in tlist:
            res.append(f"<div class='data_ub'>{self.last_data[k]:0.6f}</div>")
            tdata.append(self.last_data[k])

            p = re.compile(".*UB_([0-9])([0-9])$",re.IGNORECASE)
            m = p.match(k.strip())

            # fill information on UB
            if m is not None:
                i, j = int(m.groups()[0])-1, int(m.groups()[1])-1
                ub[i, j] = self.last_data[k]

        ubline = " ".join([f"{el:2.6f}" for el in tdata])
        res.insert(0, f"""<div style='display: grid; grid-template-columns: 7em 7em 7em;'>
                  <div class='header_ub' style='text-align:left; grid-column: 1 / span 3;'>UB matrix</div>
               """)

        # ty u information
        self.lbl_ubline.value = f"<div class='data_tyu'><b>UM S {ubline}</b></div>"

        # save matrix info
        self.matrix_ub = ub
        self.matrix_u, self.matrix_b = np.linalg.qr(ub)
        self.matrix_m = m = np.transpose(self.matrix_b) * self.matrix_b

        res.append("</div>")

        # fill b and m matrix
        self._set_html_matrix(self.lbl_b, self.matrix_b, title='B matrix')
        self._set_html_matrix(self.lbl_m, self.matrix_m, title='M matrix')

        # calculate primes: vol, a,b,c,al,be,ga
        self.a_prime = a_prime = np.sqrt(m[0, 0])
        self.b_prime = b_prime = np.sqrt(m[1, 1])
        self.c_prime = c_prime = np.sqrt(m[2, 2])

        self.alpha_prime = al_prime = np.arccos(m[1, 2] / (b_prime * c_prime))
        self.beta_prime = be_prime = np.arccos(m[0, 2] / (a_prime * c_prime))
        self.gamma_prime = ga_prime = np.arccos(m[0, 1] / (a_prime * b_prime))

        self.V_prime = V_prime = np.sqrt(1 - np.cos(al_prime) ** 2 - np.cos(be_prime) ** 2 - np.cos(ga_prime) ** 2 +
                          2 * np.cos(al_prime) * np.cos(be_prime) * np.cos(ga_prime)
                          )

        self.a = a_lat = np.sin(al_prime) / (a_prime * V_prime)
        self.b = b_lat = np.sin(be_prime) / (b_prime * V_prime)
        self.c = c_lat = np.sin(ga_prime) / (c_prime * V_prime)

        self.alpha = al = np.arccos((np.cos(be_prime) * np.cos(ga_prime) - np.cos(al_prime)) / (np.sin(be_prime) * np.sin(ga_prime)))
        self.beta = be = np.arccos((np.cos(al_prime) * np.cos(ga_prime) - np.cos(be_prime)) / (np.sin(al_prime) * np.sin(ga_prime)))
        self.gamma = ga = np.arccos((np.cos(al_prime) * np.cos(be_prime) - np.cos(ga_prime)) / (np.sin(al_prime) * np.sin(be_prime)))

        res.append("<br/>Direct lattice parameters (using the wavelength):<br/>{:6.4f} {:6.4f} {:6.4f} {:6.2f} {:6.2f} {:6.2f}".format(
            a_lat * wl, b_lat * wl, c_lat * wl,
            al / np.pi * 180., be / np.pi * 180., ga / np.pi * 180.))

        res.append("<br/>Prime lattice parameters (raw):<br/>{:6.4f} {:6.4f} {:6.4f} {:6.2f} {:6.2f} {:6.2f}".format(
            a_prime, b_prime, c_prime,
            al_prime / np.pi * 180., be_prime / np.pi * 180., ga_prime / np.pi * 180.))

        return "".join(res)

    def show_cell(self):
        """
        Prepares plotly
        :return:
        """
        if self.last_data is None:
            return

        wl = 1.
        k = PARAM_WAVELENGTH
        if k in self.last_data and isinstance(self.last_data[k], float):
            wl = self.last_data[k]

        h = np.array([1., 0., 0.])
        ap_vec = self.matrix_ub @ h

        h = np.array([0., 1., 0.])
        bp_vec = self.matrix_ub @ h

        h = np.array([0., 0., 1.])
        cp_vec = self.matrix_ub @ h

        self.a_vec = a_vec = np.cross(bp_vec, cp_vec) / (self.V_prime * self.a_prime * self.b_prime * self.c_prime) * wl
        self.b_vec = b_vec = np.cross(cp_vec, ap_vec) / (self.V_prime * self.a_prime * self.b_prime * self.c_prime) * wl
        self.c_vec = c_vec = np.cross(ap_vec, bp_vec) / (self.V_prime * self.a_prime * self.b_prime * self.c_prime) * wl

        max_a = max(npla.norm(a_vec), 1.) * 2
        max_b = max(npla.norm(b_vec), 1.) * 2
        max_c = max(npla.norm(c_vec), 1.) * 2

        max_abc = max(max_a, max_b, max_c)

        fig = go.Figure(data=[
            # AXES
            go.Scatter3d(x=[0, max_abc / 2], y=[0, 0], z=[0, 0],
                         mode='lines', line=dict(width=10, color="rgb(200, 0, 0)"), name="X - beam direction"),
            go.Scatter3d(x=[0, 0], y=[0, max_abc / 2], z=[0, 0],
                         mode='lines', line=dict(width=10, color="rgb(0, 200, 0)"), name="Y"),
            go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, max_abc / 2],
                         mode='lines', line=dict(width=10, color="rgb(0, 0, 200)"), name="Z"),

            # a,b,c
            go.Scatter3d(x=[0, a_vec[0]], y=[0, a_vec[1]], z=[0, a_vec[2]],
                         mode='lines', line=dict(width=5, color="rgb(255, 100, 100)"), name="a"),

            go.Scatter3d(x=[0, b_vec[0]], y=[0, b_vec[1]], z=[0, b_vec[2]],
                         mode='lines', line=dict(width=5, color="rgb(100, 255, 100)"), name="b"),

            go.Scatter3d(x=[0, c_vec[0]], y=[0, c_vec[1]], z=[0, c_vec[2]],
                         mode='lines', line=dict(width=5, color="rgb(100, 100, 255)"), name="c"),

            # frame forming lines
            self.get_line(a_vec, b_vec), self.get_line(b_vec, a_vec),
            self.get_line(a_vec, c_vec), self.get_line(b_vec, c_vec),
            self.get_line(c_vec + a_vec, b_vec), self.get_line(c_vec + b_vec, a_vec),
            self.get_line(c_vec, b_vec), self.get_line(c_vec, a_vec),
            self.get_line(a_vec + b_vec, c_vec),
            self.get_line([0, 0, 0], a_vec + b_vec + c_vec),
        ])

        tmsg = ""
        if isinstance(self.last_fn, str):
            p = re.compile("([^\.]+)\.cif_od", re.IGNORECASE)
            m = p.match(self.last_fn)
            if m is not None:
                tmsg = m.groups()[0]

        fig.update_layout(scene=dict(
            aspectmode='cube',
            xaxis=dict(
                range=[-max_abc, max_abc],
                backgroundcolor="rgb(200, 230, 201)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white", ),
            yaxis=dict(
                range=[-max_abc, max_abc],
                backgroundcolor="rgb(255, 204, 188)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white"),
            zaxis=dict(
                range=[-max_abc, max_abc],
                backgroundcolor="rgb(179, 229, 252)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white", ), ),
            width=600,
            height=600,
            margin=dict(
                r=10, l=10,
                b=10, t=40
            ),
            showlegend=False,
            title=f"{tmsg}"
        )

        w1 = HTML("")
        w1.value = f"""
            <div>Vectors:<br/>a: {self._get_vector(a_vec)}<br/>b: {self._get_vector(b_vec)}<br/>c: {self._get_vector(c_vec)}<br/></div>
        """
        display(w1)
        display(fig)

    def _get_vector(self, v):
        """
        Returns vector
        :param v:
        :return:
        """
        res = [f"{el:6.4f}" for el in v]
        return "[ "+", ".join(res)+" ]"


    def get_line(self, a, b, color="rgb(0, 0, 0)", width=5):
        """
        Returns line representation
        :param a: 
        :param b:
        :param color:
        :param width:
        :return:
        """
        return go.Scatter3d(x=[a[0], a[0] + b[0]],
                            y=[a[1], a[1] + b[1]],
                            z=[a[2], a[2] + b[2]],
                            mode='lines', line=dict(width=width, color=color))

    def get_angle(self, a, b):
        """
        Returns an angle between two vectors in degrees
        :param a:
        :param b:
        :return:
        """
        return np.arccos(np.dot(a, b) / (npla.norm(a) * npla.norm(b))) / np.pi * 180.

    def _set_html_matrix(self, widget, matrix, title=""):
        """
        Sets content for a matrix element
        :param widget:
        :param matrix:
        :return:
        """
        res = []

        res.insert(0, f"""<div style='display: grid; grid-template-columns: 7em 7em 7em;'>
                          <div class='header_ub' style='text-align:left; grid-column: 1 / span 3;'>{title}</div>
                       """)

        mi, ma = matrix.shape
        tlist = matrix.reshape([mi*ma])

        for k in tlist:
            res.append(f"<div class='data_ub'>{k:0.6f}</div>")

        res.append("</div>")
        widget.value = "".join(res)

    def _get_fileinfo(self, filename, date):
        """
        Returns HTML content for filename and internal info date
        :return:
        """
        return f"""
            <div style='display: grid; grid-template-columns: auto auto;'>
            <div class='filename' style='padding-right: 2em; padding-left: 2em;'>Filename: <b>{filename}</b></div><div class='date'>Audit date: <b>{date}</b></div>
            </div>
        """

    def _get_lattice_info_const(self):
        """
        Returns HTML representation of lattice data
        :return:
        """
        tdata = []
        tdataraw = []
        tdataerr = []
        tlist = (PARAM_CONST_CELLA, PARAM_CONST_CELLB, PARAM_CONST_CELLC, PARAM_CONST_AL, PARAM_CONST_BE, PARAM_CONST_GA,
                 PARAM_CONST_VOL)

        for k in tlist:
            tv = self.last_data[k]
            terr = 0.

            tdataraw.append(tv)

            # determine error if there
            if "(" in tv:
                p = re.compile("([^\(]+)\(([0-9]+)\)")
                m = p.match(tv)
                if m is not None:
                    tv, terr = m.groups()

                    pos_dot, len_dec = 0, 0
                    try:
                        pos_dot = tv.index('.') + 1
                        len_dec = len(tv) - pos_dot
                    except ValueError:
                        pass

                    try:
                        terr = float(terr) * 10 ** (-len_dec)
                    except ValueError:
                        terr = 0.

            try:
                tv = float(tv)
            except ValueError:
                pass

            tdata.append(tv)
            tdataerr.append(terr)

        tmsg = '\t'.join(tdataraw)


        tmsgjana = " ".join([f"{el:6.4f}" for (i, el) in enumerate(tdata) if i < len(tdata)-1])
        tmsgjanaerr = " ".join([f"{el:6.4f}" for (i, el) in enumerate(tdataerr) if i < len(tdataerr)-1])

        tmsgexcel = [f"{el[0]:6.4f}\t{el[1]:6.4f}" for el in zip(tdata, tdataerr)]
        tmsgexcel = "\t".join(tmsgexcel)

        # simple things - space group + etc
        sg, sgnum = '', 0
        k = PARAM_SPACEGROUP
        if k in self.last_data:
            sg = "{}".format(self.last_data[k])

        k = PARAM_SPACEGROUP_NUM
        if k in self.last_data and isinstance(self.last_data[k], int):
            sgnum = "{}".format(self.last_data[k])

        res = f"""
        <div style='display: grid; grid-template-columns: 15em auto;'>
        <div class='header_cell' style='text-align:center; grid-column: 1 / span 2;'>Constrained cell ({sg}, {sgnum})</div>
        <div class='header_cell' style='text-align:center;'>Unit cell:</div>
        <div class='data_cell'>{tmsg}</div>
        <div class='header_cell' style='text-align:center;'>Jana2006:</div>
        <div class='data_cell'>{tmsgjana}</div>
        <div></div><div>{tmsgjanaerr}</div>
        <div class='header_cell' style='text-align:center;'>Excel:</div>
        <div>{tmsgexcel}</div>
        <div></div><div><a target='_blank' href='https://support.microsoft.com/en-us/office/split-text-into-different-columns-with-the-convert-text-to-columns-wizard-30b14928-5550-41f5-97ca-7a3e9c363ed7'>
        How to split text in Excel after copying</a>
        </div>
        </div>
        """
        return res

    def _get_lattice_info(self):
        """
        Returns HTML representation of lattice data
        :return:
        """
        tdata = []
        tdataraw = []
        tdataerr = []
        tlist = (PARAM_CELLA, PARAM_CELLB, PARAM_CELLC, PARAM_AL, PARAM_BE, PARAM_GA,
                 PARAM_VOL)

        for k in tlist:
            tv = self.last_data[k]
            terr = 0.

            tdataraw.append(tv)

            # determine error if there
            if "(" in tv:
                p = re.compile("([^\(]+)\(([0-9]+)\)")
                m = p.match(tv)
                if m is not None:
                    tv, terr = m.groups()

                    pos_dot, len_dec = 0, 0
                    try:
                        pos_dot = tv.index('.') + 1
                        len_dec = len(tv) - pos_dot
                    except ValueError:
                        pass

                    try:
                        terr = float(terr) * 10 ** (-len_dec)
                    except ValueError:
                        terr = 0.

            try:
                tv = float(tv)
            except ValueError:
                pass

            tdata.append(tv)
            tdataerr.append(terr)

        tmsg = '\t'.join(tdataraw)


        tmsgjana = " ".join([f"{el:6.4f}" for (i, el) in enumerate(tdata) if i < len(tdata)-1])
        tmsgjanaerr = " ".join([f"{el:6.4f}" for (i, el) in enumerate(tdataerr) if i < len(tdataerr)-1])

        tmsgexcel = [f"{el[0]:6.4f}\t{el[1]:6.4f}" for el in zip(tdata, tdataerr)]
        tmsgexcel = "\t".join(tmsgexcel)

        res = f"""
        <div style='display: grid; grid-template-columns: 15em auto;'>
        <div class='header_cell' style='text-align:center; grid-column: 1 / span 2;'>Unconstrained cell</div>
        <div class='header_cell' style='text-align:center;'>Unit cell:</div>
        <div class='data_cell'>{tmsg}</div>
        <div class='header_cell' style='text-align:center;'>Jana2006:</div>
        <div class='data_cell'>{tmsgjana}</div>
        <div></div><div>{tmsgjanaerr}</div>
        <div class='header_cell' style='text-align:center;'>Excel:</div>
        <div>{tmsgexcel}</div>
        <div></div><div><a target='_blank' href='https://support.microsoft.com/en-us/office/split-text-into-different-columns-with-the-convert-text-to-columns-wizard-30b14928-5550-41f5-97ca-7a3e9c363ed7'>
        How to split text in Excel after copying</a>
        </div>
        </div>
        """
        return res

    def _get_stats(self):
        """
        Returns HTML representation of stats data
        :return:
        """
        wl, r1, compl, num_hkls = 0.0, None, None, 0
        th2_mi, th2_ma = None, None

        k = PARAM_REFLECTIONS
        if k in self.last_data and isinstance(self.last_data[k], int):
            num_hkls = "{}".format(self.last_data[k])

        k = PARAM_WAVELENGTH
        if k in self.last_data and isinstance(self.last_data[k], float):
            wl = "{:6.4f}".format(self.last_data[k])

        k = PARAM_COMPLETENESS
        if k in self.last_data and isinstance(self.last_data[k], float):
            compl = "{:4.2f}".format(self.last_data[k])

        k = PARAM_R1
        if k in self.last_data and isinstance(self.last_data[k], float):
            r1 = "{:3.1f}".format(self.last_data[k] * 100.)

        k = PARAM_2THETA_MIN
        if k in self.last_data and isinstance(self.last_data[k], float):
            th2_mi = "{:4.4f}".format(self.last_data[k])

        k = PARAM_2THETA_MAX
        if k in self.last_data and isinstance(self.last_data[k], float):
            th2_ma = "{:4.4f}".format(self.last_data[k])

        return f"""
            <div style='display: grid; grid-template-columns: auto auto auto auto auto;'>
            <div class='header_stat' style='padding-right: 2em; padding-left: 2em; text-align: center;'>Wavelength, A:</div>
            <div class='header_stat' style='padding-right: 2em; padding-left: 2em; text-align: center;'>Used Reflections:</div>
            <div class='header_stat' style='padding-right: 2em; padding-left: 2em; text-align: center;'>2Theta (min/max), <sup>o</sup>:</div>
            <div class='header_stat' style='padding-right: 2em; padding-left: 2em; text-align: center;'>R<sub>int</sub>, %:</div>
            <div class='header_stat' style='padding-right: 2em; padding-left: 2em; text-align: center;'>Completeness, %:</div>
            <div class='data_stat' style='text-align: center;'>{wl}</div>
            <div class='data_stat' style='text-align: center;'>{num_hkls}</div>
            <div class='data_stat' style='text-align: center;'>{th2_mi} / {th2_ma}</div>
            <div class='data_stat' style='text-align: center;'>{r1}</div>
            <div class='data_stat' style='text-align: center;'>{compl}</div>
            
            </div>
        """

    def rotateX(self, v, deg):
        """
        Rotates matrix along X direction
        """
        ang = float(deg) / 180. * np.pi
        r = R.from_matrix([[1., 0., 0.],
                           [0., np.cos(ang), -np.sin(ang)],
                           [0., np.sin(ang), np.cos(ang)]
                           ])

        return r.apply(v)

    def rotateY(self, v, deg):
        """
        Rotates matrix along Y direction
        """
        ang = float(deg) / 180. * np.pi
        r = R.from_matrix([[np.cos(ang), 0., np.sin(ang)],
                           [0., 1., 0.],
                           [-np.sin(ang), 0., np.cos(ang)]
                           ])

        return r.apply(v)

    def rotateZ(self, v, deg):
        """
        Rotates matrix along Z direction
        """
        ang = float(deg) / 180. * np.pi
        r = R.from_matrix([[np.cos(ang), -np.sin(ang), 0.],
                           [np.sin(ang), np.cos(ang), 0.],
                           [0., 0., 1.]
                           ])

        return r.apply(v)
