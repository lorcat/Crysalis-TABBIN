from app.imports import *
from app.watchdogs.custom import CustomFileEventHandler
from app.imports.crysalis.controller import CrysalisController

class Starter:
    """
    Handles file tracking for peak hunt files of Crysalis (Rigaku)
    """
    WATCHDOG_DELAY = 0.5
    DEBUG_MODE = logging.INFO

    def __init__(self, path=None):
        self.path = path

        # thread for watchdog
        self.th_watchdog = None
        self.file_observer = None

        # status and output
        self.lbl_status = None
        self.output = None
        self.output_lock = threading.Lock()
        self.lbl_statusline = None

        # path and process button
        self.btn_process = None
        self.lbl_path = None
        self.int_cleanup_radius = None
        self.int_group = None

        # message prefix
        self.msg_prefix = None

        # last path
        self.last_path = None

        self.setup_gui()
        self.start_watchdog()

        # queue stop - controls
        self.queue_stop = Queue()

        # self lock
        self.lock = threading.Lock()

        # crysalis controller
        self.ctrl_crysalis = CrysalisController(debug_mode=self.DEBUG_MODE)

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        """
        Cleans up running thread
        :return:
        """
        QUIT_MSG = "quit"
        for el in [self.th_watchdog]:
            self.queue_stop.put(QUIT_MSG)

        try:
            self.set_output("Cleaning up threads\n")
            self.queue_stop.join()
            self.set_output("All good\n")
        except AttributeError:
            pass

    def setup_gui(self):
        """
        Setups gui
        :return:
        """
        self.lbl_status = HTML("")

        tmsg = ""
        if os.path.isdir(self.path):
            tmsg = f"Path ({self.path}) is valid"
        else:
            tmsg = f"Path ({self.path}) is invalid"

        self.lbl_status.value = tmsg

        self.output = Output()

        display(self.lbl_status)
        display(self.output)

    def start_watchdog(self):
        """
        Setups watchdog - thread actively looking at the file changes
        :return:
        """
        if os.path.isdir(self.path):
            self.th_watchdog = threading.Thread(target=self._thread_watchdog, args=[])
            self.th_watchdog.setDaemon(True)
            self.th_watchdog.start()

    def _thread_watchdog(self):
        """
        Thread operating with the watchdog
        :return:
        """
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

        self.lbl_status.value = f"Starting watchdog for path ({self.path})"

        self.file_event_handler = CustomFileEventHandler(self)

        self.file_observer = Observer()
        self.file_observer.schedule(self.file_event_handler, self.path, recursive=True)
        self.file_observer.start()

        try:
            while True:
                # test for cleaning up message
                try:
                    self.queue_stop.get()
                    self.queue_stop.task_done()
                except Empty:
                    break

                time.sleep(self.WATCHDOG_DELAY)
        finally:
            self.file_observer.stop()
            self.file_observer.join()

    def clear_output(self):
        """
        Clears output
        :return:
        """
        with self.output_lock:
            if isinstance(self.output, Output):
                self.output.clear_output()

    def set_output(self, msg):
        """
        Adds information into the output field
        :param msg:
        :return:
        """
        with self.output_lock:
            if self.lbl_statusline is None:
                self.lbl_statusline = HTML("")
                display(self.lbl_statusline)

            if self.msg_prefix is not None:
                msg = self.msg_prefix + msg
                self.msg_prefix = None

            self.lbl_statusline.value = f"<div class=''>{msg}</div>"

    def add_file(self, filepath, deleted=False, prefix=None):
        """
        Adds filepath and gui elements if non existent, fills them with data
        :param filepath:
        :return:
        """
        if isinstance(filepath, str):
            filepath = filepath.replace("\\", "/")

        with self.lock:
            if None in (self.btn_process, self.lbl_path, self.int_cleanup_radius, self.int_group):
                self.btn_process = Button(
                                        description='Process',
                                        disabled=True,
                                        button_style='', # 'success', 'info', 'warning', 'danger' or ''
                                        tooltip='Starts processing',
                                        icon='check',
                                        width='100px'
                                    )
                self.btn_process.on_click(self.process_peakfile)
                self.int_cleanup_radius = BoundedIntText(
                                    value=7,
                                    min=1,
                                    max=20,
                                    step=1,
                                    description='Cleanup Radius:',
                                    disabled=False
                                )
                self.int_group = BoundedIntText(
                    value=2,
                    min=1,
                    max=10,
                    step=1,
                    description='Cleanup group:',
                    disabled=False
                )

                self.lbl_path = HTML("")

                display(self.lbl_path)
                display(HBox([self.int_group, self.int_cleanup_radius, self.btn_process]))

            if prefix is not None:
                self.set_output(prefix)

            if self.last_path is not None:
                if filepath in self.last_path and deleted:
                    self.btn_process.disabled = True
                    self.last_path = None
                    return

            if not deleted:
                self.btn_process.disabled = False
                self.last_path = filepath
                self.lbl_path.value = f"<div class='peakfile'><b>Latest file: {self.last_path}</b></div>"


    def process_peakfile(self, *args, **kwargs):
        """
        Starts a thread changing a file
        :return:
        """
        radius, group = 1, 2
        logging.info("Started")
        with self.lock:
            path = self.last_path
            self.btn_process.disabled = True
            radius = int(self.int_cleanup_radius.value)
            group = int(self.int_group.value)

        th = threading.Thread(target=self._process_peakfile, args=[path, radius, group])
        th.setDaemon(True)
        th.start()

        th.join()

    def _process_peakfile(self, path, radius, group):
        """
        Does real processing - open file, read, write
        :param radius:
        :return:
        """
        ts = time.time()
        tmsg = None
        if self.last_path is not None and os.path.isfile(self.last_path):
            tc = None
            with self.lock:
                tc = self.ctrl_crysalis.getTabbin(debug_mode=self.DEBUG_MODE)

            if tc is not None:
                tpath = path
                p = re.compile("(.*)\.tabbin", re.I)
                m = p.match(path)
                if m is not None:
                    tpath = m.groups()[0]

                tc.read_file(path)
                self.msg_prefix = f"""File {path} was changed. Operation took {time.time() - ts:6.2f} s<br/>
                                           rd t \"{tpath}\"<br/>"""

                tc.mod_list_pixelmultiframe(path, group=group, radius=radius)

            self.add_file(path)