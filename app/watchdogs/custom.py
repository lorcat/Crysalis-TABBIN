from app.imports import *

class CustomFileEventHandler(FileSystemEventHandler):
    """Processes changes of the files"""

    def __init__(self, parent, queue=None, *args, **kwargs):
        super(CustomFileEventHandler, self).__init__(*args, **kwargs)

        self.queue = queue
        self.parent = parent

    def _proc_file(self, v):
        res = v
        if isinstance(v, str):
            v = v.replace("\\", "/")
        return res

    def test_file(self, v):
        res = False
        p = re.compile(".*\.tabbin", re.I)
        if p.match(v):
            res = True
        if not os.path.isfile(v):
            res = False
        return res

    def on_created(self, event):
        super(CustomFileEventHandler, self).on_created(event)

        what = 'directory' if event.is_directory else 'file'
        #logging.info("Created %s: %s", what, event.src_path)

        fn = self._proc_file(event.src_path)

        if self.parent is not None and self.test_file(fn):
            self.parent.clear_output()
            self.parent.set_output(f"Created {what}: {fn}\n")
            self.parent.add_file(fn)

    def on_deleted(self, event):
        super(CustomFileEventHandler, self).on_deleted(event)

        what = 'directory' if event.is_directory else 'file'
        #logging.info("Deleted %s: %s", what, event.src_path)

        fn = self._proc_file(event.src_path)

        if self.parent is not None and self.test_file(fn):
            self.parent.clear_output()
            self.parent.set_output(f"Deleted {what}: {fn}\n")
            self.parent.add_file(fn, deleted=True)

    def on_modified(self, event):
        super(CustomFileEventHandler, self).on_modified(event)

        what = 'directory' if event.is_directory else 'file'
        #logging.info("Modified %s: %s", what, event.src_path)

        fn = self._proc_file(event.src_path)

        if self.parent is not None and self.test_file(fn):
            self.parent.set_output(f"Modified {what}: {fn}\n")
            self.parent.add_file(fn)