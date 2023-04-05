from qtpy import QtCore, QtWidgets, QtGui
from pathlib import Path
import threading
from saenopy.gui.common.gui_classes import ProcessSimple


class OpenScript(QtWidgets.QWidget):
    update_console = QtCore.Signal(str)
    thread = None
    unsaved = False
    is_stopped = True
    is_running = False

    def __init__(self, parent, filename):
        super().__init__(parent)
        self.parent = parent
        self.filename = Path(filename)
        self.console = ""
        self.code = Path(filename).read_text()
        self.update_console.connect(self.update)

    def update(self, text) -> None:
        self.console += text+"\n"
        self.parent.console_update.emit(self)

    def timer_callback(self, p, open_script):
        p.join()
        self.update_console.emit(f"\nProcess finished")
        self.is_stopped = True
        self.is_running = False
        self.parent.process_finished.emit()

    def run(self):
        # kill the process
        if self.thread and self.thread.is_alive():
            raise ValueError("process still running")

        self.console = ""

        if self.unsaved:
            with self.filename.open("w") as fp:
                fp.write(self.code)
            self.unsaved = False

        self.is_stopped = False
        self.is_running = True
        p = ProcessSimple(run_code, (self.filename,), {}, self.update_console)
        p.start()
        self.process = p
        self.thread = threading.Thread(target=self.timer_callback, args=(p, self,))
        self.thread.start()

    def can_run(self):
        return not self.is_running

    def can_stop(self):
        return not self.is_stopped

    def stop(self):
        if self.thread and self.thread.is_alive():
            self.is_stopped = True
            self.process.terminate()
            self.thread = None

    def change_code(self, code):
        self.code = code
        self.unsaved = True


p = print
def run_code(process, filename):
    global p
    def print(*args, **kwargs):
        text = " ".join([str(s) for s in args])
        process.put(text)
        p(text)
    source = open(filename).read()
    code = compile(source, filename, 'exec')
    exec(code, {"print": print})
