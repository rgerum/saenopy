import sys

from qtpy import QtCore, QtWidgets, QtGui
from pathlib import Path
import threading
from saenopy.gui.common.gui_classes import ProcessSimple
import html
import re


class OpenScript(QtWidgets.QWidget):
    update_console = QtCore.Signal(tuple)
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

    def update(self, tup) -> None:
        text, pipe = tup
        text = html.escape(text).replace("\n", "<br>").replace(" ", "&nbsp;")
        if pipe == "stderr":
            text = re.sub(r"File&nbsp;&quot;(.*?)&quot;,&nbsp;line&nbsp;(\d*)", r"File &quot;<a href=\1#\2 style='color: #548af7'>\1</a>&quot;, line \2", text)
            self.console += f"<span style='color: #d74b45'>{text}</span>"
        else:
            self.console += text
        self.parent.console_update.emit(self)

    def timer_callback(self, p, open_script):
        p.join()
        self.update_console.emit((f"\nProcess finished", "stdout"))
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

from contextlib import redirect_stdout, redirect_stderr
import traceback
def run_code(process, filename):

    class Capture:
        def __init__(self, pipe):
            self.pipe = pipe

        def write(self, text):
            process.put((text, self.pipe))

    with redirect_stdout(Capture("stdout")):
        with redirect_stderr(Capture("stderr")):
            try:
                source = open(filename).read()
                code = compile(source, filename, 'exec')
                exec(code, {"print": print})

            except Exception:
                error = traceback.format_exc().split("\n")
                print("\n".join(error[0:1] + error[3:]), file=sys.stderr)
