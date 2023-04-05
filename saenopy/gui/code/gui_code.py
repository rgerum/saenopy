import sys
import os
from qtpy import QtCore, QtWidgets, QtGui
import qtawesome as qta
from pathlib import Path
import threading
import subprocess
from dataclasses import dataclass
from saenopy.gui.code.syntax import PythonHighlighter

from saenopy.gui.common import QtShortCuts
from saenopy.gui.common.resources import resource_icon


@dataclass
class OpenScript:
    filename: Path = Path()
    code: str = ""
    console: str = ""
    unsaved: bool = False
    process: subprocess.Popen = None
    thread: threading.Thread = None


class MainWindowCode(QtWidgets.QWidget):
    console_update = QtCore.Signal(str, OpenScript)
    process_finished = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # QSettings
        self.settings = QtCore.QSettings("Saenopy", "Saenopy")

        with QtShortCuts.QVBoxLayout(self):
            with QtShortCuts.QHBoxLayout():
                self.button = QtShortCuts.QPushButton(None, "Open File", self.load, icon=qta.icon("fa5s.folder-open"))
                self.tabs = QtWidgets.QTabBar().addToLayout()
                self.tabs.setTabsClosable(True)
                QtShortCuts.current_layout.addStretch()
            with QtShortCuts.QHBoxLayout():
                self.input_filename = QtShortCuts.QInputString()
                self.input_filename.setDisabled(True)
                self.button_stop = QtShortCuts.QPushButton(None, "stop", self.stop, icon=qta.icon("fa5s.stop"))
                self.button_run = QtShortCuts.QPushButton(None, "run", self.run, icon=qta.icon("fa5s.play"))
            with QtShortCuts.QHBoxLayout():
                self.editor = QtWidgets.QPlainTextEdit().addToLayout()
                self.highlight = PythonHighlighter(self.editor.document())
                font = QtGui.QFont()
                font.setFamily('Courier')
                font.setFixedPitch(True)
                font.setPointSize(10)
                self.editor.setFont(font)

                self.console = QtWidgets.QTextEdit().addToLayout()
                self.console.setStyleSheet("background-color: #300a24; color: white")
                self.console.setReadOnly(True)

        self.console_update.connect(self.update_console)

        self.open_scripts = []
        self.open_scripts_index = -1

        self.editor.textChanged.connect(self.editor_text_changed)
        self.tabs.currentChanged.connect(self.select_tab)
        self.tabs.tabCloseRequested.connect(self.remove_tab)
        self.process_finished.connect(self.select_tab)

        self.select_tab(-1)

    def update_console(self, text, open_script):
        open_script.console += text
        if open_script == self.open_scripts[self.open_scripts_index]:
            self.console.setText(open_script.console)

    def load(self):
        new_path = QtWidgets.QFileDialog.getOpenFileName(None, "Open Script", os.getcwd(), "Python File (*.py)")
        if new_path:
            if isinstance(new_path, tuple):
                new_path = new_path[0]
            else:
                new_path = str(new_path)
            self.do_load(new_path)

    def do_load(self, open_script):
        # open a script
        self.open_scripts.append(OpenScript(filename=Path(open_script), console="", code=Path(open_script).read_text()))
        # and add the tab
        self.tabs.addTab(self.open_scripts[-1].filename.name)
        # select the new tab
        self.tabs.setCurrentIndex(len(self.open_scripts)-1)

    def select_tab(self, index=None):
        if index is not None:
            self.open_scripts_index = index
        if index == -1:
            with QtCore.QSignalBlocker(self.editor):
                self.editor.setPlainText("")
                self.editor.setDisabled(True)
            self.console.setText("")
            self.input_filename.setValue("")
            self.button_run.setDisabled(True)
            self.button_stop.setDisabled(True)
            return
        self.editor.setDisabled(False)
        open_script = self.open_scripts[self.open_scripts_index]
        self.input_filename.setValue(open_script.filename)
        with QtCore.QSignalBlocker(self.editor):
            self.editor.setPlainText(open_script.code)

        self.console.setText(open_script.console)
        if open_script.process is not None and open_script.process.poll() is None:
            self.button_run.setDisabled(True)
            self.button_stop.setDisabled(False)
        else:
            self.button_run.setDisabled(False)
            self.button_stop.setDisabled(True)

    def remove_tab(self, index):
        open_script = self.open_scripts.pop(index)
        self.tabs.removeTab(index)
        if open_script.process is not None and open_script.process.poll() is None:
            open_script.process.kill()
            open_script.thread.join()

    def editor_text_changed(self):
        open_script = self.open_scripts[self.open_scripts_index]
        open_script.code = self.editor.toPlainText()
        open_script.unsaved = True
        self.tabs.setTabText(self.tabs.currentIndex(), open_script.filename.name+" *")

    def stop(self):
        open_script = self.open_scripts[self.open_scripts_index]
        if open_script.process is not None and open_script.process.poll() is None:
            open_script.process.kill()
            open_script.thread.join()
        else:
            raise ValueError("process not running")

    def run(self):
        open_script = self.open_scripts[self.open_scripts_index]

        # kill the process
        if open_script.process is not None and open_script.process.poll() is None:
            raise ValueError("process still running")

        open_script.console = ""
        self.console.setText("")

        if open_script.unsaved:
            self.tabs.setTabText(self.tabs.currentIndex(), open_script.filename.name)
            with open_script.filename.open("w") as fp:
                fp.write(open_script.code)
        open_script.process = subprocess.Popen([sys.executable, open_script.filename], stdout=subprocess.PIPE, text=True)

        open_script.thread = threading.Thread(target=self.timer_callback, args=(open_script, ))
        open_script.thread.start()
        self.select_tab()

    def timer_callback(self, open_script):
        while True:
            if open_script.process.poll() is not None:
                break
            else:
                line = open_script.process.stdout.readline()
                self.console_update.emit(line, open_script)
        self.console_update.emit(f"\nProcess finished with exit code {open_script.process.poll()}", open_script)
        self.process_finished.emit()


if __name__ == '__main__':  # pragma: no cover
    app = QtWidgets.QApplication(sys.argv)
    if sys.platform.startswith('win'):
        import ctypes
        myappid = 'fabrylab.saenopy.master'  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    print(sys.argv)
    window = MainWindowCode()
    window.setMinimumWidth(1600)
    window.setMinimumHeight(900)
    window.setWindowTitle("Saenopy Viewer")
    window.setWindowIcon(resource_icon("Icon.ico"))
    for arg in sys.argv[1:]:
        if arg.endswith(".py"):
            window.do_load(arg)
    window.show()
    sys.exit(app.exec_())
