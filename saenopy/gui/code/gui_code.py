import sys
import os
from qtpy import QtCore, QtWidgets, QtGui
import qtawesome as qta

from saenopy.gui.code.syntax import PythonHighlighter
from saenopy.gui.code.code_editor import CodeEditor
from saenopy.gui.code.script_file import OpenScript

from saenopy.gui.common import QtShortCuts
from saenopy.gui.common.resources import resource_icon


class Console(QtWidgets.QTextEdit):
    def __init__(self, parent, editor):
        super().__init__(parent)
        self.editor = editor

    def mousePressEvent(self, e):
        self.anchor = self.anchorAt(e.pos())
        if self.anchor:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.PointingHandCursor)

    def mouseReleaseEvent(self, e):
        if self.anchor:
            print(self.anchor)
            if "#" in self.anchor:
                file, line = self.anchor.strip('"').split("#")

                print(file, self.editor.filename)
                if file == self.editor.filename:

                    cursor = QtGui.QTextCursor(self.editor.document().findBlockByLineNumber(int(line)-1))
                    self.editor.setTextCursor(cursor)
                    #QtWidgets.QDesktopServices.openUrl(QUrl(self.anchor))
                    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.ArrowCursor)
            self.anchor = None


class MainWindowCode(QtWidgets.QWidget):
    console_update = QtCore.Signal(OpenScript)
    process_finished = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # QSettings
        self.settings = QtCore.QSettings("Saenopy", "Saenopy")

        with QtShortCuts.QVBoxLayout(self):
            with QtShortCuts.QVBoxLayout():
                with QtShortCuts.QHBoxLayout():
                    self.button = QtShortCuts.QPushButton(None, "Open File", self.load, icon=qta.icon("fa5s.folder-open"))
                    self.tabs = QtWidgets.QTabBar().addToLayout()
                    self.tabs.setTabsClosable(True)
                    self.tabs.setMaximumHeight(32)
                    QtShortCuts.current_layout.addStretch()
                    QtShortCuts.current_layout.setSpacing(11)
                QtShortCuts.QHLine().addToLayout()
                QtShortCuts.current_layout.setSpacing(0)

            with QtShortCuts.QHBoxLayout():
                self.input_filename = QtShortCuts.QInputString()
                self.input_filename.setDisabled(True)
                self.button_stop = QtShortCuts.QPushButton(None, "stop", self.stop, icon=qta.icon("fa5s.stop"))
                self.button_run = QtShortCuts.QPushButton(None, "run", self.run, icon=qta.icon("fa5s.play"))
            with QtShortCuts.QHBoxLayout():
                self.editor = CodeEditor()
                self.highlight = PythonHighlighter(self.editor.document())
                font = QtGui.QFont()
                font.setFamily('Courier')
                font.setFixedPitch(True)
                font.setPointSize(10)
                self.editor.setFont(font)

                self.console = Console(self, self.editor)#QtWidgets.QTextEdit()
                self.console.setStyleSheet("background-color: #1e1f22; color: #bcbec4")
                self.console.setReadOnly(True)
                self.console.setFont(font)

                self.splitter = QtWidgets.QSplitter().addToLayout()
                self.splitter.addWidget(self.editor)
                self.splitter.addWidget(self.console)

        self.console_update.connect(self.update_console)

        self.open_scripts = []
        self.open_scripts_index = -1

        self.editor.textChanged.connect(self.editor_text_changed)
        self.tabs.currentChanged.connect(self.select_tab)
        self.tabs.tabCloseRequested.connect(self.remove_tab)
        self.process_finished.connect(self.script_status_changed)

        self.select_tab(-1)

    def update_console(self, open_script):
        if len(self.open_scripts) and open_script == self.open_scripts[self.open_scripts_index]:
            self.console.setText(open_script.console)

    def load(self):
        new_path = QtWidgets.QFileDialog.getOpenFileName(None, "Open Script", os.getcwd(), "Python File (*.py)")
        if new_path:
            self.do_load(new_path)

    def do_load(self, open_script):
        # open a script
        self.open_scripts.append(OpenScript(self, open_script))
        # and add the tab
        self.tabs.addTab(self.open_scripts[-1].filename.name)
        # select the new tab
        self.tabs.setCurrentIndex(len(self.open_scripts)-1)

    def select_tab(self, index=None):
        if index is not None:
            self.open_scripts_index = index
        if index == -1 or len(self.open_scripts) == 0:
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
        self.editor.filename = str(open_script.filename)

        self.script_status_changed()

    def script_status_changed(self):
        if len(self.open_scripts) == 0:
            return
        open_script = self.open_scripts[self.open_scripts_index]
        if open_script.unsaved:
            self.tabs.setTabText(self.tabs.currentIndex(), open_script.filename.name + " *")
        else:
            self.tabs.setTabText(self.tabs.currentIndex(), open_script.filename.name)
        self.console.setHtml(open_script.console)

        self.button_run.setEnabled(open_script.can_run())
        self.button_stop.setEnabled(open_script.can_stop())

    def remove_tab(self, index):
        open_script = self.open_scripts.pop(index)
        open_script.stop()
        self.tabs.removeTab(index)

    def editor_text_changed(self):
        open_script = self.open_scripts[self.open_scripts_index]
        open_script.change_code(self.editor.toPlainText())
        self.script_status_changed()

    def stop(self):
        open_script = self.open_scripts[self.open_scripts_index]
        open_script.stop()

        self.script_status_changed()

    def run(self):
        open_script = self.open_scripts[self.open_scripts_index]
        open_script.run()

        self.script_status_changed()

    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        if a0.key() == QtCore.Qt.Key_F5:
            self.run()


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
            pass
            window.do_load(arg)
    window.show()
    sys.exit(app.exec_())
