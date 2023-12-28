import sys
import os
import time
from qtpy import QtCore, QtWidgets, QtGui
from saenopy.gui.common import QtShortCuts
import saenopy


class AddFilesDialog(QtWidgets.QDialog):
    mode: str = None
    mode_data: str = None
    start_time: float = 0

    file_extension = None
    settings_group = "batch"

    examples_list = None

    def add_new_measurement_tab(self):
        pass

    def add_existing_measurent_tab(self):
        if self.file_extension is None:
            return
        with self.tabs.createTab("Existing Measurement") as self.tab3:
            self.outputText3 = QtShortCuts.QInputFilename(None, "output", settings=self.settings,
                                                          file_type=f"Results Files (*{self.file_extension})",
                                                          settings_key=f"{self.settings_group}/wildcard_existing",
                                                          allow_edit=True, existing=True)
            self.tab3.addStretch()
            with QtShortCuts.QHBoxLayout() as layout3:
                layout3.addStretch()
                self.button_addList6 = QtShortCuts.QPushButton(None, "cancel", self.reject)

                self.button_addList5 = QtShortCuts.QPushButton(None, "ok", self.accept_existing)

    def add_examples_tab(self):
        if self.examples_list is None:
            return
        with self.tabs.createTab("Examples") as self.tab4:
            examples = self.examples_list
            self.example_buttons = []
            with QtShortCuts.QHBoxLayout() as lay:
                for example_name, properties in examples.items():
                    with QtShortCuts.QGroupBox(None, example_name) as group:
                        group[0].setMaximumWidth(240)
                        label1 = QtWidgets.QLabel(properties["desc"]).addToLayout(QtShortCuts.currentLayout())
                        label1.setWordWrap(True)
                        label = QtWidgets.QLabel().addToLayout(QtShortCuts.currentLayout())
                        pix = QtGui.QPixmap(str(properties["img"]))
                        pix = pix.scaledToWidth(
                            int(200 * QtGui.QGuiApplication.primaryScreen().logicalDotsPerInch() / 96),
                            QtCore.Qt.SmoothTransformation)
                        label.setPixmap(pix)
                        label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
                        self.button_example1 = QtShortCuts.QPushButton(None, "Open",
                                                                       lambda *,
                                                                              example_name=example_name: self.load_example(
                                                                           example_name))
                        self.example_buttons.append(self.button_example1)

                        self.button_example2 = QtShortCuts.QPushButton(None, "Open (evaluated)",
                                                                       lambda *,
                                                                              example_name=example_name: self.load_example(
                                                                           example_name, evaluated=True))
                        self.button_example2.setEnabled(properties.get("url_evaluated", None) is not None)
                        self.example_buttons.append(self.button_example2)
                lay.addStretch()

            self.tab4.addStretch()
            self.download_state = QtWidgets.QLabel("").addToLayout(QtShortCuts.currentLayout())
            self.download_progress = QtWidgets.QProgressBar().addToLayout(QtShortCuts.currentLayout())
            self.download_progress.setRange(0, 100)

    def __init__(self, parent, settings):
        super().__init__(parent)
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)
        self.setWindowTitle("Add Files")
        self.settings = settings
        with QtShortCuts.QVBoxLayout(self) as layout:
            with QtShortCuts.QTabWidget(layout) as self.tabs:
                self.add_new_measurement_tab()
                self.add_existing_measurent_tab()
                self.add_examples_tab()


    def accept_new(self):
        self.mode = "new"
        self.accept()

    def accept_existing(self):
        self.mode = "existing"
        self.accept()

    def load_example(self, example_name, evaluated=False):
        self.examples_output = saenopy.load_example(example_name, None, self.reporthook, evaluated=evaluated)
        if evaluated:
            self.mode = "example_evaluated"
        else:
            self.mode = "example"
        self.mode_data = example_name
        self.accept()

    def reporthook(self, count, block_size, total_size, msg=None):
        if msg is not None:
            print(msg)
            self.download_state.setText(msg)
            return
        if count == 0:
            self.start_time = time.time()
            return
        duration = time.time() - self.start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration + 0.001))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                         (percent, progress_size / (1024 * 1024), speed, duration))
        sys.stdout.flush()
        self.download_state.setText("...%d%%, %d MB, %d KB/s, %d seconds passed" %
                                    (percent, progress_size / (1024 * 1024), speed, duration))
        self.download_progress.setValue(percent)


last_decision = None
def do_overwrite(filename, self):
    global last_decision

    # if we are in demo mode always load the files
    if os.environ.get("DEMO") == "true":  # pragma: no cover
        return "read"

    # if there is a last decistion stored use that
    if last_decision is not None:
        return last_decision

    # ask the user if they want to overwrite or read the existing file
    dialog = FileExistsDialog(self, filename)
    result = dialog.exec()
    # if the user clicked cancel
    if not result:
        return 0
    # if the user wants to remember the last decision
    if dialog.use_for_all.value():
        last_decision = dialog.mode
    # return the decision
    return dialog.mode

class FileExistsDialog(QtWidgets.QDialog):
    mode: str = None

    def __init__(self, parent, filename):
        super().__init__(parent)
        self.setWindowTitle("File Exists")
        with QtShortCuts.QVBoxLayout(self):
            self.label = QtShortCuts.SuperQLabel(f"A file with the name {filename} already exists.").addToLayout()
            self.label.setWordWrap(True)
            with QtShortCuts.QHBoxLayout():
                self.use_for_all = QtShortCuts.QInputBool(None, "remember decision for all files", False)
            with QtShortCuts.QHBoxLayout():
                self.button_addList0 = QtShortCuts.QPushButton(None, "cancel", self.reject)

                self.button_addList1 = QtShortCuts.QPushButton(None, "overwrite", self.accept_overwrite)

                self.button_addList2 = QtShortCuts.QPushButton(None, "read", self.accept_read)

    def accept_overwrite(self):
        self.mode = "overwrite"
        self.accept()

    def accept_read(self):
        self.mode = "read"
        self.accept()
