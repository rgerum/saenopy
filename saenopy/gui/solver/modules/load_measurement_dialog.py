import sys
from qtpy import QtCore, QtWidgets, QtGui
import time

import saenopy
import saenopy.multigridHelper
import saenopy.getDeformations
import saenopy.multigridHelper
import saenopy.materials
from saenopy.gui.common import QtShortCuts
from saenopy.gui.common.stack_selector import StackSelector
from saenopy.gui.common.stack_selector_crop import StackSelectorCrop
from saenopy.gui.common.stack_preview import StackPreview

from saenopy.examples import getExamples


class AddFilesDialog(QtWidgets.QDialog):
    def __init__(self, parent, settings):
        super().__init__(parent)
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)
        self.setWindowTitle("Add Files")
        with QtShortCuts.QVBoxLayout(self) as layout:
            with QtShortCuts.QTabWidget(layout) as self.tabs:
                with self.tabs.createTab("New Measurement") as self.tab:
                    with QtShortCuts.QHBoxLayout():
                        with QtShortCuts.QVBoxLayout():
                            with QtShortCuts.QHBoxLayout():
                                self.reference_choice = QtShortCuts.QInputChoice(None, "Reference", 0, [0, 1],
                                                                                 ["difference between time points",
                                                                                  "single stack"])
                                QtShortCuts.current_layout.addStretch()
                            with QtShortCuts.QHBoxLayout():
                                with QtShortCuts.QHBoxLayout():
                                    with QtShortCuts.QVBoxLayout():
                                        QtShortCuts.current_layout.setContentsMargins(0, 0, 0, 0)
                                        self.place_holder_widget = QtWidgets.QWidget().addToLayout()
                                        layout_place_holder = QtWidgets.QVBoxLayout(self.place_holder_widget)
                                        layout_place_holder.addStretch()

                                        def ref_changed():
                                            self.stack_reference.setVisible(self.reference_choice.value())
                                            self.place_holder_widget.setVisible(not self.reference_choice.value())

                                        self.reference_choice.valueChanged.connect(ref_changed)
                                        self.stack_reference = StackSelector(QtShortCuts.current_layout, "reference")
                                        self.stack_reference.glob_string_changed.connect \
                                            (lambda x, y: (print("relaxed, y"), self.stack_reference_input.setText(y)))
                                        self.stack_reference.setVisible(self.reference_choice.value())

                                        self.stack_reference_input = QtWidgets.QLineEdit().addToLayout()
                                    with QtShortCuts.QVBoxLayout():
                                        QtShortCuts.current_layout.setContentsMargins(0, 0, 0, 0)
                                        self.stack_data = StackSelector(QtShortCuts.current_layout, "active stack(s)",
                                                                        self.stack_reference, use_time=True)
                                        self.stack_data.setMinimumWidth(300)
                                        self.stack_reference.setMinimumWidth(300)
                                        self.place_holder_widget.setMinimumWidth(300)
                                        self.stack_data.glob_string_changed.connect(
                                            lambda x, y: self.stack_data_input.setText(y))
                                        self.stack_data_input = QtWidgets.QLineEdit().addToLayout()
                            self.stack_crop = StackSelectorCrop(self.stack_data, self.stack_reference).addToLayout()
                            self.stack_data.stack_crop = self.stack_crop
                        self.stack_preview = StackPreview(QtShortCuts.current_layout, self.reference_choice,
                                                          self.stack_reference, self.stack_data)
                    self.outputText = QtShortCuts.QInputFolder(None, "output", settings=settings,
                                                               settings_key="batch/wildcard2", allow_edit=True)
                    with QtShortCuts.QHBoxLayout():
                        # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                        QtShortCuts.current_layout.addStretch()
                        self.button_addList00 = QtShortCuts.QPushButton(None, "cancel", self.reject)

                        def accept():
                            self.mode = "new"
                            if self.reference_choice.value() == 1:
                                if self.stack_reference.active is None:
                                    QtWidgets.QMessageBox.critical(self, "Deformation Detector",
                                                                   "Provide a stack for the reference state.")
                                    return
                            if self.stack_data.active is None:
                                QtWidgets.QMessageBox.critical(self, "Deformation Detector",
                                                               "Provide a stack for the deformed state.")
                                return
                            if self.stack_data.get_t_count() <= 1 and self.stack_reference.active is None:
                                QtWidgets.QMessageBox.critical(self, "Deformation Detector",
                                                               "Provide either a reference stack or a time sequence.")
                                return
                            if not self.stack_crop.validator():
                                QtWidgets.QMessageBox.critical(self, "Deformation Detector",
                                                               "Enter a valid voxel size.")
                                return
                            if "{t}" in self.stack_data_input.text() and not self.stack_crop.validator_time():
                                QtWidgets.QMessageBox.critical(self, "Deformation Detector",
                                                               "Enter a valid time delta.")
                                return
                            self.accept()

                        self.button_addList0 = QtShortCuts.QPushButton(None, "ok", accept)

                with self.tabs.createTab("Existing Measurement") as self.tab3:
                    self.outputText3 = QtShortCuts.QInputFilename(None, "output", settings=settings,
                                                                  file_type="Results Files (*.npz)",
                                                                  settings_key="batch/wildcard_existing",
                                                                  allow_edit=True, existing=True)
                    self.tab3.addStretch()
                    with QtShortCuts.QHBoxLayout() as layout3:
                        # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                        layout3.addStretch()
                        self.button_addList6 = QtShortCuts.QPushButton(None, "cancel", self.reject)

                        def accept():
                            self.mode = "existing"
                            self.accept()

                        self.button_addList5 = QtShortCuts.QPushButton(None, "ok", accept)

                with self.tabs.createTab("Examples") as self.tab4:
                    start_time = 0

                    def reporthook(count, block_size, total_size, msg=None):
                        nonlocal start_time
                        if msg is not None:
                            print(msg)
                            self.download_state.setText(msg)
                            return
                        if count == 0:
                            start_time = time.time()
                            return
                        duration = time.time() - start_time
                        progress_size = int(count * block_size)
                        speed = int(progress_size / (1024 * duration + 0.001))
                        percent = int(count * block_size * 100 / total_size)
                        sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                                         (percent, progress_size / (1024 * 1024), speed, duration))
                        sys.stdout.flush()
                        self.download_state.setText("...%d%%, %d MB, %d KB/s, %d seconds passed" %
                                                    (percent, progress_size / (1024 * 1024), speed, duration))
                        self.download_progress.setValue(percent)

                    examples = getExamples()
                    with QtShortCuts.QHBoxLayout() as lay:
                        for example_name, properties in examples.items():
                            def load_example(example_name):
                                saenopy.loadExample(example_name, None, reporthook)
                                self.mode = "example"
                                self.mode_data = example_name
                                self.accept()

                            with QtShortCuts.QGroupBox(None, example_name) as group:
                                group[0].setMaximumWidth(240)
                                label1 = QtWidgets.QLabel \
                                    (properties["desc"]).addToLayout()
                                label1.setWordWrap(True)
                                label = QtWidgets.QLabel().addToLayout()
                                pix = QtGui.QPixmap(
                                    str(properties["img"]))
                                pix = pix.scaledToWidth(
                                    int(200 * QtGui.QGuiApplication.primaryScreen().logicalDotsPerInch() / 96),
                                    QtCore.Qt.SmoothTransformation)
                                label.setPixmap(pix)
                                label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
                                self.button_example1 = QtShortCuts.QPushButton(None, "Open",
                                                           lambda *, example_name=example_name: load_example(example_name))
                        lay.addStretch()

                    self.tab4.addStretch()
                    self.download_state = QtWidgets.QLabel("").addToLayout()
                    self.download_progress = QtWidgets.QProgressBar().addToLayout()
                    self.download_progress.setRange(0, 100)


class FileExistsDialog(QtWidgets.QDialog):
    def __init__(self, parent, filename):
        super().__init__(parent)
        self.setWindowTitle("File Exists")
        with QtShortCuts.QVBoxLayout(self):
            self.label = QtShortCuts.SuperQLabel(f"A file with the name {filename} already exists.").addToLayout()
            self.label.setWordWrap(True)
            with QtShortCuts.QHBoxLayout():
                self.use_for_all = QtShortCuts.QInputBool(None, "remember decision for all files", False)
            with QtShortCuts.QHBoxLayout():
                self.button_addList2 = QtShortCuts.QPushButton(None, "cancel", self.reject)

                def accept():
                    self.mode = "overwrite"
                    self.accept()

                self.button_addList1 = QtShortCuts.QPushButton(None, "overwrite", accept)

                def accept2():
                    self.mode = "read"
                    self.accept()

                self.button_addList1 = QtShortCuts.QPushButton(None, "read", accept2)
