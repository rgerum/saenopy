import os

from qtpy import QtCore, QtWidgets, QtGui

import numpy as np
from natsort import natsorted

from saenopy.gui.common import QtShortCuts
from saenopy.examples import get_examples_spheroid
from saenopy.gui.common.AddFilesDialog import AddFilesDialog


class AddFilesDialog(AddFilesDialog):
    settings_group = "open_spheroid"
    file_extension = ".saenopySpheroid"

    examples_list = get_examples_spheroid()

    def add_new_measurement_tab(self):
        with self.tabs.createTab("New Measurement") as self.tab:
            with QtShortCuts.QVBoxLayout() as layout:
                self.label = QtWidgets.QLabel(
                    "Select a path as an input wildcard.<br/>Use a <b>? placeholder</b> to specify <b>different timestamps</b> of a measurement.<br/>Optionally use a <b>* placeholder</b> to specify <b>different measurements</b> to load.")
                layout.addWidget(self.label)

                self.inputText = QtShortCuts.QInputFilename(None, "input", file_type="Image (*.tif *.png *.jpg)",
                                                            settings=self.settings,
                                                            settings_key=f"{self.settings_group}/wildcard", existing=True,
                                                            allow_edit=True)
                self.pixel_size = QtShortCuts.QInputString(None, "pixel size", 1.29, settings=self.settings,
                                                           settings_key=f"{self.settings_group}/pixel_size", unit="µm",
                                                           allow_none=False, type=float)
                self.time_delta = QtShortCuts.QInputString(None, "time delta", 120, settings=self.settings,
                                                           settings_key=f"{self.settings_group}/delta_t", unit="s",
                                                           allow_none=False, type=float)
                self.outputText = QtShortCuts.QInputFolder(None, "output", settings=self.settings,
                                                           settings_key=f"{self.settings_group}/wildcard2", allow_edit=True)

                def changed():
                    import glob, re
                    text = os.path.normpath(self.inputText.value())

                    glob_string = text.replace("?", "*")
                    files = natsorted(glob.glob(glob_string))

                    regex_string = re.escape(text).replace(r"\*", "(.*)").replace(r"\?", ".*")

                    data = {}
                    for file in files:
                        file = os.path.normpath(file)
                        match = re.match(regex_string, file).groups()
                        reconstructed_file = regex_string
                        for element in match:
                            reconstructed_file = reconstructed_file.replace("(.*)", element, 1)
                        reconstructed_file = reconstructed_file.replace(".*", "*")
                        reconstructed_file = re.sub(r'\\(.)', r'\1', reconstructed_file)

                        if reconstructed_file not in data:
                            data[reconstructed_file] = 0
                        data[reconstructed_file] += 1

                    counts = [v for v in data.values()]
                    if len(counts):
                        min_count = np.min(counts)
                        max_count = np.max(counts)
                    else:
                        min_count = 0
                        max_count = 0

                    if text == "":
                        self.label2.setText("")
                        self.label2.setStyleSheet("QLabel { color : red; }")
                        self.button_addList1.setDisabled(True)
                    elif max_count == 0:
                        self.label2.setText("No images found.")
                        self.label2.setStyleSheet("QLabel { color : red; }")
                        self.button_addList1.setDisabled(True)
                    elif max_count == 1:
                        self.label2.setText(
                            f"Found {len(counts)} measurements with {max_count} images.<br>Maybe check the wildcard placeholder. Did you forget the : placeholder?")
                        self.label2.setStyleSheet("QLabel { color : orange; }")
                        self.button_addList1.setDisabled(True)
                    elif min_count == max_count:
                        self.label2.setText(
                            f"Found {len(counts)} measurements with each {min_count} images.")
                        self.label2.setStyleSheet("QLabel { color : green; }")
                        self.button_addList1.setDisabled(False)
                    else:
                        self.label2.setText(
                            f"Found {len(counts)} measurements with {counts} images.")
                        self.label2.setStyleSheet("QLabel { color : green; }")
                        self.button_addList1.setDisabled(False)

                self.inputText.line.textChanged.connect(changed)

                self.label2 = QtWidgets.QLabel()  # .addToLayout()
                layout.addWidget(self.label2)
                layout.addStretch()

                with QtShortCuts.QHBoxLayout() as layout3:
                    # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                    layout3.addStretch()
                    self.button_addList2 = QtShortCuts.QPushButton(None, "cancel", self.reject)
                    self.button_addList1 = QtShortCuts.QPushButton(None, "ok", self.accept_new)
                changed()
