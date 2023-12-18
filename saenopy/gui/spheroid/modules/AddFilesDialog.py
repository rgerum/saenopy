import sys

# Setting the Qt bindings for QtPy
import os

import pandas as pd
import qtawesome as qta

os.environ["QT_API"] = "pyqt5"

from qtpy import QtCore, QtWidgets, QtGui

import numpy as np
from natsort import natsorted


from saenopy.gui.common import QtShortCuts, QExtendedGraphicsView
from qimage2ndarray import array2qimage
import matplotlib.pyplot as plt
import imageio
import threading
import glob
import time

from matplotlib.figure import Figure
import jointforces as jf
import urllib
from pathlib import Path

import ctypes
import saenopy
from saenopy.examples import get_examples_spheriod

class AddFilesDialog(QtWidgets.QDialog):
    def __init__(self, parent, settings):
        super().__init__(parent)
        self.setWindowTitle("Add Files")
        with QtShortCuts.QVBoxLayout(self) as layout:
            with QtShortCuts.QTabWidget(layout) as self.tabs:
                with self.tabs.createTab("New Measurement") as self.tab:
                    with QtShortCuts.QVBoxLayout() as layout:
                        self.label = QtWidgets.QLabel(
                            "Select a path as an input wildcard.<br/>Use a <b>? placeholder</b> to specify <b>different timestamps</b> of a measurement.<br/>Optionally use a <b>* placeholder</b> to specify <b>different measurements</b> to load.")
                        layout.addWidget(self.label)

                        self.inputText = QtShortCuts.QInputFilename(None, "input", file_type="Image (*.tif *.png *.jpg)",
                                                                    settings=settings,
                                                                    settings_key="batch/wildcard", existing=True,
                                                                    allow_edit=True)
                        self.outputText = QtShortCuts.QInputFolder(None, "output", settings=settings,
                                                                   settings_key="batch/wildcard2", allow_edit=True)

                        def changed():
                            import glob, re
                            text = os.path.normpath(self.inputText.value())

                            glob_string = text.replace("?", "*")
                            files = natsorted(glob.glob(glob_string))

                            regex_string = re.escape(text).replace(r"\*", "(.*)").replace(r"\?", ".*")

                            data = {}
                            for file in files:
                                file = os.path.normpath(file)
                                print(file, regex_string)
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

                with self.tabs.createTab("Examples") as self.tab4:
                    examples = get_examples_spheriod()
                    self.example_buttons = []
                    with QtShortCuts.QHBoxLayout() as lay:
                        for example_name, properties in examples.items():
                            with QtShortCuts.QGroupBox(None, example_name) as group:
                                group[0].setMaximumWidth(240)
                                label1 = QtWidgets.QLabel(properties["desc"]).addToLayout()
                                label1.setWordWrap(True)
                                label = QtWidgets.QLabel().addToLayout()
                                pix = QtGui.QPixmap(str(properties["img"]))
                                pix = pix.scaledToWidth(
                                    int(200 * QtGui.QGuiApplication.primaryScreen().logicalDotsPerInch() / 96),
                                    QtCore.Qt.SmoothTransformation)
                                label.setPixmap(pix)
                                label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
                                self.button_example1 = QtShortCuts.QPushButton(None, "Open",
                                                           lambda *, example_name=example_name: self.load_example(example_name))
                                self.example_buttons.append(self.button_example1)
                                #self.button_example2 = QtShortCuts.QPushButton(None, "Open (evaluated)",
                                #                           lambda *, example_name=example_name: self.load_example(
                                #                                                   example_name, evaluated=True))
                                #self.button_example2.setEnabled(properties.get("url_evaluated", None) is not None)
                                #self.example_buttons.append(self.button_example2)
                        lay.addStretch()

                    self.tab4.addStretch()
                    self.download_state = QtWidgets.QLabel("").addToLayout()
                    self.download_progress = QtWidgets.QProgressBar().addToLayout()
                    self.download_progress.setRange(0, 100)

    def accept_new(self):
        self.mode = "new"
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
        if total_size == -1:
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
