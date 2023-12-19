import sys

# Setting the Qt bindings for QtPy
import os

import pandas as pd
import qtawesome as qta
from qtpy import QtCore, QtWidgets, QtGui
import numpy as np
from saenopy.gui.common import QtShortCuts, QExtendedGraphicsView
from saenopy.gui.common.gui_classes import Spoiler, CheckAbleGroup, QHLine, QVLine, MatplotlibWidget, NavigationToolbar, execute, kill_thread, ListWidget
import imageio
from qimage2ndarray import array2qimage
import matplotlib.pyplot as plt
import glob
import imageio
import threading
import glob
import re
from pathlib import Path
import saenopy
from saenopy.examples import get_examples_orientation


class AddFilesDialog(QtWidgets.QDialog):
    def __init__(self, parent, settings):
        super().__init__(parent)
        self.setWindowTitle("Add Files")
        with QtShortCuts.QVBoxLayout(self) as layout:
            with QtShortCuts.QTabWidget(layout) as self.tabs:
                with self.tabs.createTab("New Measurement") as self.tab:
                    with QtShortCuts.QVBoxLayout() as layout:
                        self.label = QtWidgets.QLabel(
                            "Select two paths as an input wildcard. Use * to specify a placeholder. One should be for the fiber images and one for the cell images.")
                        layout.addWidget(self.label)

                        self.cellText = QtShortCuts.QInputFilename(None, "Cell Images", file_type="Image (*.tif *.png *.jpg)",
                                                                   settings=settings,
                                                                   settings_key="batch/wildcard_cell", existing=True,
                                                                   allow_edit=True)
                        self.fiberText = QtShortCuts.QInputFilename(None, "Fiber Images", file_type="Image (*.tif *.png *.jpg)",
                                                                    settings=settings,
                                                                    settings_key="batch/wildcard_fiber", existing=True,
                                                                    allow_edit=True)
                        self.outputText = QtShortCuts.QInputFolder(None, "output", settings=settings,
                                                                   settings_key="batch/output_folder", allow_edit=True)

                        def changed():
                            from CompactionAnalyzer.CompactionFunctions import generate_lists
                            fiber_list_string = os.path.normpath(self.fiberText.value())
                            cell_list_string = os.path.normpath(self.cellText.value())
                            output_folder = os.path.normpath(self.outputText.value())
                            fiber_list, cell_list, out_list = generate_lists(fiber_list_string, cell_list_string,
                                                                             output_main=output_folder)
                            if self.fiberText.value() == "" or self.cellText.value() == "":
                                self.label2.setText("")
                                self.label2.setStyleSheet("QLabel { color : red; }")
                                self.button_addList1.setDisabled(True)
                            elif len(fiber_list) != len(cell_list):
                                self.label2.setText(
                                    f"Warning: {len(fiber_list)} fiber images found and {len(cell_list)} cell images found. Numbers do not match.")
                                self.label2.setStyleSheet("QLabel { color : red; }")
                                self.button_addList1.setDisabled(True)
                            else:
                                if "*" not in fiber_list_string:
                                    if len(fiber_list) == 0:
                                        self.label2.setText(f"'Fiber Images' not found")
                                        self.label2.setStyleSheet("QLabel { color : red; }")
                                        self.button_addList1.setDisabled(True)
                                    elif len(cell_list) == 0:
                                        self.label2.setText(f"'Cell Images' not found")
                                        self.label2.setStyleSheet("QLabel { color : red; }")
                                        self.button_addList1.setDisabled(True)
                                    else:
                                        self.label2.setText(f"No * found in 'Fiber Images', will only import a single image.")
                                        self.label2.setStyleSheet("QLabel { color : orange; }")
                                        self.button_addList1.setDisabled(False)
                                else:
                                    self.label2.setText(
                                        f"{len(fiber_list)} fiber images found and {len(cell_list)} cell images found.")
                                    self.label2.setStyleSheet("QLabel { color : green; }")
                                    self.button_addList1.setDisabled(False)

                        self.fiberText.line.textChanged.connect(changed)
                        self.cellText.line.textChanged.connect(changed)
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
                    examples = get_examples_orientation()
                    self.example_buttons = []
                    with QtShortCuts.QHBoxLayout() as lay:
                        for example_name, properties in examples.items():
                            with QtShortCuts.QGroupBox(None, example_name) as group:
                                group[0].setMaximumWidth(240)
                                label1 = QtWidgets.QLabel(properties["desc"]).addToLayout(QtShortCuts.current_layout)
                                label1.setWordWrap(True)
                                label = QtWidgets.QLabel().addToLayout(QtShortCuts.current_layout)
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
                    self.download_state = QtWidgets.QLabel("").addToLayout(QtShortCuts.current_layout)
                    self.download_progress = QtWidgets.QProgressBar().addToLayout(QtShortCuts.current_layout)
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
