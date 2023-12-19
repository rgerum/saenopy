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
from saenopy.gui.common.AddFilesDialog import AddFilesDialog


class AddFilesDialog(AddFilesDialog):
    settings_group = "open_orientation"

    examples_list = get_examples_orientation()

    def add_new_measurement_tab(self):
        with self.tabs.createTab("New Measurement") as self.tab:
            with QtShortCuts.QVBoxLayout() as layout:
                self.label = QtWidgets.QLabel(
                    "Select two paths as an input wildcard. Use * to specify a placeholder. One should be for the fiber images and one for the cell images.")
                layout.addWidget(self.label)

                self.cellText = QtShortCuts.QInputFilename(None, "Cell Images", file_type="Image (*.tif *.png *.jpg)",
                                                           settings=self.settings,
                                                           settings_key=f"{self.settings_group}/wildcard_cell", existing=True,
                                                           allow_edit=True)
                self.fiberText = QtShortCuts.QInputFilename(None, "Fiber Images", file_type="Image (*.tif *.png *.jpg)",
                                                            settings=self.settings,
                                                            settings_key=f"{self.settings_group}/wildcard_fiber", existing=True,
                                                            allow_edit=True)
                self.outputText = QtShortCuts.QInputFolder(None, "output", settings=self.settings,
                                                           settings_key=f"{self.settings_group}/output_folder", allow_edit=True)

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
