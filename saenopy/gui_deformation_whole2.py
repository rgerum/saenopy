import sys

# Setting the Qt bindings for QtPy
import os

os.environ["QT_API"] = "pyqt5"

from qtpy import QtCore, QtWidgets, QtGui

import numpy as np

import pyvista as pv
import vtk
from pyvistaqt import QtInteractor

import saenopy
import saenopy.multigridHelper
from saenopy.gui import QtShortCuts, QExtendedGraphicsView
from saenopy.gui.stack_selector import StackSelector
from saenopy.gui.gui_classes import Spoiler, CheckAbleGroup, QHLine, QVLine, MatplotlibWidget, execute, kill_thread, ListWidget
import imageio
from qimage2ndarray import array2qimage
import matplotlib.pyplot as plt
import glob
import imageio
import threading
import saenopy.getDeformations
import saenopy.multigridHelper
import saenopy.materials
from saenopy.gui.stack_selector import StackSelector
import matplotlib as mpl
from pathlib import Path
import re

""" some magic to prevent PyQt5 from swallowing exceptions """
# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook
# Set the exception hook to our wrapping function
sys.excepthook = lambda *args: sys._excepthook(*args)

class Pipeline:
    stack_deformed = None
    stack_relaxed = None
    solver = None

    def __init__(self):
        pass

def double_glob(text):
    glob_string = text.replace("?", "*")
    print("globbing", glob_string)
    files = glob.glob(glob_string)

    output_base = glob_string
    while "*" in str(output_base):
        output_base = Path(output_base).parent

    regex_string = re.escape(text).replace("\*", "(.*)").replace("\?", ".*")

    results = []
    for file in files:
        file = os.path.normpath(file)
        print(file, regex_string)
        match = re.match(regex_string, file).groups()
        reconstructed_file = regex_string
        for element in match:
            reconstructed_file = reconstructed_file.replace("(.*)", element, 1)
        reconstructed_file = reconstructed_file.replace(".*", "*")
        reconstructed_file = re.sub(r'\\(.)', r'\1', reconstructed_file)
        if reconstructed_file not in results:
            results.append(reconstructed_file)
    return results, output_base


class BatchEvaluate(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.settings = QtCore.QSettings("Saenopy", "Seanopy_deformation")

        with QtShortCuts.QHBoxLayout(self) as main_layout:
            with QtShortCuts.QSplitter() as lay:
                with QtShortCuts.QVBoxLayout() as layout:
                    layout.setContentsMargins(0, 0, 0, 0)
                    self.list = ListWidget(layout, add_item_button="add measurements")
                    self.list.addItemClicked.connect(self.show_files)
                    self.list.itemSelectionChanged.connect(self.listSelected)
                    self.progress1 = QtWidgets.QProgressBar()
                    layout.addWidget(self.progress1)
         #           self.label = QtWidgets.QLabel(
         #               "Select a path as an input wildcard. Use * to specify a placeholder. All paths that match the wildcard will be added.").addToLayout()

        self.data = []
        self.list.setData(self.data)
        #self.list.addData("foo", True, [], mpl.colors.to_hex(f"C0"))

    def show_files(self):
        settings = self.settings

        class AddFilesDialog(QtWidgets.QDialog):
            def __init__(self, parent):
                super().__init__(parent)
                self.setMinimumWidth(800)
                self.setMinimumHeight(600)
                self.setWindowTitle("Add Files")
                with QtShortCuts.QVBoxLayout(self) as layout:
                    self.outputText = QtShortCuts.QInputFolder(None, "output", settings=settings,
                                                               settings_key="batch/wildcard2", allow_edit=True)
                    with QtShortCuts.QHBoxLayout() as layout3:
                        self.stack_before = StackSelector(layout3, "deformed")
                        self.stack_before.glob_string_changed.connect(lambda x, y: self.input_deformed.setText(y.replace("*", "?")))
                        self.stack_after = StackSelector(layout3, "relaxed", self.stack_before)
                        self.stack_after.glob_string_changed.connect(lambda x, y: self.input_relaxed.setText(y.replace("*", "?")))
                    with QtShortCuts.QHBoxLayout() as layout3:
                        self.input_deformed = QtWidgets.QLineEdit().addToLayout()
                        self.input_relaxed = QtWidgets.QLineEdit().addToLayout()
                        self.input_relaxed.text()
                    with QtShortCuts.QHBoxLayout() as layout3:
                        # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                        layout3.addStretch()
                        self.button_addList2 = QtShortCuts.QPushButton(None, "cancel", self.reject)
                        self.button_addList1 = QtShortCuts.QPushButton(None, "ok", self.accept)
        #getStack
        dialog = AddFilesDialog(self)
        if not dialog.exec():
            return

        results1, output_base = double_glob(dialog.input_deformed.text())
        results2, _ = double_glob(dialog.input_relaxed.text())

        for r1, r2 in zip(results1, results2):
            output = Path(dialog.outputText.value()) / os.path.relpath(r1, output_base)
            output = output.parent / output.stem
            data = dict(
                output=output,
                r1=r1,
                r2=r2,
            )
            print(r1, r2)
            self.list.addData(r1, True, data, mpl.colors.to_hex(f"gray"))

        #import matplotlib as mpl
        #for fiber, cell, out in zip(fiber_list, cell_list, out_list):
        #    self.list.addData(fiber, True, [fiber, cell, out, {"segmention_thres": None, "seg_gaus1": None, "seg_gaus2": None}], mpl.colors.to_hex(f"gray"))

    def listSelected(self):
        pass

class MainWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # QSettings
        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

        self.setMinimumWidth(800)
        self.setMinimumHeight(400)
        self.setWindowTitle("Saenopy Viewer")

        main_layout = QtWidgets.QHBoxLayout(self)

        with QtShortCuts.QTabWidget(main_layout) as self.tabs:
            """ """
            with self.tabs.createTab("Compaction") as v_layout:
                with QtShortCuts.QHBoxLayout() as h_layout:
                    # self.deformations = Deformation(h_layout, self)
                    self.deformations = BatchEvaluate(self)
                    h_layout.addWidget(self.deformations)
                    self.description = QtWidgets.QTextEdit()
                    self.description.setDisabled(True)
                    self.description.setMaximumWidth(300)
                    h_layout.addWidget(self.description)
                    self.description.setText("""
                    <h1>Start Evaluation</h1>
                     """.strip())
                v_layout.addWidget(QHLine())
                with QtShortCuts.QHBoxLayout() as h_layout:
                    h_layout.addStretch()
                    #self.button_previous = QtShortCuts.QPushButton(None, "back", self.previous)
                    #self.button_next = QtShortCuts.QPushButton(None, "next", self.next)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    print(sys.argv)
    window = MainWindow()
    if len(sys.argv) >= 2:
        window.loadFile(sys.argv[1])
    window.show()
    sys.exit(app.exec_())
