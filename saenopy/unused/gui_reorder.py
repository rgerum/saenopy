import sys

# Setting the Qt bindings for QtPy
import os

import pandas as pd

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

# \\131.188.117.96\biophysDS\lbischof\tif_and_analysis_backup\2021-06-21-NK92_BlebbRock\Blebb_round4\Mark_and_Find_001
""" some magic to prevent PyQt5 from swallowing exceptions """
# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook
# Set the exception hook to our wrapping function
sys.excepthook = lambda *args: sys._excepthook(*args)

def format_glob(pattern):
    pattern = str(Path(pattern))
    regexp_string = re.sub(r"\\{([^}]*)\\}", r"(?P<\1>.*)", re.escape(pattern).replace("\\*\\*", ".*").replace("\\*", ".*"))
    regexp_string2 = re.compile(regexp_string)
    glob_string = re.sub(r"({[^}]*})", "*", pattern)

    file_list = []
    for file in glob.glob(glob_string, recursive=True):
        file = str(Path(file))
        group = regexp_string2.match(file).groupdict()
        print(file, group)
        file_list.append([file, group])
    return file_list

class MainWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # QSettings
        self.settings = QtCore.QSettings("Saenopy", "SeanopyReorder")

        self.setMinimumWidth(800)
        self.setMinimumHeight(400)
        self.setWindowTitle("Saenopy Viewer")

        with QtShortCuts.QVBoxLayout(self):
            self.input = QtShortCuts.QInputFilename(None, "Input folder", existing=True, file_type="images (*.tif *.jpg *.png *.lif)", allow_edit=True, settings=self.settings, settings_key='inut')
            self.input_fov = QtShortCuts.QInputString(None, "FOV", settings=self.settings, settings_key='fov')
            self.input_z = QtShortCuts.QInputString(None, "z", settings=self.settings, settings_key='z')
            self.input_time = QtShortCuts.QInputString(None, "time", settings=self.settings, settings_key='time')
            self.input_meta = QtWidgets.QTextEdit().addToLayout()

            with QtShortCuts.QHBoxLayout() as layout:
                layout.addStretch()
                QtShortCuts.QPushButton(None, 'test', connect=self.update)

    def update(self):
        file_list = format_glob(self.input.value())
        data = []
        for file, group in file_list:
            z = eval(self.input_z.value().format(**group))
            fov = eval(self.input_fov.value().format(**group))
            time = eval(self.input_time.value().format(**group))
            data.append(dict(file=file, z=z, fov=fov, time=time))
            print(file, z, fov, time)
        data = pd.DataFrame(data)
        print(data)



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    print(sys.argv)
    window = MainWindow()
    if len(sys.argv) >= 2:
        window.loadFile(sys.argv[1])
    window.show()
    sys.exit(app.exec_())
