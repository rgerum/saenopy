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
import imageio
from qimage2ndarray import array2qimage
import matplotlib.pyplot as plt
import glob
import imageio
import threading
from pathlib import Path
import saenopy.getDeformations
import saenopy.multigridHelper
import saenopy.materials
from saenopy.gui_deformation_whole2 import MainWindow as SolverMain
from saenopy.gui_deformation_spheriod import MainWindow as SpheriodMain
from saenopy.gui_orientation import MainWindow as OrientationMain

class InfoBox(QtWidgets.QWidget):
    def __init__(self, name, func):
        super().__init__()
        self.setMinimumWidth(200)
        self.setMaximumHeight(500)
        with QtShortCuts.QHBoxLayout(self) as l:
            with QtShortCuts.QGroupBox(l, name):
                with QtShortCuts.QVBoxLayout() as l2:
                    if name == "Solver":
                        self.text = QtWidgets.QLabel("Calculate the forces from a\n3D stack or a series of 3D stacks.").addToLayout()
                    elif name == "Spheriod":
                        self.text = QtWidgets.QLabel("Calculate the forces of\nmulticellular aggregates\nfrom a timeseries of 2D images.").addToLayout()
                    else:
                        self.text = QtWidgets.QLabel("Measure the orientations\nof fiberes in 2D images.\n\nAs a proxy for contractility.").addToLayout()
                    self.button1 = QtShortCuts.QPushButton(None, name, func)

class MainWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # QSettings
        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

        self.setMinimumWidth(800)
        self.setMinimumHeight(400)
        self.setWindowTitle("Saenopy")
        self.setWindowIcon(QtGui.QIcon(str(Path(__file__).parent / "img/Icon.ico")))

        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

        with QtShortCuts.QHBoxLayout(self) as main_layout:
            main_layout.setContentsMargins(0, 0, 0, 0)
            with QtShortCuts.QTabWidget(main_layout) as self.tabs:
                self.tabs.currentChanged.connect(self.changedTab)
                self.tabs.setTabPosition(QtWidgets.QTabWidget.West)
                with self.tabs.createTab("Home") as layout:
                    layout.addStretch()
                    with QtShortCuts.QHBoxLayout() as layout2:
                        layout.addStretch()
                        self.image = QtWidgets.QLabel("x").addToLayout()
                        self.image.setPixmap(QtGui.QPixmap(str(Path(__file__).parent / "img/Logo.png")))
                        self.image.setScaledContents(True)
                        self.image.setMaximumWidth(400)
                        self.image.setMaximumHeight(200)
                        layout.addStretch()
                    with QtShortCuts.QHBoxLayout() as layout2:
                        layout2.addStretch()
                        InfoBox("Solver", lambda: self.setTab(1)).addToLayout()
                        layout2.addStretch()
                        InfoBox("Spheriod", lambda: self.setTab(2)).addToLayout()
                        layout2.addStretch()
                        InfoBox("Orientation", lambda: self.setTab(3)).addToLayout()
                        layout2.addStretch()
                    layout.addStretch()
                with self.tabs.createTab("Solver") as self.layout_solver:
                    self.layout_solver.setContentsMargins(0, 0, 0, 0)
                with self.tabs.createTab("Spheriod") as self.layout_spheriod:
                    self.layout_spheriod.setContentsMargins(0, 0, 0, 0)
                with self.tabs.createTab("Orientation") as self.layout_orientation:
                    self.layout_orientation.setContentsMargins(0, 0, 0, 0)

        #self.tabs.setCurrentIndex(self.settings.value("master_tab", 0))
        self.first_tab_change = False

    first_tab_change = True
    solver = None
    spheriod = None
    orientation = None
    def changedTab(self, value):
        if self.first_tab_change is False:
            self.settings.setValue("master_tab", value)

        if value == 1 and self.solver is None:
            self.solver = SolverMain().addToLayout(self.layout_solver)
        if value == 2 and self.spheriod is None:
            self.spheriod = SpheriodMain().addToLayout(self.layout_spheriod)
        if value == 3 and self.orientation is None:
            self.orientation = OrientationMain().addToLayout(self.layout_orientation)

    def setTab(self, value):
        self.tabs.setCurrentIndex(value)


def main():
    app = QtWidgets.QApplication(sys.argv)
    if sys.platform.startswith('win'):
        import ctypes
        myappid = 'fabrylab.saenopy.master'  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    print(sys.argv)
    window = MainWindow()
    if len(sys.argv) >= 2:
        window.loadFile(sys.argv[1])
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
