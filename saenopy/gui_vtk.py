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
import imageio
from qimage2ndarray import array2qimage
import matplotlib.pyplot as plt

""" some magic to prevent PyQt5 from swallowing exceptions """
# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook
# Set the exception hook to our wrapping function
sys.excepthook = lambda *args: sys._excepthook(*args)


class MainWindow(QtWidgets.QMainWindow):
    vector_field = None
    scale = 1
    im = None

    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        # QSettings
        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

        self.setMinimumWidth(800)
        self.setMinimumHeight(400)
        self.setWindowTitle("Saenopy Viewer")

        # create the frame
        self.frame = QtWidgets.QFrame()
        hlayout = QtWidgets.QHBoxLayout()
        vlayout = QtWidgets.QVBoxLayout()
        hlayout.addLayout(vlayout)

        layout_vert_plot = QtWidgets.QHBoxLayout()
        vlayout.addLayout(layout_vert_plot)

        self.input_checks = {}
        for name in ["U_target", "U", "f", "stiffness"]:
            input = QtShortCuts.QInputBool(layout_vert_plot, name, True)
            input.valueChanged.connect(self.replot)
            self.input_checks[name] = input
        layout_vert_plot.addStretch()

        # add the pyvista interactor object
        self.plotter_layout = QtWidgets.QHBoxLayout()
        self.plotter = QtInteractor(self.frame)
        self.plotter_layout.addWidget(self.plotter.interactor)
        vlayout.addLayout(self.plotter_layout)

        self.frame.setLayout(hlayout)
        self.setCentralWidget(self.frame)

        # simple menu to demo functions
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        exitButton = QtWidgets.QAction('Load', self)
        exitButton.setShortcut('Ctrl+L')
        exitButton.triggered.connect(self.openLoadDialog)
        fileMenu.addAction(exitButton)

        exitButton = QtWidgets.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        # accept url lists (files by drag and drop)
        for url in event.mimeData().urls():
            if str(url.toString()).strip().endswith(".npz"):
                event.accept()
                return
        event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        event.acceptProposedAction()

    def dropEvent(self, event: QtCore.QEvent):
        for url in event.mimeData().urls():
            print(url)
            url = str(url.toString()).strip()
            if url.startswith("file:///"):
                url = url[len("file:///"):]
            if url.startswith("file:"):
                url = url[len("file:"):]
            self.loadFile(url)


    def openLoadDialog(self):
        # opening last directory von sttings
        self._open_dir = self.settings.value("_open_dir")
        if self._open_dir is None:
            self._open_dir = os.getcwd()

        dialog = QtWidgets.QFileDialog()
        dialog.setDirectory(self._open_dir)
        filename = dialog.getOpenFileName(self, "Open Positions", "", "Position Files (*.npz)")
        if isinstance(filename, tuple):
            filename = str(filename[0])
        else:
            filename = str(filename)
        if os.path.exists(filename):
            # noting directory to q settings
            self._open_dir = os.path.split(filename)[0]
            self.settings.setValue("_open_dir", self._open_dir)
            self.settings.sync()
            self.loadFile(filename)

    plot_field = None
    def loadFile(self, filename):
        print("Loading", filename)
        self.M = saenopy.load(filename)

        def scale(m):
            vmin, vmax = np.nanpercentile(m, [1, 99.9])
            return np.clip((m-vmin)/(vmax-vmin), 0, 1)*(vmax-vmin)

        self.point_cloud = pv.PolyData(self.M.R)
        self.point_cloud.point_arrays["f"] = -self.M.f
        self.point_cloud["f_mag"] = np.linalg.norm(self.M.f, axis=1)
        self.point_cloud.point_arrays["U"] = self.M.U
        self.point_cloud["U_mag"] = np.linalg.norm(self.M.U, axis=1)
        self.point_cloud.point_arrays["U_target"] = self.M.U_target
        self.point_cloud["U_target_mag"] = np.linalg.norm(self.M.U_target, axis=1)

        self.point_cloud2 = None

        self.offset = np.min(self.M.R, axis=0)
        if 0:
            self.stats_label.setText(f"""
            Nodes = {self.M.R.shape[0]}
            Tets = {self.M.T.shape[0]}
            """)
        self.replot()

    def calculateStiffness(self):
        self.point_cloud2 = pv.PolyData(np.mean(self.M.R[self.M.T], axis=1))
        from saenopy.materials import SemiAffineFiberMaterial
        self.M.setMaterialModel(SemiAffineFiberMaterial(1645, 0.0008, 0.0075, 0.033), generate_lookup=False)
        self.M._check_relax_ready()
        self.M._prepare_temporary_quantities()
        self.point_cloud2["stiffness"] = self.M.getMaxTetStiffness()/6

    point_cloud = None
    def replot(self):
        names = [name for name, input in self.input_checks.items() if input.value()]
        if len(names) == 0:
            return
        if len(names) <= 3:
            shape = (1, len(names))
        else:
            shape = (2, 2)
        if self.plotter.shape != shape:
            self.plotter_layout.removeWidget(self.plotter)
            self.plotter.close()

            self.plotter = QtInteractor(self.frame, shape=shape, border=False)
            self.plotter.set_background("black")
            self.plotter_layout.addWidget(self.plotter.interactor)

        plotter = self.plotter
        for i, name in enumerate(names):
            plotter.subplot(i//plotter.shape[1], i%plotter.shape[1])
            if name == "stiffness":
                if self.point_cloud2 is None:
                    self.calculateStiffness()
                plotter.add_mesh(self.point_cloud2,  colormap="turbo", point_size=4., render_points_as_spheres=True)
                plotter.update_scalar_bar_range(np.nanpercentile(self.point_cloud2["stiffness"], [50, 99.9]))
            elif name == "f":
                arrows = self.point_cloud.glyph(orient="f", scale="f_mag", factor=3e4)
                plotter.add_mesh(arrows, colormap='turbo', name="arrows")
            else:
                arrows = self.point_cloud.glyph(orient=name, scale=name + "_mag", factor=5)
                plotter.add_mesh(arrows, colormap='turbo', name="arrows")
                plotter.update_scalar_bar_range(np.nanpercentile(self.point_cloud["U_mag"], [1, 99.9]))
            plotter.show_grid()
        print(names)

        plotter.link_views()
        plotter.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    print(sys.argv)
    window = MainWindow()
    if len(sys.argv) >= 2:
        window.loadFile(sys.argv[1])
    window.show()
    sys.exit(app.exec_())
