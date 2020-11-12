import sys

# Setting the Qt bindings for QtPy
import os

os.environ["QT_API"] = "pyqt5"

from qtpy import QtCore, QtWidgets, QtGui

import numpy as np

import pyvista as pv
from pyvistaqt import QtInteractor

import saenopy
import saenopy.multigridHelper
from saenopy.gui import QtShortCuts, QExtendedGraphicsView
import imageio
from qimage2ndarray import array2qimage, rgb_view

""" some magic to prevent PyQt5 from swallowing exceptions """
# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook
# Set the exception hook to our wrapping function
sys.excepthook = lambda *args: sys._excepthook(*args)


class MainWindow(QtWidgets.QMainWindow):
    vector_field = None

    def __init__(self, parent=None, show=True):
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

        self.button_plot1 = QtWidgets.QPushButton("Plot U")
        self.button_plot1.clicked.connect(self.doPlotU)
        layout_vert_plot.addWidget(self.button_plot1)

        self.button_plot1 = QtWidgets.QPushButton("Plot U Target")
        self.button_plot1.clicked.connect(self.doPlotUt)
        layout_vert_plot.addWidget(self.button_plot1)

        self.button_plot1 = QtWidgets.QPushButton("Plot F")
        self.button_plot1.clicked.connect(self.doPlotF)
        layout_vert_plot.addWidget(self.button_plot1)

        # add the pyvista interactor object
        self.plotter = QtInteractor(self.frame)
        vlayout.addWidget(self.plotter.interactor)

        self.view = QExtendedGraphicsView.QExtendedGraphicsView(dropTarget=self)
        self.origin = self.view.origin
        self.pixmap = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap(), self.origin)
        hlayout.addWidget(self.view)

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

        if show:
            self.show()

        self.loadFile(
            r"//131.188.117.96/biophysDS/dboehringer/Platte_4/Software/2-integrate-piv-saenopy/Eval/4-tumor-cell-piv/cell3/testdeformations_win30.npz")

        self.input_max = QtShortCuts.QInputNumber(vlayout, "max", value=1, min=0, max=1, use_slider=True)
        self.input_min = QtShortCuts.QInputNumber(vlayout, "min", value=0, min=0, max=1, use_slider=True)
        self.input_scale = QtShortCuts.QInputNumber(vlayout, "scale", value=1, min=1, max=10000000, use_slider=True)
        self.input_max.valueChanged.connect(self.replot)
        self.input_min.valueChanged.connect(self.replot)
        self.input_scale.valueChanged.connect(self.replot)

        self.z = 0
        self.channel = 0
        self.before_after = 0
        self.displayImage()

    def keyPressEvent(self, event):
        # @key ---- General ----
        if event.key() == QtCore.Qt.Key_Up:
            self.z += 1
            self.displayImage()
        if event.key() == QtCore.Qt.Key_Down:
            self.z -= 1
            self.displayImage()
        if event.key() == QtCore.Qt.Key_PageDown:
            self.channel = 1-self.channel
            self.displayImage()
        if event.key() == QtCore.Qt.Key_PageUp:
            self.channel = 1-self.channel
            self.displayImage()
        if event.key() == QtCore.Qt.Key_Left:
            self.before_after = 1-self.before_after
            self.displayImage()
        if event.key() == QtCore.Qt.Key_Right:
            self.before_after = 1-self.before_after
            self.displayImage()

    def displayImage(self):
        filename = fr"\\131.188.117.96\biophysDS\dboehringer\Platte_4\Measurements_NK_TFM\single-cell-tfm-tom-paper\20170914_A172_rep1\{['Before', 'After'][self.before_after]}\Mark_and_Find_001_Pos001_S001_z{self.z:03d}_ch{self.channel:02d}.tif"
        im = imageio.imread(filename)
        self.pixmap.setPixmap(QtGui.QPixmap(array2qimage(im.astype(np.uint8))))
        self.view.setExtend(im.shape[1], im.shape[0])

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

    def loadFile(self, filename):
        print("Loading", filename)
        self.M = saenopy.load(filename)
        if 0:
            self.stats_label.setText(f"""
            Nodes = {self.M.R.shape[0]}
            Tets = {self.M.T.shape[0]}
            """)
        self.plotter.clear()

    def doPlotU(self):
        self.vector_field = self.M.U
        self.replot()

    def doPlotUt(self):
        self.vector_field = self.M.U_target
        self.replot()

    def doPlotF(self):
        self.vector_field = self.M.f
        self.replot()

    def replot(self):
        self.plotter.clear()
        if self.vector_field is not None:
            mag = np.linalg.norm(self.vector_field, axis=1)
            v1, v2 = np.quantile(mag[~np.isnan(mag)], [self.input_min.value(), self.input_max.value()])
            indices = (v1 < mag) & (mag < v2)
            self.plotter.add_arrows(self.M.R[indices], self.vector_field[indices], self.input_scale.value(), cmap="turbo")
        self.plotter.add_axes()
        self.plotter.add_floor()
        self.plotter.add_bounding_box()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())