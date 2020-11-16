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
    im = None

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
        self.input_height = QtShortCuts.QInputNumber(hlayout, "", value=1, min=1, max=500, use_slider=True, float=False)
        self.input_height.slider.setOrientation(QtCore.Qt.Vertical)
        self.input_height.spin_box.setVisible(False)
        self.input_height.valueChanged.connect(self.displayImage)

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

        #self.loadFile(
        #    r"//131.188.117.96/biophysDS/dboehringer/Platte_4/Software/2-integrate-piv-saenopy/Eval/4-tumor-cell-piv/cell3/testdeformations_win30.npz")

        self.input_max = QtShortCuts.QInputNumber(vlayout, "max", value=1, min=0, max=1, use_slider=True)
        self.input_min = QtShortCuts.QInputNumber(vlayout, "min", value=0, min=0, max=1, use_slider=True)
        self.input_scale = QtShortCuts.QInputNumber(vlayout, "scale", value=1, min=1, max=10000000, use_slider=True)
        self.input_max.valueChanged.connect(self.replot)
        self.input_min.valueChanged.connect(self.replot)
        self.input_scale.valueChanged.connect(self.replot)
        self.input_xy_switch = QtShortCuts.QInputBool(vlayout, "swap xy", value=False)
        self.input_xy_switch.valueChanged.connect(self.replot)
        self.input_xy_switch.valueChanged.connect(self.displayImage)
        self.input_x_invert = QtShortCuts.QInputBool(vlayout, "x invert", value=False)
        self.input_x_invert.valueChanged.connect(self.replot)
        self.input_x_invert.valueChanged.connect(self.displayImage)
        self.input_y_invert = QtShortCuts.QInputBool(vlayout, "y invert", value=False)
        self.input_y_invert.valueChanged.connect(self.replot)
        self.input_y_invert.valueChanged.connect(self.displayImage)

        self.lines = []

        self.z = 0
        self.channel = 0
        self.before_after = 0

        self.pixel_size = np.zeros(3)
        self.pixel_size[:2] = 0.7222 * 1e-6
        self.pixel_size[2] = 0.741 * 1e-6

        self.offset = np.zeros(3)

        #self.displayImage()

    def pixel_to_mu(self, x):
        return np.asarray(x)*self.pixel_size + self.offset

    def mu_to_pixel(self, x, vector=False):
        if vector is True:
            return np.asarray(x) / self.pixel_size
        return (np.asarray(x)-self.offset) / self.pixel_size

    def addVectorLines(self, R, V):
        mags = np.linalg.norm(V, axis=1)
        range_min, range_max = np.nanmin(mags), np.nanmax(mags)
        cmap = plt.get_cmap("turbo")
        R = self.mu_to_pixel(R)
        V = self.mu_to_pixel(V, vector=True)

        i = 0
        for r, v, m in zip(R, V, mags):
            if np.isnan(m):
                continue

            if i < len(self.lines):
                line = self.lines[i]
            else:
                line = QtWidgets.QGraphicsLineItem(self.origin)
                self.lines.append(line)
            line.setLine(r[0], r[1], r[0]+v[0], r[1]+v[1])
            color = (np.array(cmap(int((m-range_min)/(range_max-range_min)*255)))*255).astype(np.uint8)
            line.setPen(QtGui.QPen(QtGui.QColor(*color)))
            i += 1
        new_count = i
        if new_count < len(self.lines):
            for i in range(new_count, len(self.lines)):
                self.lines[i].scene().removeItem(self.lines[i])
            self.lines = self.lines[:new_count]


    def keyPressEvent(self, event):
        # @key ---- General ----
        if event.key() == QtCore.Qt.Key_Up:
            self.input_height.setValue(self.input_height.value()+1)
            self.displayImage()
        if event.key() == QtCore.Qt.Key_Down:
            self.input_height.setValue(self.input_height.value()-1)
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
        self.z = self.input_height.value()
        filename = fr"\\131.188.117.96\biophysDS\dboehringer\Platte_4\Measurements_NK_TFM\single-cell-tfm-tom-paper\20170914_A172_rep1\{['Before', 'After'][self.before_after]}\Mark_and_Find_001_Pos010_S001_z{self.z:03d}_ch{self.channel:02d}.tif"
        im = imageio.imread(filename)
        self.im = im
        self.pixmap.setPixmap(QtGui.QPixmap(array2qimage(im.astype(np.uint8))))
        self.view.setExtend(im.shape[1], im.shape[0])
        self.replot_plane()

        if self.vector_field is not None:
            x, v = self.getX_Vector()
            diff = np.abs(x[:, 2] - self.pixel_to_mu([0, 0, self.z])[2])
            i = diff < diff.min()*1.1
            self.addVectorLines(x[i], v[i])

    def getX_Vector(self):
        if self.vector_field is not None:
            xyz = np.array([0, 1, 2], dtype=np.uint8)
            if self.input_xy_switch.value() is True:
                xyz = np.array([1, 0, 2], dtype=np.uint8)
            offset = np.array([0, 0, 0], dtype=np.float)
            factor = np.array([1, 1, 1])
            max = np.max(self.M.R, axis=0)
            if self.input_x_invert.value() is True:
                offset[0] = max[0]
                factor[0] = -1
            if self.input_y_invert.value() is True:
                offset[1] = max[1]
                factor[1] = -1
            return self.M.R[:, xyz]*factor + offset, self.vector_field[:, xyz]*factor

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
        self.offset = np.min(self.M.R, axis=0)
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
            x, v = self.getX_Vector()
            mag = np.linalg.norm(self.vector_field, axis=1)
            v1, v2 = np.quantile(mag[~np.isnan(mag)], [self.input_min.value(), self.input_max.value()])
            indices = (v1 < mag) & (mag < v2)
            self.add_arrows(x[indices], v[indices], self.input_scale.value(), cmap="turbo")
        self.plotter.add_axes()
        #self.plotter.add_floor()

        self.plotter.add_bounding_box()
        self.replot_plane()

    plane_actor = None
    def replot_plane(self):
        if self.im is not None:
            x = np.array([0, self.im.shape[1]])
            y = np.array([0, self.im.shape[0]])
            z = self.z
            x, y, z = self.pixel_to_mu(np.array([x, y, z]).T).T

            x, y = np.meshgrid(x, y)
            z = np.ones_like(x)*z
            mesh = pv.StructuredGrid(x, y, z)

            mesh.texture_map_to_plane(inplace=True)

            im = pv.numpy_to_texture(self.im[::-1, ::])
            if self.plane_actor is not None:
                self.plotter.remove_actor(self.plane_actor)
            self.plane_actor = self.plotter.add_mesh(mesh, show_edges=False, color='white', texture=im)

    def add_arrows(self, cent, direction, mag=1, **kwargs):
        """Add arrows to the plotter.

        Parameters
        ----------
        cent : np.ndarray
            Array of centers.

        direction : np.ndarray
            Array of direction vectors.

        mag : float, optional
            Amount to scale the direction vectors.

        Examples
        --------
        Plot a random field of vectors and save a screenshot of it.

        >>> import numpy as np
        >>> import pyvista
        >>> cent = np.random.random((10, 3))
        >>> direction = np.random.random((10, 3))
        >>> plotter = pyvista.Plotter()
        >>> _ = plotter.add_arrows(cent, direction)
        >>> plotter.show()  # doctest:+SKIP

        """
        if cent.shape != direction.shape:  # pragma: no cover
            raise ValueError('center and direction arrays must have the same shape')

        direction = direction.copy()
        if cent.ndim != 2:
            cent = cent.reshape((-1, 3))

        if direction.ndim != 2:
            direction = direction.reshape((-1, 3))

        pdata = pv.vector_poly_data(cent, direction)
        # Create arrow object
        arrow = vtk.vtkArrowSource()
        arrow.Update()
        glyph3D = vtk.vtkGlyph3D()
        glyph3D.SetSourceData(arrow.GetOutput())
        glyph3D.SetInputData(pdata)
        glyph3D.SetVectorModeToUseVector()
        glyph3D.SetScaling(True)
        glyph3D.SetScaleFactor(mag)
        glyph3D.Update()

        arrows = pv.utilities.wrap(glyph3D.GetOutput())

        return self.plotter.add_mesh(arrows, **kwargs)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
