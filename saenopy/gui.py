import os, sys
import re
import numpy as np
import matplotlib.pyplot as plt
from qtpy import QtWidgets, QtCore
from qtpy import API_NAME as QT_API_NAME
if QT_API_NAME.startswith("PyQt4"):
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as Canvas
    from matplotlib.backends.backend_qt4agg import FigureManager
    from matplotlib.backends.backend_qt4 import NavigationToolbar2QT as NavigationToolbar
else:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
    from matplotlib.backends.backend_qt5agg import FigureManager
    from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib
from itertools import chain
from saenopy import load
import time
from matplotlib import _pylab_helpers

from qtpy import QtCore, QtWidgets, QtGui

""" some magic to prevent PyQt5 from swallowing exceptions """
# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook
# Set the exception hook to our wrapping function
sys.excepthook = lambda *args: sys._excepthook(*args)


class MatplotlibWidget(Canvas):

    def __init__(self, parent=None, width=4, height=3, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.figure.patch.set_facecolor([0, 1, 0, 0])
        self.axes = self.figure.add_subplot(111)

        Canvas.__init__(self, self.figure)
        self.setParent(parent)

        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)

        self.manager = FigureManager(self, 1)
        self.manager._cidgcf = self.figure

        """
        _pylab_helpers.Gcf.figs[num] = canvas.manager
        # get the canvas of the figure
        manager = _pylab_helpers.Gcf.figs[num]
        # set the size if it is defined
        if figsize is not None:
            _pylab_helpers.Gcf.figs[num].window.setGeometry(100, 100, figsize[0] * 80, figsize[1] * 80)
        # set the figure as the active figure
        _pylab_helpers.Gcf.set_active(manager)
        """
        _pylab_helpers.Gcf.set_active(self.manager)


def quiver_3D(u, v, w, x=None, y=None, z=None, image_dim=None, mask_filtered=None, filter_def=0, filter_reg=[1], cmap="jet", quiv_args={}, cbound=None):
    #filter_def filters values with smaler absolute deformation
    # nans are also removed
    # setting the filter to <0 will probably mess up the arrow colors
    # filter_reg filters every n-th value, separate for x, y, z axis
    # you can also provide your own mask with mask_filtered !!! make sure to filter out arrows with zero total deformation!!!!
    # other wise the arrows are not colored correctly
    # use indices for x,y,z axis as default - can be specified by x,y,z

    # default arguments for the quiver plot. can be overwritten by quiv_args
    quiver_args = {"normalize":False, "alpha":0.8, "pivot":'tail', "linewidth":1, "length":20}
    quiver_args.update(quiv_args)

    u = np.array(u)
    v = np.array(v)
    w = np.array(w)

    if not isinstance(image_dim, (list, tuple, np.ndarray)):
        image_dim = np.array(u.shape)

    # generating coordinates if not provided
    if x is None:
        # if you provide deformations as a list
        if len(u.shape) == 1:
            x, y, z = [np.indices(u.shape)[0] for i in range(3)]
        # if you provide deformations as an array
        elif len(u.shape) == 3:
            x, y, z = np.indices(u.shape)
        else:
            raise ValueError("displacement data has wrong number of dimensions (%s). Use 1d array or list, or 3d array."%str(len(u.shape)))

    # multiplying coordinates with "image_dim" factor if coordinates are provided
    #x, y, z = np.array([x, y, z]) * np.expand_dims(np.array(image_dim) / np.array(u.shape), axis=list(range(1, len(u.shape)+1)))

    deformation = np.sqrt(u ** 2 + v ** 2 + w ** 2)
    if not isinstance(mask_filtered, np.ndarray):
        mask_filtered = deformation > filter_def
        if isinstance(filter_reg, list):
            show_only = np.zeros(u.shape).astype(bool)
            if len(filter_reg) == 1:
                show_only[::filter_reg[0]] = True
            elif len(filter_reg) == 3:
                show_only[::filter_reg[0], ::filter_reg[1], ::filter_reg[2]] = True
            else:
                raise ValueError(
                    "filter_reg data has wrong length (%s). Use list with length 1 or 3." % str(len(filter_reg.shape)))
            mask_filtered = np.logical_and(mask_filtered, show_only)

    xf = x[mask_filtered]
    yf = y[mask_filtered]
    zf = z[mask_filtered]
    uf = u[mask_filtered]
    vf = v[mask_filtered]
    wf = w[mask_filtered]
    df = deformation[mask_filtered]

    # make cmap
    if not cbound:
        cbound = [0,np.nanmax(df)]
    # create normalized color map for arrows
    norm = matplotlib.colors.Normalize(vmin=cbound[0], vmax=cbound[1])  # 10 ) #cbound[1] ) #)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # different option
    colors = matplotlib.cm.jet(norm(df))  #

    colors = [c for c, d in zip(colors, df) if d > 0] + list(chain(*[[c, c] for c, d in zip(colors, df) if d > 0]))
    # colors in ax.quiver 3d is really fucked up/ will probably change with updates:
    # requires list with: first len(u) entries define the colors of the shaft, then the next len(u)*2 entries define
    # the color ofleft and right arrow head side in alternating order. Try for example:
    # colors = ["red" for i in range(len(cf))] + list(chain(*[["blue", "yellow"] for i in range(len(cf))]))
    # to see this effect
    # BUT WAIT THERS MORE: zeor length arrows are apparently filtered out in the matplolib with out filtering the color list appropriately
    # so we have to do this our selfs as well

    # plotting
    fig = plt.gcf()
    #plt.clf()
    ax = fig.gca(projection='3d')#, rasterized=True)

    try:
        ax.q.remove()
    except AttributeError:
        pass
    q = ax.quiver(xf, yf, zf, uf, vf, wf, colors=colors, **quiver_args)
    ax.q = q
    #
    #plt.colorbar(sm)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.w_xaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
    ax.w_yaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
    ax.w_zaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
    return fig


class Commander(QtWidgets.QWidget):
    M = None

    def __init__(self, parent=None):
        super().__init__(parent)

        # widget layout and elements
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        self.setWindowTitle("Saenopy")
        layout_main = QtWidgets.QHBoxLayout(self)

        layout_vert = QtWidgets.QVBoxLayout()
        layout_main.addLayout(layout_vert)

        self.button_load = QtWidgets.QPushButton("load")
        self.button_load.clicked.connect(self.openLoadDialog)
        layout_vert.addWidget(self.button_load)

        self.stats_label = QtWidgets.QLabel()
        layout_vert.addWidget(self.stats_label)

        layout_vert_plot = QtWidgets.QVBoxLayout()
        layout_main.addLayout(layout_vert_plot)

        self.button_plot1 = QtWidgets.QPushButton("Plot U")
        self.button_plot1.clicked.connect(self.doPlotU)
        layout_vert_plot.addWidget(self.button_plot1)

        self.button_plot1 = QtWidgets.QPushButton("Plot U Target")
        self.button_plot1.clicked.connect(self.doPlotUt)
        layout_vert_plot.addWidget(self.button_plot1)

        self.button_plot1 = QtWidgets.QPushButton("Plot F")
        self.button_plot1.clicked.connect(self.doPlotF)
        layout_vert_plot.addWidget(self.button_plot1)

        self.canvas = MatplotlibWidget(self)
        layout_vert_plot.addWidget(self.canvas)

        self.canvas.figure.add_axes(projection="3d")

        #self.canvas.axes.plot([1,2,3],[1,2,3])
        #plt.plot([2,1],[1,2])
        #plt.subplot(121)
        #plt.plot([2, 1], [1, 2])

        #self.loadFile(r"..\output.npz")

    def openLoadDialog(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, "Open Positions", "", "Position Files (*.npz)")
        if isinstance(filename, tuple):
            filename = str(filename[0])
        else:
            filename = str(filename)
        if os.path.exists(filename):
            self.loadFile(filename)

    def loadFile(self, filename):
        self.M = load(filename)
        self.stats_label.setText(f"""
        Nodes = {self.M.R.shape[0]}
        Tets = {self.M.T.shape[0]}
        """)

    def doPlotU(self):
        U = self.M.U  # U_target - M.U
        factor = 1e6
        R = self.M.R
        print(U.min(), U.max(), U.mean())
        lengths = np.linalg.norm(U, axis=1)
        indices = lengths > np.quantile(lengths, 0.95)
        U = U[indices]
        R = R[indices]
        quiver_3D(U[:, 0] * factor, U[:, 1] * factor, U[:, 2] * factor,
                  R[:, 0] * 1e6, R[:, 1] * 1e6, R[:, 2] * 1e6, filter_def=0, filter_reg=[1],
                  quiv_args={"linewidth": 0.3, "alpha": 0.8, "normalize": True, "length":5})
        ax = plt.gca()
        ax.set_xlim(self.M.R[:, 0].min() * 1e6, self.M.R[:, 0].max() * 1e6)
        ax.set_ylim(self.M.R[:, 1].min() * 1e6, self.M.R[:, 1].max() * 1e6)
        ax.set_zlim(self.M.R[:, 2].min() * 1e6, self.M.R[:, 2].max() * 1e6)
        self.canvas.draw()

    def doPlotUt(self):
        U = self.M.U_target  # U_target - M.U
        factor = 1e6
        R = self.M.R
        print(U.min(), U.max(), U.mean())
        lengths = np.linalg.norm(U, axis=1)
        indices = lengths > np.quantile(lengths, 0.95)
        U = U[indices]
        R = R[indices]
        quiver_3D(U[:, 0] * factor, U[:, 1] * factor, U[:, 2] * factor,
                  R[:, 0] * 1e6, R[:, 1] * 1e6, R[:, 2] * 1e6, filter_def=0, filter_reg=[1],
                  quiv_args={"linewidth": 0.3, "alpha": 0.8, "normalize": True, "length":5})
        ax = plt.gca()
        ax.set_xlim(self.M.R[:, 0].min() * 1e6, self.M.R[:, 0].max() * 1e6)
        ax.set_ylim(self.M.R[:, 1].min() * 1e6, self.M.R[:, 1].max() * 1e6)
        ax.set_zlim(self.M.R[:, 2].min() * 1e6, self.M.R[:, 2].max() * 1e6)
        self.canvas.draw()


    def doPlotF(self):
        U = self.M.f  # U_target - M.U
        factor = 1e6
        R = self.M.R
        lengths = np.linalg.norm(U, axis=1)
        indices = lengths > np.quantile(lengths, 0.95)
        U = U[indices]
        R = R[indices]
        quiver_3D(U[:, 0] * factor, U[:, 1] * factor, U[:, 2] * factor,
                  R[:, 0] * 1e6, R[:, 1] * 1e6, R[:, 2] * 1e6, filter_def=0, filter_reg=[1],
                  quiv_args={"linewidth": 0.3, "alpha": 0.8, "normalize": True, "length":5})
        ax = plt.gca()
        ax.set_xlim(self.M.R[:, 0].min() * 1e6, self.M.R[:, 0].max() * 1e6)
        ax.set_ylim(self.M.R[:, 1].min() * 1e6, self.M.R[:, 1].max() * 1e6)
        ax.set_zlim(self.M.R[:, 2].min() * 1e6, self.M.R[:, 2].max() * 1e6)
        self.canvas.draw()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    window = Commander()
    window.show()
    app.exec_()
