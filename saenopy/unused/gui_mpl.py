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
#from openpiv.PIV_3D_plotting import set_axes_equal
from qtpy import QtCore, QtWidgets, QtGui
from qtpy.QtCore import QSettings





def quiver_3D(u, v, w, x=None, y=None, z=None, mask_filtered=None, filter_def=0, filter_reg=(1, 1, 1),
              cmap="jet", quiv_args=None, vmin=None, vmax=None, arrow_scale=0.15, equal_ax=True):
    """ 
    Displaying 3D deformation fields vector arrows

    Parameters
    ----------
     u,v,w: 3d ndarray or lists
         arrays or list with deformation in the direction of axis 0, axis 1 and axis 2 
         (according to the axis sequence of numpy arrays)

     x,y,z: 3d ndarray or lists
          Arrays or list with deformation the coordinates of the deformations.
          Must match the dimensions of the u,v qnd w. If not provided x,y and z are created
          with np.indices(u.shape)

     mask_filtered, boolean 3d ndarray or 1d ndarray
          Array, or list with same dimensions as the deformations. Defines the area where deformations are drawn
          
     filter_def: float
          Filter that prevents the display of deformations arrows with length < filter_def
          
     filter_reg: tuple,list or int
          Filter that prevents the display of every i-th deformations arrows separatly alon each axis.
          filter_reg=(2,2,2) means that only every second arrow along x,y z axis is displayed leading to
          a total reduction of displayed arrows by a factor of 8. filter_reg=3 is interpreted
          as (3,3,3).
          
     cmap: string
          matplotlib colorbar that defines the coloring of the arrow
          
     quiv_args: dict
         Dictionary with kwargs passed on to the matplotlib quiver function.

     vmin,vmax: float
         Upper and lower bounds for the colormap. Works like vmin and vmax in plt.imshow().
         
    arrow_scale: float
        Automatic scaling of the quiver arrows so that the longest arrow has the 
        length axis length * arrow_scale. Arrow length can alternatively be set by
        passing a "lenght" argument in quiv_args. 
    
    equal_axes: bool
        resize the figure axis so that they are have equal scaling.
    

    Returns
    -------
     fig: matploltib figure object

     ax: mattplotlib axes object
         the holding the main 3D quiver plot

    """

    # default arguments for the quiver plot. can be overwritten by quiv_args
    quiver_args = {"normalize":False, "alpha":0.8, "pivot":'tail', "linewidth":1, "length":1}
    if isinstance(quiv_args, dict):
        quiver_args.update(quiv_args)
    # overwriting length if an arrow scale and a "length" argument in quiv_args 
    # is provided at the same
    if arrow_scale is not None:
        quiver_args["length"] = 1
        
    # convert filter ot list if proveided as int    
    if not isinstance(filter_reg, (tuple, list)):
        filter_reg = [filter_reg] * 3

    # generating coordinates if not provided
    if x is None:
        # if you provide deformations as a list
        if len(u.shape) == 1:
            x, y, z = [np.indices(u.shape)[0] for i in range(3)]
        # if you provide deformations as an array
        elif len(u.shape) == 3:
            x, y, z = np.indices(u.shape)
        else:
            raise ValueError(
                "displacement data has wrong number of dimensions (%s). Use 1d array, list, or 3d array." % str(
                    len(u.shape)))

    # conversion to array
    x, y, z = np.array([x, y, z])

    deformation = np.sqrt(u ** 2 + v ** 2 + w ** 2)
    if not isinstance(mask_filtered, np.ndarray):
        mask_filtered = deformation > filter_def
        if isinstance(filter_reg, list):
            show_only = np.zeros(u.shape).astype(bool)
            # filtering out every x-th
            show_only[::filter_reg[0], ::filter_reg[1], ::filter_reg[2]] = True
            mask_filtered = np.logical_and(mask_filtered, show_only)

    xf = x[mask_filtered]
    yf = y[mask_filtered]
    zf = z[mask_filtered]
    uf = u[mask_filtered]
    vf = v[mask_filtered]
    wf = w[mask_filtered]
    df = deformation[mask_filtered]

    # make cmap
    # create normalized color map for arrows
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)  # 10 ) #cbound[1] ) #)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # different option 
    colors = matplotlib.cm.get_cmap(cmap)(norm(df))  
    colors = [c for c, d in zip(colors, df) if d > 0] + list(chain(*[[c, c] for c, d in zip(colors, df) if d > 0]))
    # colors in ax.quiver 3d is really fucked up/ will probably change with updates:
    # requires list with: first len(u) entries define the colors of the shaft, then the next len(u)*2 entries define
    # the color ofleft and right arrow head side in alternating order. Try for example:
    # colors = ["red" for i in range(len(cf))] + list(chain(*[["blue", "yellow"] for i in range(len(cf))]))
    # to see this effect.
    # BUT WAIT THERS MORE: zeor length arrows are apparently filtered out in the matplolib with out filtering 
    # the color list appropriately so we have to do this our selfs as well

    # scale arrows to axis dimensions:
    ax_dims = [(x.min(), x.max()), (y.min(), y.max()), (z.min(), z.max())]
    if arrow_scale is not None:
        max_length = df.max()
        max_dim_length= np.max([(d[1] - d[0] + 1) for d in ax_dims] )
        scale = max_dim_length * arrow_scale / max_length
    else:
        scale = 1
        
     # plotting
    fig = plt.gcf()
    #plt.clf()
    ax = fig.gca(projection='3d')#, rasterized=True)

    try:
        ax.q.remove()
    except AttributeError:
        pass
    
    q = ax.quiver(xf, yf, zf, uf*scale, vf*scale, wf*scale, colors=colors, **quiver_args)
    
    ax.q = q
     
    #plt.colorbar(sm)
    ax.set_xlim(ax_dims[0])
    ax.set_ylim(ax_dims[1])
    ax.set_zlim(ax_dims[2])

    #if equal_ax:
    #    set_axes_equal(ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.w_xaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
    ax.w_yaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
    ax.w_zaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
    
    return fig



""" some magic to prevent PyQt5 from swallowing exceptions """
# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook
# Set the exception hook to our wrapping function
sys.excepthook = lambda *args: sys._excepthook(*args)


class MatplotlibWidget(Canvas):

    def __init__(self, parent=None, width=4, height=3, dpi=100):
        plt.ioff()
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
        


class Commander(QtWidgets.QWidget):
    M = None

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # QSettings
        self.settings = QSettings("Saenopy", "Seanopy")
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
            self.settings.setValue("_open_dir",self._open_dir)
            self.settings.sync()
            self.loadFile(filename)

    def loadFile(self, filename):
        self.M = load(filename)
        self.stats_label.setText(f"""
        Nodes = {self.M.R.shape[0]}
        Tets = {self.M.T.shape[0]}
        """)

    def doPlotU(self):       
        if self.M is None:
            print("No deformation Field found")
            return
        U = self.M.U  # U_target - M.U
        factor = 1e6
        R = self.M.R
        print(U.min(), U.max(), U.mean())
        lengths = np.linalg.norm(U, axis=1)
        print(U.shape, R.shape)
        fig = quiver_3D(U[:, 0] * factor, U[:, 1] * factor, U[:, 2] * factor,
                  R[:, 0] * factor, R[:, 1] * factor, R[:, 2] * factor, filter_reg=(1,1,1), filter_def=np.nanpercentile(lengths, 0) * factor,
                  arrow_scale = 0.05,quiv_args={"alpha":0.8, "pivot":'tail', "linewidth":0.5, "length":1})
        ax = plt.gca()
        ax.set_xlim(self.M.R[:, 0].min() * 1e6, self.M.R[:, 0].max() * 1e6)
        ax.set_ylim(self.M.R[:, 1].min() * 1e6, self.M.R[:, 1].max() * 1e6)
        ax.set_zlim(self.M.R[:, 2].min() * 1e6, self.M.R[:, 2].max() * 1e6)
        #set_axes_equal(ax)
        self.canvas.draw()
        
        

    def doPlotUt(self):
        if self.M is None:
            print("No deformation Field found")
            return
        U = self.M.U_target  # U_target - M.U
        factor = 1e6
        R = self.M.R
        print(U.min(), U.max(), U.mean())
        lengths = np.linalg.norm(U, axis=1)
        print(U.shape, R.shape)
        fig = quiver_3D(U[:, 0] * factor, U[:, 1] * factor, U[:, 2] * factor,
                  R[:, 0] * factor, R[:, 1] * factor, R[:, 2] * factor, filter_reg=(1,1,1), filter_def=np.nanpercentile(lengths, 0) * factor,
                  arrow_scale = 0.05,quiv_args={"alpha":0.8, "pivot":'tail', "linewidth":0.5, "length":1})
        ax = plt.gca()
        ax.set_xlim(self.M.R[:, 0].min() * 1e6, self.M.R[:, 0].max() * 1e6)
        ax.set_ylim(self.M.R[:, 1].min() * 1e6, self.M.R[:, 1].max() * 1e6)
        ax.set_zlim(self.M.R[:, 2].min() * 1e6, self.M.R[:, 2].max() * 1e6)
        #set_axes_equal(ax)
        self.canvas.draw()


    def doPlotF(self):
        if self.M is None:
            print("No deformation Field found")
            return
        U = self.M.f  # U_target - M.U
        factor = 1e6
        R = self.M.R
        print(U.min(), U.max(), U.mean())
        lengths = np.linalg.norm(U, axis=1)
        print(U.shape, R.shape)
        fig = quiver_3D(U[:, 0] * factor, U[:, 1] * factor, U[:, 2] * factor,
                  R[:, 0] * factor, R[:, 1] * factor, R[:, 2] * factor, filter_reg=(1,1,1),
                  vmin=np.nanpercentile(lengths, 90) * factor, vmax=np.nanpercentile(lengths, 99) * factor,
                  cmap = "autumn_r", arrow_scale = 0.15,quiv_args={"alpha":0.8, "pivot":'tail', "linewidth":0.5, "length":1})
        ax = plt.gca()
        ax.set_xlim(self.M.R[:, 0].min() * 1e6, self.M.R[:, 0].max() * 1e6)
        ax.set_ylim(self.M.R[:, 1].min() * 1e6, self.M.R[:, 1].max() * 1e6)
        ax.set_zlim(self.M.R[:, 2].min() * 1e6, self.M.R[:, 2].max() * 1e6)
        #set_axes_equal(ax)
        self.canvas.draw()
        
        



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    window = Commander()
    window.show()
    app.exec_()
