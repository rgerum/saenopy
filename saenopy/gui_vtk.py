import sys

# Setting the Qt bindings for QtPy
import os

os.environ["QT_API"] = "pyqt5"

from qtpy import QtCore, QtWidgets, QtGui
from pathlib import Path
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


def pathParts(path):
    if path.parent == path:
        return [path]
    return pathParts(path.parent) + [path]


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
        #self.setCentralWidget(self.frame)
        hlayout = QtWidgets.QVBoxLayout(self.frame)
        self.frame.setLayout(hlayout)

        """ browser"""
        self.fileBrowserWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.fileBrowserWidget)

        self.dirmodel = QtWidgets.QFileSystemModel()
        # Don't show files, just folders
        # self.dirmodel.setFilter(QtCore.QDir.NoDotAndDotDot | QtCore.QDir.AllDirs)
        self.dirmodel.setNameFilters(["*.npz"])
        self.dirmodel.setNameFilterDisables(False)
        self.folder_view = QtWidgets.QTreeView(parent=self)
        self.folder_view.setModel(self.dirmodel)
        self.folder_view.activated[QtCore.QModelIndex].connect(self.clicked)
        #self.folder_view.selected[QtCore.QModelIndex].connect(self.clicked)

        # Don't show columns for size, file type, and last modified
        self.folder_view.setHeaderHidden(True)
        self.folder_view.hideColumn(1)
        self.folder_view.hideColumn(2)
        self.folder_view.hideColumn(3)

        self.selectionModel = self.folder_view.selectionModel()

        splitter_filebrowser = QtWidgets.QSplitter()
        splitter_filebrowser.addWidget(self.folder_view)
        splitter_filebrowser.addWidget(self.frame)
        splitter_filebrowser.setStretchFactor(0, 2)
        splitter_filebrowser.setStretchFactor(1, 4)

        hbox = QtWidgets.QHBoxLayout(self.fileBrowserWidget)
        hbox.addWidget(splitter_filebrowser)
        self.set_path(__file__)
        """"""
        #return

        vlayout = QtWidgets.QVBoxLayout()
        hlayout.addLayout(vlayout)

        layout_vert_plot = QtWidgets.QHBoxLayout()
        vlayout.addLayout(layout_vert_plot)

        self.input_checks = {}
        for name in ["U_target", "U", "f", "stiffness"]:
            input = QtShortCuts.QInputBool(layout_vert_plot, name, name != "stiffness")
            input.valueChanged.connect(self.replot)
            self.input_checks[name] = input
        layout_vert_plot.addStretch()
        self.button_export = QtWidgets.QPushButton("save image")
        layout_vert_plot.addWidget(self.button_export)
        self.button_export.clicked.connect(self.saveScreenshot)

        # add the pyvista interactor object
        self.plotter_layout = QtWidgets.QHBoxLayout()
        self.plotter = QtInteractor(self.frame)
        self.plotter_layout.addWidget(self.plotter.interactor)
        vlayout.addLayout(self.plotter_layout)

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
        print("show")

    def set_path(self, path):
        path = Path(path)
        self.dirmodel.setRootPath(str(path.parent))
        for p in pathParts(path):
            self.folder_view.expand(self.dirmodel.index(str(p)))
        self.folder_view.setCurrentIndex(self.dirmodel.index(str(path)))
        print("scroll to ", str(path), self.dirmodel.index(str(path)))
        self.folder_view.scrollTo(self.dirmodel.index(str(path)))

    def clicked(self, index):
        # get selected path of folder_view
        index = self.selectionModel.currentIndex()
        dir_path = self.dirmodel.filePath(index)

        if dir_path.endswith(".npz"):
            print("################# load", dir_path)
            self.loadFile(dir_path)

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
        self.set_path(Path(filename))
        self.M = saenopy.load(filename)

        def scale(m):
            vmin, vmax = np.nanpercentile(m, [1, 99.9])
            return np.clip((m-vmin)/(vmax-vmin), 0, 1)*(vmax-vmin)

        R = self.M.R
        minR = np.min(R, axis=0)
        maxR = np.max(R, axis=0)
        
        if self.M.reg_mask is None: 
            border = (R[:, 0] < minR[0] + 0.5e-6) | (R[:, 0] > maxR[0] - 0.5e-6) | \
                     (R[:, 1] < minR[1] + 0.5e-6) | (R[:, 1] > maxR[1] - 0.5e-6) | \
                     (R[:, 2] < minR[2] + 0.5e-6) | (R[:, 2] > maxR[2] - 0.5e-6)
            self.M.reg_mask = ~border 
                 

        self.point_cloud = pv.PolyData(self.M.R)
        self.point_cloud.point_arrays["f"] = -self.M.f*self.M.reg_mask[:, None]
        self.point_cloud["f_mag"] = np.linalg.norm(self.M.f*self.M.reg_mask[:, None], axis=1)
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
        #self.M.setMaterialModel(SemiAffineFiberMaterial(1645, 0.0008, 0.0075, 0.033), generate_lookup=False)
        if self.M.material_parameters is not None:
            print("loading material")
            self.M.setMaterialModel(SemiAffineFiberMaterial(*self.M.material_parameters[1:]), generate_lookup=False)
        else:
            print("Warning using default material parameters")
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
            #pv.set_plot_theme("document")
            self.plotter_layout.addWidget(self.plotter.interactor)

        plotter = self.plotter
        # color bar design properties
        # Set a custom position and size
        sargs = dict(position_x=0.05, position_y=0.95,
                    title_font_size=15,
                    label_font_size=9,
                    n_labels=3,
                    italic=True,  ##height=0.25, #vertical=True,
                    fmt="%.1e",  
                    font_family="arial")
   
    
        for i, name in enumerate(names):
            plotter.subplot(i//plotter.shape[1], i%plotter.shape[1])
            # scale plot with axis length later
            norm_stack_size = np.abs(np.max(self.M.R)-np.min(self.M.R))
            

            if name == "stiffness":
                if self.point_cloud2 is None:
                    self.calculateStiffness()
                #clim =  np.nanpercentile(self.point_cloud2["stiffness"], [50, 99.9])   
                plotter.add_mesh(self.point_cloud2,  colormap="turbo", point_size=4., render_points_as_spheres=True,
                                 scalar_bar_args=sargs, stitle = "Stiffness [Pa]" )
                plotter.update_scalar_bar_range( np.nanpercentile(self.point_cloud2["stiffness"], [50, 99.9]))
            elif name == "f":
                arrows = self.point_cloud.glyph(orient="f", scale="f_mag", \
                                                # Automatically scale maximal force to 15% of axis length
                                                factor=0.15 * norm_stack_size / np.nanmax(
                                                    np.linalg.norm(self.M.f*self.M.reg_mask[:,None], axis=1)))
                plotter.add_mesh(arrows, colormap='turbo', name="arrows", 
                                scalar_bar_args=sargs, stitle = "Force [N]",
                               )
                plotter.update_scalar_bar_range(np.nanpercentile(self.point_cloud["f_mag"], [50, 99.9]))
                # plot center points if desired
                # plotter.add_points(np.array([self.M.getCenter(mode="Force")]), color='r')   
                
            elif name == "U_target":
                arrows2 = self.point_cloud.glyph(orient="U_target", scale="U_target_mag", \
                                                 # Automatically scale maximal force to 10% of axis length
                                                 factor=0.1 * norm_stack_size / np.nanmax(
                                                     np.linalg.norm(self.M.U_target, axis=1)))
                plotter.add_mesh(arrows2, colormap='turbo', name="arrows2",stitle = "Deformations [m]",                            
                                 scalar_bar_args=sargs)#
               
                # plot center if desired
                #plotter.add_points(np.array([self.M.getCenter(mode="deformation_target")]), color='w')
                
                plotter.update_scalar_bar_range(np.nanpercentile(self.point_cloud["U_target_mag"], [50, 99.9]))
                #plotter.update_scalar_bar_range([0,1.5e-6])
            elif name == "U":
                  arrows3 = self.point_cloud.glyph(orient=name, scale=name + "_mag", \
                                                     # Automatically scale maximal force to 10% of axis length
                                                     factor=0.1 * norm_stack_size / np.nanmax(
                                                         np.linalg.norm(self.M.U, axis=1)))
                  plotter.add_mesh(arrows3, colormap='turbo', name="arrows3", stitle = "Rec. Deformations [m]",
                                     scalar_bar_args=sargs)
                  plotter.update_scalar_bar_range(np.nanpercentile(self.point_cloud["U_mag"], [50, 99.9]))
                  # plotter.update_scalar_bar_range([0,1.5e-6])
                    
                # plot center points if desired
                # plotter.add_points(np.array([self.M.getCenter(mode="Deformation")]), color='w')
                # plotter.add_points(np.array([self.M.getCenter(mode="Force")]), color='r')

            plotter.show_grid()
        #print(names)
        plotter.link_views()
        plotter.show()

    def saveScreenshot(self):
        new_path = QtWidgets.QFileDialog.getSaveFileName(self, "Save Image", os.getcwd(), "Image Files (*.jpg, *.png)")
        # if we got one, set it
        if new_path:
            if isinstance(new_path, tuple):
                new_path = new_path[0]
            else:
                new_path = str(new_path)
            imageio.imsave(new_path, self.plotter.image)
            print("saved", new_path)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    print(sys.argv)
    window = MainWindow()
    if len(sys.argv) >= 2:
        window.loadFile(sys.argv[1])
    window.show()
    sys.exit(app.exec_())
