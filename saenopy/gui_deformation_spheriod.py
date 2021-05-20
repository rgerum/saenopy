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
import saenopy.getDeformations
import saenopy.multigridHelper
import saenopy.materials

if QtCore.qVersion() >= "5.":
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
import jointforces as jf
import urllib
from pathlib import Path

""" some magic to prevent PyQt5 from swallowing exceptions """
# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook
# Set the exception hook to our wrapping function
sys.excepthook = lambda *args: sys._excepthook(*args)

class QHLine(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


class LookUpTable(QtWidgets.QWidget):
    progress_signal = QtCore.Signal(int, int)

    def __init__(self, layout, mesh_creator):
        super().__init__()
        layout.addWidget(self)

        main_layout = QtWidgets.QVBoxLayout(self)

        self.mesh_creator = mesh_creator

        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

        self.material_parameters = QtWidgets.QGroupBox("Material Parameters")
        main_layout.addWidget(self.material_parameters)
        layout = QtWidgets.QVBoxLayout(self.material_parameters)

        self.input_k = QtShortCuts.QInputString(layout, "k", "1449", type=float)
        self.input_d0 = QtShortCuts.QInputString(layout, "d0", "0.00215", type=float)
        self.input_lamda_s = QtShortCuts.QInputString(layout, "lamdba_s", "0.032", type=float)
        self.input_ds = QtShortCuts.QInputString(layout, "ds", "0.055", type=float)

        self.material_parameters = QtWidgets.QGroupBox("Pressure Range")
        main_layout.addWidget(self.material_parameters)
        layout = QtWidgets.QVBoxLayout(self.material_parameters)

        layout2 = QtWidgets.QHBoxLayout()
        layout.addLayout(layout2)
        self.start = QtShortCuts.QInputString(layout2, "min", "0.1", type=float)
        self.end = QtShortCuts.QInputString(layout2, "max", "1000", type=float)
        self.n = QtShortCuts.QInputString(layout2, "count", "150", type=int)

        self.material_parameters = QtWidgets.QGroupBox("Iteration Parameters")
        main_layout.addWidget(self.material_parameters)
        layout = QtWidgets.QVBoxLayout(self.material_parameters)

        layout2 = QtWidgets.QHBoxLayout()
        layout.addLayout(layout2)
        self.max_iter = QtShortCuts.QInputString(layout2, "max_iter", "600", type=int)
        self.step = QtShortCuts.QInputString(layout2, "step", "0.0033", type=float)

        self.material_parameters = QtWidgets.QGroupBox("Run Parameters")
        main_layout.addWidget(self.material_parameters)
        layout = QtWidgets.QVBoxLayout(self.material_parameters)

        self.n_cores = QtShortCuts.QInputString(layout, "n_cores", "3", type=int)
        #layout=None, name=None, value=None, dialog_title="Choose File", file_type="All", filename_checker=None, existing=False, **kwargs):
        self.output = QtShortCuts.QInputFolder(layout, "Output Folder")

        main_layout.addStretch()

        url = "https://raw.githubusercontent.com/christophmark/jointforces/master/docs/data/spherical-inclusion.msh"
        self.localpath = "spherical-inclusion.msh"
        if not Path(self.localpath).exists():
            print("Downloading", url, "...")
            urllib.request.urlretrieve(str(url), self.localpath)

        layout2 = QtWidgets.QHBoxLayout()
        main_layout.addLayout(layout2)

        self.button_run = QtWidgets.QPushButton("run")
        self.button_run.clicked.connect(self.run)
        layout2.addStretch()
        layout2.addWidget(self.button_run)

        self.progressbar = QtWidgets.QProgressBar()
        main_layout.addWidget(self.progressbar)

        self.progress_signal.connect(self.progress_callback)

    def run(self):
        self.thread = threading.Thread(target=self.run_thread, daemon=True)
        self.thread.start()

    def progress_callback(self, i, n):
        self.progressbar.setRange(0, n)
        self.progressbar.setValue(i)

    def run_thread(self):
        out_table = Path(self.output.value())
        out_folder = out_table.parent / out_table.stem

        material = jf.materials.custom(self.input_k.value(),
                                       self.input_d0.value(),
                                       self.input_lamda_s.value(),
                                       self.input_ds.value(),
                                       )

        jf.simulation.distribute_solver('jf.simulation.spherical_contraction_solver',
                                        const_args={'meshfile': self.localpath,
                                                    # path to the provided or the new generated mesh
                                                    'outfolder': str(out_folder),
                                                    # output folder to store individual simulations
                                                    'max_iter': self.max_iter.value(),  # maximal iterationts for convergence
                                                    'step': self.step.value(),  # step size of iteration
                                                    'material': material},
                                        # Enter your own material parameters here
                                        var_arg='pressure', start=self.start.value(), end=self.end.value(),
                                        n=self.n.value(), log_scaling=True, n_cores=self.n_cores.value(),
                                        get_initial=True, callback=lambda i, n: self.progress_signal.emit(i, n))


        #lookup_table = jf.simulation.create_lookup_table_solver(str(out_folder), x0=1, x1=50,
        #                                                        n=100)  # output folder for combining the individual simulations
        #get_displacement, get_pressure = jf.simulation.create_lookup_functions(lookup_table)
        #jf.simulation.save_lookup_functions(get_displacement, get_pressure, str(out_table))


class LookUpTable2(QtWidgets.QWidget):
    progress_signal = QtCore.Signal(int, int)

    def __init__(self, layout, mesh_creator):
        super().__init__()
        layout.addWidget(self)

        main_layout = QtWidgets.QVBoxLayout(self)

        self.mesh_creator = mesh_creator

        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

        self.material_parameters = QtWidgets.QGroupBox("Generate Material Lookup Table")
        main_layout.addWidget(self.material_parameters)
        with QtShortCuts.QVBoxLayout(self.material_parameters) as layout:

            self.output = QtShortCuts.QInputFolder(layout, "Input Folder")

            with QtShortCuts.QHBoxLayout(layout) as layout2:
                self.x0 = QtShortCuts.QInputString(layout2, "x0", "1", type=float)
                self.x1 = QtShortCuts.QInputString(layout2, "x1", "50", type=float)
                self.n = QtShortCuts.QInputString(layout2, "n", "100", type=int)

            self.lookup_table = QtShortCuts.QInputFilename(layout, "Output Lookup Table", 'lookup_example.pkl', file_type="Pickle Lookup Table (*.pkl)")

            with QtShortCuts.QHBoxLayout(layout) as layout2:
                layout2.addStretch()
                self.button_run = QtShortCuts.QPushButton(layout2, "generate", self.run)

        self.material_parameters = QtWidgets.QGroupBox("Plot Material Lookup Table")
        main_layout.addWidget(self.material_parameters)
        with QtShortCuts.QVBoxLayout(self.material_parameters) as layout:

            with QtShortCuts.QHBoxLayout(layout):
                self.p0 = QtShortCuts.QInputString(None, "p0", "0", type=float)
                self.p1 = QtShortCuts.QInputString(None, "p1", "10000", type=float)
                self.d1 = QtShortCuts.QInputString(None, "d1", "2", type=float)
                self.d2 = QtShortCuts.QInputString(None, "d2", "50", type=float)

            self.lookup_table2 = QtShortCuts.QInputFilename(layout, "Input Lookup Table", 'lookup_example.pkl', file_type="Pickle Lookup Table (*.pkl)", existing=True)

            with QtShortCuts.QHBoxLayout(layout) as layout2:
                layout2.addStretch()
                self.button_run = QtShortCuts.QPushButton(layout2, "plot", self.run2)

        main_layout.addStretch()

    def run(self):
        lookup_table = jf.simulation.create_lookup_table_solver(str(self.output.value()), x0=int(self.x0.value()), x1=int(self.x1.value()),
                                                                n=int(self.n.value()))  # output folder for combining the individual simulations
        get_displacement, get_pressure = jf.simulation.create_lookup_functions(lookup_table)
        jf.simulation.save_lookup_functions(get_displacement, get_pressure, str(self.lookup_table.value()))

    def run2(self):
        figure = jf.simulation.plot_lookup_table(str(self.lookup_table2.value()), pressure=[float(self.p0.value()), float(self.p1.value())], distance=[float(self.d1.value()), float(self.d2.value())])


class Deformation(QtWidgets.QWidget):
    progress_signal = QtCore.Signal(int, int)

    def __init__(self, layout, mesh_creator):
        super().__init__()
        layout.addWidget(self)

        main_layout0 = QtShortCuts.QHBoxLayout(self)
        main_layout = QtShortCuts.QVBoxLayout(main_layout0)

        self.mesh_creator = mesh_creator

        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

        self.material_parameters = QtWidgets.QGroupBox("Generate Material Lookup Table")
        main_layout.addWidget(self.material_parameters)
        with QtShortCuts.QVBoxLayout(self.material_parameters) as layout:
            self.input_folder = QtShortCuts.QInputFolder(layout, "Input Folder", settings=self.settings, settings_key="spheriod/deformation/input")
            self.window_size = QtShortCuts.QInputString(layout, "window size", "50", type=int, settings=self.settings, settings_key="spheriod/deformation/window_siye")
            with QtShortCuts.QHBoxLayout(layout):
                self.wildcard = QtShortCuts.QInputString(None, "Wildcard", "*.tif", settings=self.settings, settings_key="spheriod/deformation/wildcard")
                self.n_min = QtShortCuts.QInputString(None, "n_min", "None", allow_none=True, type=int, settings=self.settings, settings_key="spheriod/deformation/n_min")
                self.n_max = QtShortCuts.QInputString(None, "n_max", "None", allow_none=True, type=int, settings=self.settings, settings_key="spheriod/deformation/n_max")

            with QtShortCuts.QHBoxLayout(layout):
                self.thres_segmentation = QtShortCuts.QInputString(None, "thres_segmentation", 0.9, type=float, settings=self.settings, settings_key="spheriod/deformation/thres_segmentation")
                self.continous_segmentation = QtShortCuts.QInputBool(None, "continous_segmentation", False, settings=self.settings, settings_key="spheriod/deformation/continous_segemntation")

            self.output_folder = QtShortCuts.QInputFolder(layout, "Output Folder", settings=self.settings, settings_key="spheriod/deformation/output")

        self.material_parameters = QtWidgets.QGroupBox("Plot Parameters")
        main_layout.addWidget(self.material_parameters)
        with QtShortCuts.QVBoxLayout(self.material_parameters) as layout:
            with QtShortCuts.QHBoxLayout(None):
                self.plot = QtShortCuts.QInputBool(None, "plot", True, settings=self.settings, settings_key="spheriod/deformation/plot")
                self.draw_mask = QtShortCuts.QInputBool(None, "draw mask", True, settings=self.settings, settings_key="spheriod/deformation/draw_mask")

            with QtShortCuts.QHBoxLayout(None):
                self.color_norm = QtShortCuts.QInputString(None, "color norm", 75., type=float, settings=self.settings, settings_key="spheriod/deformation/color_norm")
                self.cbar_um_scale = QtShortCuts.QInputString(None, "cbar_um_scale", None, allow_none=True, type=int, settings=self.settings, settings_key="spheriod/deformation/cbar_um_scale")
                self.quiver_scale = QtShortCuts.QInputString(None, "quiver_scale", 1, type=int, settings=self.settings, settings_key="spheriod/deformation/quiver_scale")

            self.dpi = QtShortCuts.QInputString(None, "dpi", 150, allow_none=True, type=int, settings=self.settings, settings_key="spheriod/deformation/dpi")
            self.dt_min = QtShortCuts.QInputString(None, "dt_min", None, allow_none=True, type=int, settings=self.settings, settings_key="spheriod/deformation/dt_min")

        self.button_run = QtShortCuts.QPushButton(main_layout, "run", self.run)

        main_layout.addStretch()

        main_layout = QtShortCuts.QVBoxLayout(main_layout0)

        self.slider = QtWidgets.QSlider()
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.slider_changed)
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        main_layout.addWidget(self.slider)

        self.label = QExtendedGraphicsView.QExtendedGraphicsView()
        self.label.setMinimumWidth(300)
        self.pixmap = QtWidgets.QGraphicsPixmapItem(self.label.origin)
        main_layout.addWidget(self.label)

        self.progressbar = QtWidgets.QProgressBar()
        main_layout.addWidget(self.progressbar)

        self.progress_signal.connect(self.progress_callback)

    def run(self):
        self.thread = threading.Thread(target=self.run_thread, daemon=True)
        self.thread.start()

    def progress_callback(self, i, n):
        self.progressbar.setRange(0, n)
        self.progressbar.setValue(i)
        # set the range for the slider
        self.slider.setRange(1, i)
        # it the slider was at the last value, move it to the new maximum
        if self.slider.value() == i-1:
            self.slider.setValue(i)

    def slider_changed(self, i):
        #im = plt.imread(fr"\\131.188.117.96\biophysDS2\dboehringer\Test_spheroid\data\20210416-165158_Mic5_rep{i:04d}_pos02_in-focus_modeBF_slice0_z0.tif")
        im = imageio.imread(str(self.output_folder.value()) + '/plot' + str(i).zfill(6) + '.png')
        self.pixmap.setPixmap(QtGui.QPixmap(array2qimage(im)))
        self.label.setExtend(im.shape[1], im.shape[0])

    def run_thread(self):
        print("comjpute displacements")
        jf.piv.compute_displacement_series(str(self.input_folder.value()),
                                           self.wildcard.value(),
                                           str(self.output_folder.value()),
                                           n_max=self.n_max.value(),
                                           n_min=self.n_min.value(),
                                           plot=self.plot.value(),
                                           draw_mask=self.draw_mask.value(),
                                           color_norm=self.color_norm.value(),
                                           cbar_um_scale=(self.cbar_um_scale.value()),
                                           quiver_scale=(self.quiver_scale.value()),
                                           dpi=(self.dpi.value()),
                                           continous_segmentation=self.continous_segmentation.value(),
                                           thres_segmentation=(self.thres_segmentation.value()),
                                           window_size=(self.window_size.value()),
                                           dt_min=(self.dt_min.value()),
                                           cutoff=650, cmap="turbo",
                                           callback=lambda i, n: self.progress_signal.emit(i, n))



class MainWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # QSettings
        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

        self.setMinimumWidth(800)
        self.setMinimumHeight(400)
        self.setWindowTitle("Saenopy Viewer")

        main_layout = QtWidgets.QHBoxLayout(self)

        self.tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(self.tabs)


        self.tab_stack = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_stack, "Simulations")
        v_layout = QtWidgets.QVBoxLayout(self.tab_stack)
        h_layout = QtWidgets.QHBoxLayout()
        v_layout.addLayout(h_layout)

        self.stack_before = LookUpTable(h_layout, self)
        self.description = QtWidgets.QTextEdit()
        self.description.setDisabled(True)
        h_layout.addWidget(self.description)
        self.description.setText("""
<h1>Step 1: Lookup Table</h1>
First we need to load the image data of the recorded stacks.<br/>
<br/>
Select the image stacks for the <b>deformed</b> and the <b>relaxed</b> state. In the deformed state the cell should pull on the 
fibers and in the relaxed state there should be no cell or a non pulling cell (e.g. treated with CytoD).<br/>
<br/>
Images can either be a single Leica file or a .tif file with a naming sceme that allows to find the other images of the 
stack in the same folder.<br/>
<br/>
You can select which image <b>channel</b> to use for evaluation.<br/>
<br/>
If you load a custom stack, you need to specifiy which dimension is the <b>z-component</b>.
And you need to supply a <b>voxel size</b>.<br/>
<br/>
<br/>
<b>Hint</b>: you can optionally <b>export</b> the current z slices as images or as a gif. You can also right click and drag on 
the slider to make a <b>z-projection</b> over the selected z range.
         """.strip())
#TODO make image loader also with wildcard

        v_layout.addWidget(QHLine())
        h_layout = QtWidgets.QHBoxLayout()
        v_layout.addLayout(h_layout)

        h_layout.addStretch()
        self.button_next = QtWidgets.QPushButton("next")
        self.button_next.clicked.connect(self.next)
        h_layout.addWidget(self.button_next)

        """ """
        self.tab_stack = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_stack, "Lookup Table")
        v_layout = QtWidgets.QVBoxLayout(self.tab_stack)
        h_layout = QtWidgets.QHBoxLayout()
        v_layout.addLayout(h_layout)

        self.stack_before = LookUpTable2(h_layout, self)
        self.description = QtWidgets.QTextEdit()
        self.description.setDisabled(True)
        h_layout.addWidget(self.description)
        self.description.setText("""
<h1>Step 2: Deformation Detection</h1>
Now we need to extract the deformations from the recorded image data.<br/>
<br/>
This part uses a 3D PIV (particle image velocimetry) algorithm to determine how much each voxel has shifted.

<h2>Parameters</h2>
<ul>
<li><b>Overlap</b>: the fraction by wich two neighbouring windows overlap.
Higher overlap increases the resolution (and calculation time). Default is 0.6.
</li>
<li><b>Window size</b>: 
A cube of this size (um) will be shifted in space until the best match is found. 
The window size should be bigger
than the expected deformation. Default is 30um.</li>
<li><b>Signoise</b>: above which signal to noice level to discatd deformations.</li>
<li><b>Driftcorrection</b>: whether to perform drift correction or not.</li>
</ul>
                 """.strip())
# TODO: check the error message for Out of Memory
# TODO: add mesh size calculation based on the current values, stack size

        v_layout.addWidget(QHLine())
        h_layout = QtWidgets.QHBoxLayout()
        v_layout.addLayout(h_layout)

        h_layout.addStretch()
        self.button_next = QtWidgets.QPushButton("next")
        self.button_next.clicked.connect(self.next)
        self.button_previous = QtWidgets.QPushButton("back")
        self.button_previous.clicked.connect(self.previous)
        h_layout.addWidget(self.button_previous)
        h_layout.addWidget(self.button_next)

        """ """
        self.tab_stack = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_stack, "Deformation")
        v_layout = QtWidgets.QVBoxLayout(self.tab_stack)
        h_layout = QtWidgets.QHBoxLayout()
        v_layout.addLayout(h_layout)

        self.deformations = Deformation(h_layout, self)
        self.description = QtWidgets.QTextEdit()
        self.description.setDisabled(True)
        h_layout.addWidget(self.description)
        self.description.setText("""
        <h1>Step 3: Mesh Generation</h1>
        Now we need to create a mesh and interpolate the displacements onto this mesh.<br/>
        <br/>

        <h2>Parameters</h2>
        <ul>
        <li><b>Element size</b>: spacing between nodes of the mesh in um.</li>
        <li><b>Inner Region</b>: the spacing will be exactly the one given in element size in a centerl region of this size.</li>
        <li><b>Thinning Factor</b>: how much to thin the elements outside the inner region. 1</li>
        </ul>
                         """.strip())
#TODO add better description of the thining
        v_layout.addWidget(QHLine())
        h_layout = QtWidgets.QHBoxLayout()
        v_layout.addLayout(h_layout)

        h_layout.addStretch()
        self.button_next = QtWidgets.QPushButton("next")
        self.button_next.clicked.connect(self.next)
        self.button_previous = QtWidgets.QPushButton("back")
        self.button_previous.clicked.connect(self.previous)
        h_layout.addWidget(self.button_previous)
        h_layout.addWidget(self.button_next)

    def load(self):
        files = glob.glob(self.input_filename.value())
        self.input_label.setText("\n".join(files))
#        self.input_filename

    def next(self):
        self.tabs.setCurrentIndex(self.tabs.currentIndex()+1)

    def previous(self):
        self.tabs.setCurrentIndex(self.tabs.currentIndex()-1)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    print(sys.argv)
    window = MainWindow()
    if len(sys.argv) >= 2:
        window.loadFile(sys.argv[1])
    window.show()
    sys.exit(app.exec_())
