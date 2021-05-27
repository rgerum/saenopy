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

import ctypes

def kill_thread(thread):
    """
    thread: a threading.Thread object
    """
    thread_id = thread.ident
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(SystemExit))
    if res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
        print('Exception raise failure')

class QHLine(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


class LookUpTable(QtWidgets.QWidget):
    progress_signal = QtCore.Signal(int, int)
    finished_signal = QtCore.Signal()
    thread = None

    def __init__(self, layout, mesh_creator):
        super().__init__()
        layout.addWidget(self)

        main_layout = QtWidgets.QVBoxLayout(self)

        self.mesh_creator = mesh_creator

        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

        self.material_parameters = QtWidgets.QGroupBox("Material Parameters")
        main_layout.addWidget(self.material_parameters)

        with QtShortCuts.QVBoxLayout(self.material_parameters) as layout:
            with QtShortCuts.QHBoxLayout(None):
                self.input_k = QtShortCuts.QInputString(None, "k", "1449", type=float)
                self.input_d0 = QtShortCuts.QInputString(None, "d0", "0.00215", type=float)
            with QtShortCuts.QHBoxLayout(None):
                self.input_lamda_s = QtShortCuts.QInputString(None, "lamdba_s", "0.032", type=float)
                self.input_ds = QtShortCuts.QInputString(None, "ds", "0.055", type=float)

        self.material_parameters = QtWidgets.QGroupBox("Pressure Range")
        main_layout.addWidget(self.material_parameters)
        with QtShortCuts.QHBoxLayout(self.material_parameters):
            self.start = QtShortCuts.QInputString(None, "min", "0.1", type=float)
            self.end = QtShortCuts.QInputString(None, "max", "1000", type=float)
            self.n = QtShortCuts.QInputString(None, "count", "150", type=int)

        self.material_parameters = QtWidgets.QGroupBox("Iteration Parameters")
        main_layout.addWidget(self.material_parameters)
        with QtShortCuts.QHBoxLayout(self.material_parameters):
            self.max_iter = QtShortCuts.QInputString(None, "max_iter", "600", type=int)
            self.step = QtShortCuts.QInputString(None, "step", "0.0033", type=float)

        self.material_parameters = QtWidgets.QGroupBox("Run Parameters")
        main_layout.addWidget(self.material_parameters)
        layout = QtWidgets.QVBoxLayout(self.material_parameters)

        self.n_cores = QtShortCuts.QInputNumber(layout, "n_cores", 1, float=False)
        #layout=None, name=None, value=None, dialog_title="Choose File", file_type="All", filename_checker=None, existing=False, **kwargs):
        self.output = QtShortCuts.QInputFolder(layout, "Output Folder")

        main_layout.addStretch()

        self.input_list = [
            self.input_k,
            self.input_d0,
            self.input_lamda_s,
            self.input_ds,
            self.start,
            self.end,
            self.n,
            self.max_iter,
            self.step,
            self.n_cores,
            self.output,
        ]

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
        self.finished_signal.connect(self.finished)

    def run(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self.run_thread, daemon=True)
            self.thread.start()
            self.button_run.setText("stop")
            for widget in self.input_list:
                widget.setDisabled(True)
        else:
            kill_thread(self.thread)
            self.thread = None
            self.button_run.setText("run")
            for widget in self.input_list:
                widget.setDisabled(False)

    def finished(self):
        self.thread = None
        self.button_run.setText("run")
        for widget in self.input_list:
            widget.setDisabled(False)

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
        self.finished_signal.emit()

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
    finished_signal = QtCore.Signal()
    thread = None

    def __init__(self, layout, mesh_creator):
        super().__init__()
        layout.addWidget(self)

        main_layout0 = QtShortCuts.QHBoxLayout(self)
        main_layout = QtShortCuts.QVBoxLayout(main_layout0)

        self.mesh_creator = mesh_creator

        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

        self.material_parameters = QtWidgets.QGroupBox("Measure Matrix Deformations")
        main_layout.addWidget(self.material_parameters)
        with QtShortCuts.QVBoxLayout(self.material_parameters) as layout:
            self.input_folder = QtShortCuts.QInputFolder(layout, "Raw Images (Folder)", settings=self.settings, settings_key="spheriod/deformation/input")
            self.window_size = QtShortCuts.QInputString(layout, "window size", "50", type=int, settings=self.settings, settings_key="spheriod/deformation/window_siye")
            with QtShortCuts.QHBoxLayout(layout):
                self.wildcard = QtShortCuts.QInputString(None, "Wildcard", "*.tif", settings=self.settings, settings_key="spheriod/deformation/wildcard")
                self.n_min = QtShortCuts.QInputString(None, "n_min", "None", allow_none=True, type=int, settings=self.settings, settings_key="spheriod/deformation/n_min")
                self.n_max = QtShortCuts.QInputString(None, "n_max", "None", allow_none=True, type=int, settings=self.settings, settings_key="spheriod/deformation/n_max")

            with QtShortCuts.QHBoxLayout(layout):
                self.thres_segmentation = QtShortCuts.QInputString(None, "thres_segmentation", 0.9, type=float, settings=self.settings, settings_key="spheriod/deformation/thres_segmentation")
                self.continous_segmentation = QtShortCuts.QInputBool(None, "continous_segmentation", False, settings=self.settings, settings_key="spheriod/deformation/continous_segemntation")

            self.output_folder = QtShortCuts.QInputFolder(layout, "Result Folder", settings=self.settings, settings_key="spheriod/deformation/output")

        self.material_parameters = QtWidgets.QGroupBox("Plot Matrix Deformations")
        main_layout.addWidget(self.material_parameters)
        with QtShortCuts.QVBoxLayout(self.material_parameters) as layout:
            with QtShortCuts.QHBoxLayout(None):
                self.plot = QtShortCuts.QInputBool(None, "plot", True, settings=self.settings, settings_key="spheriod/deformation/plot")
                #self.draw_mask = QtShortCuts.QInputBool(None, "draw mask", True, settings=self.settings, settings_key="spheriod/deformation/draw_mask")

            with QtShortCuts.QHBoxLayout(None):
                self.color_norm = QtShortCuts.QInputString(None, "color norm", 75., type=float, settings=self.settings, settings_key="spheriod/deformation/color_norm")
                self.cbar_um_scale = QtShortCuts.QInputString(None, "cbar_um_scale", None, allow_none=True, type=int, settings=self.settings, settings_key="spheriod/deformation/cbar_um_scale")
                self.quiver_scale = QtShortCuts.QInputString(None, "quiver_scale", 1, type=int, settings=self.settings, settings_key="spheriod/deformation/quiver_scale")

            self.dpi = QtShortCuts.QInputString(None, "dpi", 150, allow_none=True, type=int, settings=self.settings, settings_key="spheriod/deformation/dpi")
            self.dt_min = QtShortCuts.QInputString(None, "dt_min", None, allow_none=True, type=int, settings=self.settings, settings_key="spheriod/deformation/dt_min")

        self.button_run = QtShortCuts.QPushButton(main_layout, "run", self.run)

        self.input_list = [
            self.input_folder,
            self.window_size,
            self.wildcard,
            self.n_max,
            self.n_min,
            self.thres_segmentation,
            self.continous_segmentation,
            self.output_folder,
            self.plot,
            self.color_norm,
            self.cbar_um_scale,
            self.quiver_scale,
            self.dpi,
            self.dt_min,
        ]

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
        self.finished_signal.connect(self.finished)

    def run(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self.run_thread, daemon=True)
            self.thread.start()
            self.button_run.setText("stop")
            for widget in self.input_list:
                widget.setDisabled(True)
        else:
            kill_thread(self.thread)
            self.thread = None
            self.button_run.setText("run")
            for widget in self.input_list:
                widget.setDisabled(False)

    def finished(self):
        self.thread = None
        self.button_run.setText("run")
        for widget in self.input_list:
            widget.setDisabled(False)

    def progress_callback(self, i, n):
        self.progressbar.setRange(0, n-1)
        self.progressbar.setValue(i)
        # when plotting show the slider
        if self.plot.value() is True:
            # set the range for the slider
            self.slider.setRange(1, i)
            # it the slider was at the last value, move it to the new maximum
            if self.slider.value() == i-1:
                self.slider.setValue(i)

    def slider_changed(self, i):
        if self.plot.value() is True:
            #im = plt.imread(fr"\\131.188.117.96\biophysDS2\dboehringer\Test_spheroid\data\20210416-165158_Mic5_rep{i:04d}_pos02_in-focus_modeBF_slice0_z0.tif")
            im = imageio.imread(str(self.output_folder.value()) + '/plot' + str(i).zfill(6) + '.png')
            self.pixmap.setPixmap(QtGui.QPixmap(array2qimage(im)))
            self.label.setExtend(im.shape[1], im.shape[0])

    def run_thread(self):
        print("compute displacements")
        jf.piv.compute_displacement_series(str(self.input_folder.value()),
                                           self.wildcard.value(),
                                           str(self.output_folder.value()),
                                           n_max=self.n_max.value(),
                                           n_min=self.n_min.value(),
                                           plot=self.plot.value(),
                                           draw_mask=False,
                                           color_norm=self.color_norm.value(),
                                           cbar_um_scale=(self.cbar_um_scale.value()),
                                           quiver_scale=(self.quiver_scale.value()),
                                           dpi=(self.dpi.value()),
                                           continous_segmentation=self.continous_segmentation.value(),
                                           thres_segmentation=(self.thres_segmentation.value()),
                                           window_size=(self.window_size.value()),
                                           dt_min=(self.dt_min.value()),
                                           cutoff=None, cmap="turbo",
                                           callback=lambda i, n: self.progress_signal.emit(i, n))
        self.finished_signal.emit()



class Force(QtWidgets.QWidget):
    progress_signal = QtCore.Signal(int, int)

    def __init__(self, layout, mesh_creator):
        super().__init__()
        layout.addWidget(self)

        main_layout = QtWidgets.QVBoxLayout(self)

        self.mesh_creator = mesh_creator

        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

        self.material_parameters = QtWidgets.QGroupBox("Generate Forces")
        main_layout.addWidget(self.material_parameters)
        with QtShortCuts.QVBoxLayout(self.material_parameters) as layout:

            self.output = QtShortCuts.QInputFolder(layout, "Result Folder")
            self.lookup_table = QtShortCuts.QInputFilename(layout, "Lookup Table", 'lookup_example.pkl', file_type="Pickle Lookup Table (*.pkl)", existing=True)

            self.pixel_size = QtShortCuts.QInputString(layout, "pixel_size", "1.29", type=float)

            with QtShortCuts.QHBoxLayout(layout) as layout2:
                self.x0 = QtShortCuts.QInputString(layout2, "r_min", "2", type=float)
                self.x1 = QtShortCuts.QInputString(layout2, "r_max", "None", type=float, allow_none=True)

            with QtShortCuts.QHBoxLayout(layout) as layout2:
                layout2.addStretch()
                self.button_run = QtShortCuts.QPushButton(layout2, "run", self.run)

        self.material_parameters = QtWidgets.QGroupBox("Plot Forces")
        main_layout.addWidget(self.material_parameters)
        with QtShortCuts.QVBoxLayout(self.material_parameters) as layout:
            self.type = QtShortCuts.QInputChoice(None, "type", "Pressure", ["Pressure", "Contractility"])
            self.dt = QtShortCuts.QInputString(None, "dt", "2", type=float)
            with QtShortCuts.QHBoxLayout(None) as layout2:
                layout2.addStretch()
                self.button_run = QtShortCuts.QPushButton(layout2, "plot", self.run2)

        main_layout.addStretch()

    def run(self):
        jf.force.reconstruct(str(self.output.value()),  # PIV output folder
                             str(self.lookup_table.value()),  # lookup table
                             self.pixel_size.value(),  # pixel size (µm)
                             None, r_min=self.x0.value(), r_max=self.x1.value())

    def run2(self):
        res = pd.read_excel(Path(self.output.value()) / "result.xlsx")

        t = np.arange(len(res)) * self.dt.value() / 60
        print("self.type.value()", self.type.value())
        if self.type.value() == "Contractility":
            mu = res['Mean Contractility (µN)']
            std = res['St.dev. Contractility (µN)']
        else:
            mu = res['Mean Pressure (Pa)']
            std = res['St.dev. Pressure (Pa)']

        plt.figure(figsize=(6, 3))
        plt.plot(t, mu, lw=2, c='C0')
        plt.fill_between(t, mu - std, mu + std, facecolor='C0', lw=0, alpha=0.5)
        plt.grid()
        plt.xlabel('Time (h)')
        if self.type.value() == "Contractility":
            plt.ylabel('Contractility (µN)')
        else:
            plt.ylabel('Pressure (Pa)')
        plt.tight_layout()
        plt.show()

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

<h1>Step 1: Material Simulations </h1><br/>

To calculate the collective forces that spheroids and organoids excert on the surrounding
matrix, we generate material lookup-tables that predict the contractilile pressure
from the size of the matrix deformations as a function of the distance to the spheroid center as 
described in Mark et al. (2020). <br/>
<br/>

To generate a material lookup-table we model the nonlinear fiber material according to the
 given material properties <b>k</b>, <b>d0</b>, <b>ds</b> and <b>lambda_s</b> 
 (see  Saenopy documentation and material fitting for more information).<br/>
 <i> Default values are taken from a collagen I (1.2mg/ml) hydrogel with K_0=1449, D_0=0.00215, D_S=0.055, lambda_s=0.032.</i><br/>
<br/>

 The simulations then approximate the multicellular aggregate as a spherical inclusion that is
 surrounded by the user-defined nonlinear material and excerts a contractile pressure on the matrix. We conduct a range of simulations   
for n=<b>count</b> different pressures (default 150) that are spaced between a pressure of <b>min</b> Pa (default 600) 
and <b>max</b> Pa (default 1000). <br/>
<br/>


For the individual simulations the maximal allowed number of iteration before convergence 
is limited by <b>max_iter</b> (default 600) and the stepwidth between iteration is given by <b>step</b> 
(default is 0.0033). <br/>
<br/>


All simulations are stored in the <b>Output folder</b>, which should be empty 
before starting the simulations. After the simulations are finished, the lookup-table 
can be generated from the obtained files in the next step. <br/>
<br/>


<b>Hint</b>: This step can take several hours or even days. Depending on the used computer <b>n_cores</b>
can be increased to speed up the calculations. Attention: If you overload the memory an error 
will lead the simulations to crash. Therefore the default value is 1. The material simulations need 
to be conducted only a single time for a certain material. Additionally, lookuptables for purely linear fiber material (assuming poission ratio of v=0.25) with 
arbitrary Youngs modulus can be created without conducting simulations
using the XXXX function in the next step.



         """.strip())
#TODO make image loader also with wildcard

        r"""
        
        E:\saenopy\saenopy\output_example\*\+.tif
        
        E:\saenopy\saenopy\input\20212031\david\well*_pos*_time+.tif
        
        E:\saenopy\saenopy\input\20212030\david\well*_pos*_time+.tif
        D:\meine_auswertrung\output_analysis\20212030\david\
        
        E:\saenopy\saenopy\output_analysis\datum\david\well*_pos*_time+.tif
        
        
        E:\saenopy\saenopy\output_example2\well*_pos*_time+.tif
        
        E:\saenopy\saenopy\output_example2\well1_pos1_time00\result.xls
        E:\saenopy\saenopy\output_example2\well1_pos1_time00.tif
        E:\saenopy\saenopy\output_example2\well1_pos1_time01.tif
        E:\saenopy\saenopy\output_example2\well1_pos1_time02.tif
        
        
        """
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

        """ """
        self.tab_stack = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_stack, "Force")
        v_layout = QtWidgets.QVBoxLayout(self.tab_stack)
        h_layout = QtWidgets.QHBoxLayout()
        v_layout.addLayout(h_layout)

        self.deformations = Force(h_layout, self)
        self.description = QtWidgets.QTextEdit()
        self.description.setDisabled(True)
        h_layout.addWidget(self.description)
        self.description.setText("""
                <h1>Step 4: Force</h1>
                Now we need to create a mesh and interpolate the displacements onto this mesh.<br/>
                <br/>

                <h2>Parameters</h2>
                <ul>
                <li><b>Element size</b>: spacing between nodes of the mesh in um.</li>
                <li><b>Inner Region</b>: the spacing will be exactly the one given in element size in a centerl region of this size.</li>
                <li><b>Thinning Factor</b>: how much to thin the elements outside the inner region. 1</li>
                </ul>
                                 """.strip())
        # TODO add better description of the thining
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
