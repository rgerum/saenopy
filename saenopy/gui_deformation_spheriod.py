import sys

# Setting the Qt bindings for QtPy
import os

import pandas as pd
import qtawesome as qta

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
import glob
import re
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

from qtpy import API_NAME as QT_API_NAME
if QT_API_NAME.startswith("PyQt4"):
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as Canvas
    from matplotlib.backends.backend_qt4agg import FigureManager
    from matplotlib.backends.backend_qt4 import NavigationToolbar2QT as NavigationToolbar
else:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
    from matplotlib.backends.backend_qt5agg import FigureManager
    from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar


class MatplotlibWidget(Canvas):

    def __init__(self, parent=None, width=4, height=3, dpi=100):
        from matplotlib import _pylab_helpers
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

def execute(func, *args, **kwargs):
    func(*args, **kwargs)
    import inspect
    code_lines = inspect.getsource(func).split("\n")[1:]
    indent = len(code_lines[0]) - len(code_lines[0].lstrip())
    code = "\n".join(line[indent:] for line in code_lines)
    for key, value in kwargs.items():
        if isinstance(value, str):
            code = code.replace(key, "'"+value+"'")
        else:
            code = code.replace(key, value)
    code = code.replace("self.canvas.draw()", "plt.show()")
    return code

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

"""
class Spoiler(QtWidgets.QWidget):
    def __int__(self, title, animationDuration, parent):
        toggleButton = QtWidgets.QToolButton()
        toggleButton.setStyleSheet("QToolButton { border: none; }")
        toggleButton.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        toggleButton.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        toggleButton.setText(title)
        toggleButton.setCheckable(True)
        toggleButton.setChecked(False)

        headerLine = QtWidgets.QFrame()
        headerLine.setFrameShape(QtWidgets.QFrame.HLine)
        headerLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        headerLine.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Maximum)
    
        contentArea = QtWidgets.QScrollArea()
        contentArea.setStyleSheet("QScrollArea { background-color: white; border: none; }")
        contentArea.setSizePolicy(QtGui..QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        # start out collapsed
        contentArea.setMaximumHeight(0);
        contentArea.setMinimumHeight(0);
        #let the entire widget grow and shrink with its content
        toggleAnimation.addAnimation(new QPropertyAnimation(this, "minimumHeight"));
        toggleAnimation.addAnimation(new QPropertyAnimation(this, "maximumHeight"));
        toggleAnimation.addAnimation(new QPropertyAnimation(&contentArea, "maximumHeight"));
        // don't waste space
        mainLayout.setVerticalSpacing(0);
        mainLayout.setContentsMargins(0, 0, 0, 0);
        int row = 0;
        mainLayout.addWidget(&toggleButton, row, 0, 1, 1, Qt::AlignLeft);
        mainLayout.addWidget(&headerLine, row++, 2, 1, 1);
        mainLayout.addWidget(&contentArea, row, 0, 1, 3);
        setLayout(&mainLayout);
        QObject::connect(&toggleButton, &QToolButton::clicked, [this](const bool checked) {
            toggleButton.setArrowType(checked ? Qt::ArrowType::DownArrow : Qt::ArrowType::RightArrow);
            toggleAnimation.setDirection(checked ? QAbstractAnimation::Forward : QAbstractAnimation::Backward);
            toggleAnimation.start();
        });
}

void Spoiler::setContentLayout(QLayout & contentLayout) {
    delete contentArea.layout();
    contentArea.setLayout(&contentLayout);
    const auto collapsedHeight = sizeHint().height() - contentArea.maximumHeight();
    auto contentHeight = contentLayout.sizeHint().height();
    for (int i = 0; i < toggleAnimation.animationCount() - 1; ++i) {
        QPropertyAnimation * spoilerAnimation = static_cast<QPropertyAnimation *>(toggleAnimation.animationAt(i));
        spoilerAnimation->setDuration(animationDuration);
        spoilerAnimation->setStartValue(collapsedHeight);
        spoilerAnimation->setEndValue(collapsedHeight + contentHeight);
    }
    QPropertyAnimation * contentAnimation = static_cast<QPropertyAnimation *>(toggleAnimation.animationAt(toggleAnimation.animationCount() - 1));
    contentAnimation->setDuration(animationDuration);
    contentAnimation->setStartValue(0);
    contentAnimation->setEndValue(contentHeight);
}
"""
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

        with QtShortCuts.QGroupBox(main_layout, "Generate Forces") as (_, layout):
            self.output = QtShortCuts.QInputFolder(layout, "Result Folder")
            self.lookup_table = QtShortCuts.QInputFilename(layout, "Lookup Table", 'lookup_example.pkl', file_type="Pickle Lookup Table (*.pkl)", existing=True)

            self.pixel_size = QtShortCuts.QInputString(layout, "pixel_size", "1.29", type=float)

            with QtShortCuts.QHBoxLayout(layout) as layout2:
                self.x0 = QtShortCuts.QInputString(layout2, "r_min", "2", type=float)
                self.x1 = QtShortCuts.QInputString(layout2, "r_max", "None", type=float, allow_none=True)

            with QtShortCuts.QHBoxLayout(layout) as layout2:
                layout2.addStretch()
                self.button_run = QtShortCuts.QPushButton(layout2, "run", self.run)

        with QtShortCuts.QGroupBox(main_layout, "Plot Forces") as (_, layout):
            self.type = QtShortCuts.QInputChoice(None, "type", "Pressure", ["Pressure", "Contractility"])
            self.dt = QtShortCuts.QInputString(None, "dt", "2", type=float)
            with QtShortCuts.QHBoxLayout() as layout2:
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

        with QtShortCuts.QTabWidget(main_layout) as self.tabs:
            with self.tabs.createTab("Simulations") as v_layout:
                with QtShortCuts.QHBoxLayout() as h_layout:
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

To generate a material lookup-table, we model the nonlinear fiber material according to the
 given material properties <b>k</b>, <b>d0</b>, <b>ds</b> and <b>lambda_s</b> 
 (see Saenopy documentation for more information).<br/>
 <i> Default values are taken from a collagen I hydrogel (1.2mg/ml) with k=1449, d0=0.00215, ds=0.055, lambda_s=0.032.</i><br/>
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

                v_layout.addWidget(QHLine())
                with QtShortCuts.QHBoxLayout() as h_layout:
                    h_layout.addStretch()
                    self.button_next = QtWidgets.QPushButton("next")
                    self.button_next.clicked.connect(self.next)
                    h_layout.addWidget(self.button_next)

            """ """
            with self.tabs.createTab("Lookup Table") as v_layout:
                with QtShortCuts.QHBoxLayout() as h_layout:

                    self.stack_before = LookUpTable2(h_layout, self)
                    self.description = QtWidgets.QTextEdit()
                    self.description.setDisabled(True)
                    h_layout.addWidget(self.description)
                    self.description.setText("""
<h1>Step 2: Material Lookup-Table</h1>

From the individual simulations we  now generate our material lookup-table. <br/>
<br/>

Therefore, we load the <b>Input folder</b> where the indivudal simulations are stored.<b>Generate</b> will then create and save the material-lookuptable 
as a *.pkl-file for further force analysis in the specified <b>Output Lookup Table</b> location.<br/>
<br/>
<i> If desired the interpolation of data points from the FE-simulations to different relative distances can be changed.  
By default a grid with <b>n</b>=150 logarithmically spaced intervals is generated between a
 distance of <b>x0</b>=1 effective spheroid radii to a distance of <b>x1</b>=50 effective spheroid radii away from the center.</i><br/>
<br/>

The material lookup-table can be visualized by reading in the *.pkl-file as <b>Input Lookup Table</b> and adjust the desired range
of pressures (from <b>p0</b> to <b>p1</b> Pascal) and distances (from <b>d1</b> to <b>d2</b> in 
effective spheroid radii). <br/>
<br/>

Add Data to plot ? ToDo <br/>
<br/>

Additionally, lookuptables for purely linear fiber material (assuming poission ratio of v=0.25) with 
arbitrary Youngs modulus can be created without conducting simulations
using the XXXX function.
                 """.strip())

                v_layout.addWidget(QHLine())
                with QtShortCuts.QHBoxLayout() as h_layout:
                    h_layout.addStretch()
                    self.button_previous = QtShortCuts.QPushButton(None, "back", self.previous)
                    self.button_next = QtShortCuts.QPushButton(None, "next", self.next)

        """ """
        with self.tabs.createTab("Deformation") as v_layout:
            with QtShortCuts.QHBoxLayout() as h_layout:
                #self.deformations = Deformation(h_layout, self)
                self.deformations = BatchEvaluate(self)
                h_layout.addWidget(self.deformations)
                self.description = QtWidgets.QTextEdit()
                self.description.setDisabled(True)
                h_layout.addWidget(self.description)
                self.description.setText("""
                <h1>Step 3: Deformation Detection</h1>
                Now we need to extract the deformations from the recorded image data. <br/>
                <br/>
                Therefore we use 2D PIV (particle image velocimetry) algorithm to determine the movement 
                of beads that surround the spheroid in the equatorial plane. By exploitng spherical symetry this deformation 
                is then simply compared to the 3D simulations by using our material lookup-table. <br/>
                <br/>
                
                <h2>Parameters</h2>
                <ul>
                <li><b>Raw Images</b>: Path to the folder containing the raw image data.</li>
                <li><b>Wildcard</b>: Wildcard to selection elements within this folder (Example: Pos01_t*.tif; star will allow all timesteps). </li>
                <li><b>n_min, n_max</b>: Set a minimum or maximum elemnet if first or last time steps from this selection should be ignored (default is None). 1</li>
                <li><b>thres_segmentation</b>: Factor to change the threshold for spheroid segmentation (default is 0.9). 1</li>
                <li><b>continous_segmentation</b>: If active, the segmentation is repeated for every timestep individually.
                By default we use the segmentation of the first time step (less sensitive to fluctuations)  </li>
                <li><b>thres_segmentation</b>: Factor to change the threshold for spheroid segmentation (default is 0.9). 1</li>
                
                </ul>
                                 """.strip())
            v_layout.addWidget(QHLine())
            with QtShortCuts.QHBoxLayout() as h_layout:
                h_layout.addStretch()
                self.button_previous = QtShortCuts.QPushButton(None, "back", self.previous)
                self.button_next = QtShortCuts.QPushButton(None, "next", self.next)

        """ """
        with self.tabs.createTab("Force") as v_layout:
            with QtShortCuts.QHBoxLayout() as h_layout:

                #self.deformations = Force(h_layout, self)
                self.deformations = PlottingWindow(self)
                h_layout.addWidget(self.deformations)

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
            with QtShortCuts.QHBoxLayout() as h_layout:
                h_layout.addStretch()
                self.button_previous = QtShortCuts.QPushButton(None, "back", self.previous)
                self.button_next = QtShortCuts.QPushButton(None, "next", self.next)

    def load(self):
        files = glob.glob(self.input_filename.value())
        self.input_label.setText("\n".join(files))
#        self.input_filename

    def next(self):
        self.tabs.setCurrentIndex(self.tabs.currentIndex()+1)

    def previous(self):
        self.tabs.setCurrentIndex(self.tabs.currentIndex()-1)




class QSlider(QtWidgets.QSlider):
    min = None
    max = None
    evaluated = 0

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        if self.maximum()-self.minimum():
            if self.min is not None:
                self.drawRect(0, self.min, "gray", 0.2)
            if self.max is not None:
                value = self.max
                if self.max < 0:
                    value = self.maximum()+self.max
                self.drawRect(value, self.maximum(), "gray", 0.2)
            self.drawRect(self.min, self.evaluated, "lightgreen", 0.3)
        super().paintEvent(ev)

    def drawRect(self, start, end, color, border):
        p = QtGui.QPainter(self)
        p.setPen(QtGui.QPen(QtGui.QColor("transparent")))
        p.setBrush(QtGui.QBrush(QtGui.QColor(color)))
        s = self.width() * (start - self.minimum()) / (self.maximum() - self.minimum())
        e = self.width() * (end - self.minimum()) / (self.maximum() - self.minimum())
        p.drawRect(s, self.height()*border,
                   e-s, self.height()*(1-border*2))

    def setEvaluated(self, value):
        self.evaluated = value
        self.update()


class BatchEvaluate(QtWidgets.QWidget):
    progress_signal = QtCore.Signal(int, int, int, int)
    finished_signal = QtCore.Signal()
    thread = None

    def __init__(self, parent=None):
        super().__init__(parent)

        # QSettings
        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

        self.setMinimumWidth(800)
        self.setMinimumHeight(400)
        self.setWindowTitle("Saenopy Viewer")

        with QtShortCuts.QHBoxLayout(self) as main_layout:
            with QtShortCuts.QVBoxLayout() as layout:
                self.inputText = QtShortCuts.QInputFilename(None, "input_wildcard", settings=self.settings, settings_key="batch/wildcard", existing=True, allow_edit=True)
                self.outputText = QtShortCuts.QInputFolder(None, "output", settings=self.settings, settings_key="batch/wildcard2", allow_edit=True)
                with QtShortCuts.QHBoxLayout() as layout2:
                    self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                    self.button_addList = QtShortCuts.QPushButton(None, "add to list", self.show_files)
                self.list = QtWidgets.QListWidget()
                layout.addWidget(self.list)
                self.list.itemSelectionChanged.connect(self.listSelected)
                self.progress1 = QtWidgets.QProgressBar()
                layout.addWidget(self.progress1)

            with QtShortCuts.QVBoxLayout() as layout:
                self.slider = QSlider()
                self.slider.setRange(0, 0)
                self.slider.valueChanged.connect(self.slider_changed)
                self.slider.setOrientation(QtCore.Qt.Horizontal)
                layout.addWidget(self.slider)

                self.label_text = QtWidgets.QLabel()
                layout.addWidget(self.label_text)

                self.label = QExtendedGraphicsView.QExtendedGraphicsView()
                self.label.setMinimumWidth(300)
                self.pixmap = QtWidgets.QGraphicsPixmapItem(self.label.origin)
                self.contour = QtWidgets.QGraphicsPathItem(self.label.origin)
                pen = QtGui.QPen(QtGui.QColor(255, 0, 0))
                pen.setCosmetic(True)
                self.contour.setPen(pen)
                layout.addWidget(self.label)

                self.label2 = QExtendedGraphicsView.QExtendedGraphicsView()
                self.label2.setMinimumWidth(300)
                self.pixmap2 = QtWidgets.QGraphicsPixmapItem(self.label2.origin)
                layout.addWidget(self.label2)

                self.label_text2 = QtWidgets.QLabel()
                layout.addWidget(self.label_text2)
                self.progress2 = QtWidgets.QProgressBar()
                layout.addWidget(self.progress2)

            with QtShortCuts.QVBoxLayout() as layout:
                with QtShortCuts.QVBoxLayout() as layout2:
                    self.window_size = QtShortCuts.QInputString(layout2, "window size", "50", type=int,
                                                                settings=self.settings,
                                                                settings_key="spheriod/deformation/window_siye")
                    layout2.addWidget(QHLine())
                    with QtShortCuts.QHBoxLayout(None):
                        self.n_min = QtShortCuts.QInputString(None, "n_min", "None", allow_none=True, type=int,
                                                              settings=self.settings,
                                                              settings_key="spheriod/deformation/n_min")
                        self.n_max = QtShortCuts.QInputString(None, "n_max", "None", allow_none=True, type=int,
                                                              settings=self.settings,
                                                              settings_key="spheriod/deformation/n_max")

                    self.thres_segmentation = QtShortCuts.QInputNumber(None, "thres_segmentation", 0.9, float=True,
                                                                       min=0.2, max=1.5, step=0.1,
                                                                       use_slider=True,
                                                                       settings=self.settings,
                                                                       settings_key="spheriod/deformation/thres_segmentation2")
                    self.thres_segmentation.valueChanged.connect(lambda: self.param_changed("thres_segmentation", True))
                    self.continous_segmentation = QtShortCuts.QInputBool(None, "continous_segmentation", False,
                                                                         settings=self.settings,
                                                                         settings_key="spheriod/deformation/continous_segemntation")
                    self.continous_segmentation.valueChanged.connect(lambda: self.param_changed("continous_segmentation", True))
                    self.n_min.valueChanged.connect(lambda: self.param_changed("n_min"))
                    self.n_max.valueChanged.connect(lambda: self.param_changed("n_max"))
                layout.addWidget(QHLine())
                with QtShortCuts.QVBoxLayout(None) as layout:
                    with QtShortCuts.QHBoxLayout(None):
                        self.plot = QtShortCuts.QInputBool(None, "plot", True, settings=self.settings,
                                                           settings_key="spheriod/deformation/plot")
                        # self.draw_mask = QtShortCuts.QInputBool(None, "draw mask", True, settings=self.settings, settings_key="spheriod/deformation/draw_mask")

                    with QtShortCuts.QHBoxLayout(None):
                        self.color_norm = QtShortCuts.QInputString(None, "color norm", 75., type=float,
                                                                   settings=self.settings,
                                                                   settings_key="spheriod/deformation/color_norm")
                        self.cbar_um_scale = QtShortCuts.QInputString(None, "cbar_um_scale", None, allow_none=True,
                                                                      type=int, settings=self.settings,
                                                                      settings_key="spheriod/deformation/cbar_um_scale")
                        self.quiver_scale = QtShortCuts.QInputString(None, "quiver_scale", 1, type=int,
                                                                     settings=self.settings,
                                                                     settings_key="spheriod/deformation/quiver_scale")

                    self.dpi = QtShortCuts.QInputString(None, "dpi", 150, allow_none=True, type=int,
                                                        settings=self.settings, settings_key="spheriod/deformation/dpi")
                    self.dt_min = QtShortCuts.QInputString(None, "dt_min", None, allow_none=True, type=int,
                                                           settings=self.settings,
                                                           settings_key="spheriod/deformation/dt_min")

                layout.addStretch()


                self.button_run = QtShortCuts.QPushButton(None, "run", self.run)
        self.images = []
        self.data = {}

        self.input_list = [
            self.inputText,
            self.outputText,
            self.button_clear,
            self.button_addList,
            self.continous_segmentation,
            self.thres_segmentation,
            self.n_min, self.n_max,
        ]

        self.progress_signal.connect(self.progress_callback)
        self.finished_signal.connect(self.finished)

    def show_files(self):
        import glob
        import re
        text = os.path.normpath(self.inputText.value())
        glob_string = text.replace("+", "*")
        print("globbing", glob_string)
        files = glob.glob(glob_string)

        output_base = glob_string
        while "*" in str(output_base):
            output_base = Path(output_base).parent

        regex_string = re.escape(text).replace("\*", "(.*)").replace("\+", ".*")

        data = {}
        for file in files:
            file = os.path.normpath(file)
            print(file, regex_string)
            match = re.match(regex_string, file).groups()
            reconstructed_file = regex_string
            for element in match:
                reconstructed_file = reconstructed_file.replace("(.*)", element, 1)
            reconstructed_file = reconstructed_file.replace(".*", "*")
            reconstructed_file = re.sub(r'\\(.)', r'\1', reconstructed_file)

            if reconstructed_file not in data:
                output = Path(self.outputText.value()) / os.path.relpath(file, output_base)
                output = output.parent / output.stem
                data[reconstructed_file] = dict(
                    images=[],
                    output=output,
                    thres_segmentation=None,
                    continous_segmentation=None,
                    custom_mask=None,
                    n_min=None,
                    n_max=None,
                )
            data[reconstructed_file]["images"].append(file)
            #if len(data[reconstructed_file]["images"]) > 4:
            #    data[reconstructed_file]["images"] = data[reconstructed_file]["images"][:4]
        data.update(self.data)
        self.data = data
        self.list.clear()
        self.list.addItems(list(data.keys()))

    def clear_files(self):
        self.list.clear()
        self.data = {}

    def listSelected(self):
        if len(self.list.selectedItems()):
            data = self.data[self.list.selectedItems()[0].text()]
            self.images = data["images"]
            self.last_image = None
            self.last_seg = None
            for name in ["thres_segmentation", "continous_segmentation", "n_min", "n_max"]:
                if data[name] is not None:
                    getattr(self, name).setValue(data[name])
            self.slider.setRange(0, len(self.images)-1)
            self.slider_changed(self.slider.value())
            self.label_text2.setText(str(data["output"]))
            self.slider.min = self.n_min.value()
            self.slider.max = self.n_max.value()
            self.slider.update()

    def param_changed(self, name, update_image=False):
        if len(self.list.selectedItems()):
            data = self.data[self.list.selectedItems()[0].text()]
            data[name] = getattr(self, name).value()
            if update_image:
                self.slider_changed(self.slider.value())
            self.slider.min = self.n_min.value()
            self.slider.max = self.n_max.value()
            self.slider.update()

    last_image = None
    last_seg = None
    def slider_changed(self, i):
        if self.last_image is not None and self.last_image[0] == i:
            i, im, im0 = self.last_image
            print("cached")
        else:
            im = imageio.imread(self.images[i]).astype(np.float)
            if self.continous_segmentation.value() is True:
                im0 = im
            else:
                im0 = imageio.imread(self.images[0]).astype(np.float)
            self.last_image = [i, im, im0]

        if self.last_seg is not None and \
                self.last_seg[1] == self.thres_segmentation.value() and \
                self.continous_segmentation.value() is False:
            pass
            print("cached")
        else:
            print(self.last_seg, i, self.thres_segmentation.value())
            seg0 = jf.piv.segment_spheroid(im0, True, self.thres_segmentation.value())
            from skimage import measure
            # Find contours at a constant value of 0.8
            contours = measure.find_contours(seg0["mask"], 0.5)

            path = QtGui.QPainterPath()
            for c in contours:
                path.moveTo(c[0][1], im.shape[0]-c[0][0])
                for cc in c:
                    path.lineTo(cc[1], im.shape[0]-cc[0])
            self.contour.setPath(path)
            self.last_seg = [i, self.thres_segmentation.value()]
        im = im - im.min()
        im = (im/im.max()*255).astype(np.uint8)

        self.pixmap.setPixmap(QtGui.QPixmap(array2qimage(im)))
        self.label.setExtend(im.shape[1], im.shape[0])
        self.label_text.setText(f"{i+1}/{len(self.images)} {self.images[i]}")

        data = self.data[self.list.selectedItems()[0].text()]
        try:
            im = imageio.imread(str(data["output"]) + '/plot' + str(i).zfill(6) + '.png')
        except FileNotFoundError:
            im = np.zeros(im.shape)
        self.pixmap2.setPixmap(QtGui.QPixmap(array2qimage(im)))
        self.label2.setExtend(im.shape[1], im.shape[0])

        self.line_views()

    def line_views(self):

        def changes1(*args):
            self.label2.setOriginScale(self.label.getOriginScale() * self.label.view_rect[0] / self.label2.view_rect[0])
            start_x, start_y, end_x, end_y = self.label.GetExtend()
            center_x, center_y = start_x + (end_x - start_x) / 2, start_y + (end_y - start_y) / 2
            center_x = center_x / self.label.view_rect[0] * self.label2.view_rect[0]
            center_y = center_y / self.label.view_rect[1] * self.label2.view_rect[1]
            self.label2.centerOn(center_x, center_y)

        def zoomEvent(scale, pos):
            changes1()

        self.label.zoomEvent = zoomEvent
        self.label.panEvent = changes1

        def changes2(*args):
            self.label.setOriginScale(self.label2.getOriginScale() * self.label2.view_rect[0] / self.label.view_rect[0])
            start_x, start_y, end_x, end_y = self.label2.GetExtend()
            center_x, center_y = start_x + (end_x - start_x) / 2, start_y + (end_y - start_y) / 2
            center_x = center_x / self.label2.view_rect[0] * self.label.view_rect[0]
            center_y = center_y / self.label2.view_rect[1] * self.label.view_rect[1]
            self.label.centerOn(center_x, center_y)

        def zoomEvent(scale, pos):
            changes2()

        self.label2.zoomEvent = zoomEvent
        self.label2.panEvent = changes2
        changes2()

    def run(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self.run_thread, daemon=True)
            self.thread.start()
            self.setState(True)
        else:
            kill_thread(self.thread)
            self.thread = None
            self.setState(False)

    def setState(self, running):
        if running:
            self.button_run.setText("stop")
            for widget in self.input_list:
                widget.setDisabled(True)
        else:
            self.button_run.setText("run")
            for widget in self.input_list:
                widget.setDisabled(False)

    def finished(self):
        self.thread = None
        self.setState(False)

    def progress_callback(self, i, n, ii, nn):
        self.progress1.setRange(0, n)
        self.progress1.setValue(i)
        self.progress2.setRange(0, nn-1)
        self.progress2.setValue(ii)
        for j in range(self.list.count()):
            if j < i:
                self.list.item(j).setIcon(qta.icon("fa.check", options=[dict(color="darkgreen")]))
            else:
                self.list.item(j).setIcon(qta.icon("fa.circle", options=[dict(color="white")]))
        self.list.setCurrentRow(i)
        self.slider.setEvaluated(ii)
        self.slider.setValue(ii)
        return
        # when plotting show the slider
        if self.plot.value() is True:
            # set the range for the slider
            self.slider.setRange(1, i)
            # it the slider was at the last value, move it to the new maximum
            if self.slider.value() == i-1:
                self.slider.setValue(i)

    def run_thread(self):
        try:
            print("compute displacements")
            n = self.list.count()
            for i in range(n):
                data = self.data[self.list.item(i).text()]
                self.progress_signal.emit(i, n, 0, len(data["images"]))
                folder, file = os.path.split(self.list.item(i).text())

                continous_segmentation = data["continous_segmentation"] or self.continous_segmentation.value()
                thres_segmentation = data["thres_segmentation"] or self.thres_segmentation.value()
                n_min = data["n_min"]
                n_max = data["n_max"]

                jf.piv.compute_displacement_series(str(folder),
                                                   str(file),
                                                   str(data["output"]),
                                                   n_max=n_max,
                                                   n_min=n_min,
                                                   #plot=self.plot.value(),
                                                   draw_mask=False,
                                                   #color_norm=self.color_norm.value(),
                                                   #cbar_um_scale=(self.cbar_um_scale.value()),
                                                   #quiver_scale=(self.quiver_scale.value()),
                                                   #dpi=(self.dpi.value()),
                                                   continous_segmentation=continous_segmentation,
                                                   thres_segmentation=thres_segmentation,
                                                   window_size=(self.window_size.value()),
                                                   #dt_min=(self.dt_min.value()),
                                                   cutoff=None, cmap="turbo",
                                                   callback=lambda ii, nn: self.progress_signal.emit(i, n, ii, nn))
                self.progress_signal.emit(i+1, n, 0, len(data["images"]))
        finally:
            self.finished_signal.emit()


class ListWidget(QtWidgets.QListWidget):
    itemSelectionChanged2 = QtCore.Signal()
    itemChanged2 = QtCore.Signal()
    addItemClicked = QtCore.Signal()

    data = []
    def __init__(self, layout, editable=False, add_item_button=False):
        super().__init__()
        layout.addWidget(self)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.list2_context_menu)
        self.itemChanged.connect(self.list2_checked_changed)
        self.itemChanged = self.itemChanged2
        self.act_delete = QtWidgets.QAction(qta.icon("fa.trash"), "Delete", self)
        self.act_delete.triggered.connect(self.delete_item)

        self.flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable
        if editable:
            self.flags |= QtCore.Qt.ItemIsEditable

        self.add_item_button = add_item_button
        self.addAddItem()
        self.itemSelectionChanged.connect(self.listSelected)
        self.itemSelectionChanged = self.itemSelectionChanged2

        self.itemClicked.connect(self.item_clicked)

    def item_clicked(self, item):
        if item == self.add_item:
            self.addItemClicked.emit()

    def listSelected(self):
        if self.no_list_change is True:
            return
        self.no_list_change = True
        if self.currentItem() == self.add_item:
            self.no_list_change = False
            return
        self.itemSelectionChanged.emit()
        self.no_list_change = False

    add_item = None
    def addAddItem(self):
        if self.add_item_button is False:
            return
        if self.add_item is not None:
            del self.add_item
        self.add_item = QtWidgets.QListWidgetItem(qta.icon("fa.plus"), self.add_item_button, self)
        self.add_item.setFlags(QtCore.Qt.ItemIsEnabled)

    def list2_context_menu(self, position):
        # context menu
        menu = QtWidgets.QMenu()
        menu.addAction(self.act_delete)

        # open menu at mouse click position
        if menu:
            menu.exec_(self.viewport().mapToGlobal(position))

    def delete_item(self):
        index = self.currentRow()
        self.data.pop(index)
        self.takeItem(index)

    def setData(self, data):
        self.no_list_change = True
        self.data = data
        self.clear()
        for d, checked, _ in data:
            self.customAddItem(d, checked)

        self.addAddItem()
        self.no_list_change = False

    no_list_change = False
    def list2_checked_changed(self, item):
        if self.no_list_change is True:
            return
        data = self.data
        for i in range(len(data)):
            item = self.item(i)
            data[i][0] = item.text()
            data[i][1] = item.checkState()
        self.itemChanged.emit()

    def addData(self, d, checked, extra=None):
        self.no_list_change = True
        if self.add_item is not None:
            self.takeItem(self.count()-1)
        self.data.append([d, checked, extra])
        item = self.customAddItem(d, checked)
        self.addAddItem()
        self.no_list_change = False
        return item

    def customAddItem(self, d, checked):
        item = QtWidgets.QListWidgetItem(d, self)
        item.setFlags(self.flags)
        item.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
        return item


class PlottingWindow(QtWidgets.QWidget):
    progress_signal = QtCore.Signal(int, int, int, int)
    finished_signal = QtCore.Signal()
    thread = None

    def __init__(self, parent=None):
        super().__init__(parent)

        # QSettings
        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

        self.setMinimumWidth(800)
        self.setMinimumHeight(400)
        self.setWindowTitle("Saenopy Evaluation")

        self.images = []
        self.data_folders = []
        self.current_plot_func = lambda: None

        with QtShortCuts.QHBoxLayout(self) as main_layout:
            with QtShortCuts.QVBoxLayout() as layout:
                with QtShortCuts.QGroupBox(None, "Groups") as (_, layout2):
                    self.list = ListWidget(layout2, True, add_item_button="add group")
                    self.list.itemSelectionChanged.connect(self.listSelected)
                    self.list.itemChanged.connect(self.replot)
                    self.list.itemChanged.connect(self.update_group_name)
                    self.list.addItemClicked.connect(self.addGroup)

                with QtShortCuts.QGroupBox(layout, "Group") as (self.box_group, layout2):
                    self.list2 = ListWidget(layout2, add_item_button="add files")
                    self.list2.itemSelectionChanged.connect(self.run2)
                    self.list2.itemChanged.connect(self.replot)
                    self.list2.addItemClicked.connect(self.addFiles)

            with QtShortCuts.QGroupBox(main_layout, "Plot Forces") as (_, layout):
                self.type = QtShortCuts.QInputChoice(None, "type", "Pressure", ["Pressure", "Contractility"])
                self.type.valueChanged.connect(self.replot)
                self.dt = QtShortCuts.QInputString(None, "dt", 2, unit="min", type=float)
                self.dt.valueChanged.connect(self.replot)
                self.input_tbar = QtShortCuts.QInputString(None, "Comparison Time", 2, type=float)
                self.input_tbar_unit = QtShortCuts.QInputChoice(self.input_tbar.layout(), None, "min", ["steps", "min", "h"])
                self.input_tbar_unit.valueChanged.connect(self.replot)
                self.input_tbar.valueChanged.connect(self.replot)

                self.canvas = MatplotlibWidget(self)
                layout.addWidget(self.canvas)
                layout.addWidget(NavigationToolbar(self.canvas, self))

                with QtShortCuts.QHBoxLayout() as layout2:
                    self.button_export = QtShortCuts.QPushButton(layout2, "Export", self.export)
                    layout2.addStretch()
                    self.button_run = QtShortCuts.QPushButton(layout2, "Single Time Course", self.run2)
                    self.button_run2 = QtShortCuts.QPushButton(layout2, "Grouped Time Courses", self.plot_groups)
                    self.button_run3 = QtShortCuts.QPushButton(layout2, "Grouped Bar Plot", self.barplot)
                    self.plot_buttons = [self.button_run, self.button_run2, self.button_run3]
                    for button in self.plot_buttons:
                        button.setCheckable(True)

        self.list.setData(self.data_folders)
        self.addGroup()
        self.current_plot_func = self.run2

    def update_group_name(self):
        if self.list.currentItem() is not None:
            self.box_group.setTitle(f"Files for '{self.list.currentItem().text()}'")
            self.box_group.setEnabled(True)
        else:
            self.box_group.setEnabled(False)

    def addGroup(self):
        text = f"Group{1+len(self.data_folders)}"
        item = self.list.addData(text, True, [])
        self.list.setCurrentItem(item)
        self.list.editItem(item)

    def addFiles(self):
        settings = self.settings
        class AddFilesDialog(QtWidgets.QDialog):
            def __init__(self, parent):
                super().__init__(parent)
                self.setWindowTitle("Add Files")
                with QtShortCuts.QVBoxLayout(self) as layout:
                    self.label = QtWidgets.QLabel("Select a path as an input wildcard. Use * to specify a placeholder. All paths that match the wildcard will be added.")
                    layout.addWidget(self.label)
                    self.inputText = QtShortCuts.QInputFilename(None, None, file_type="Result files (result.xlsx)", settings=settings,
                                                                settings_key="batch_eval/wildcard", existing=True, allow_edit=True)
                    with QtShortCuts.QHBoxLayout() as layout3:
                        # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                        layout3.addStretch()
                        self.button_addList = QtShortCuts.QPushButton(None, "cancel", self.reject)
                        self.button_addList = QtShortCuts.QPushButton(None, "ok", self.accept)

        dialog = AddFilesDialog(self)
        if not dialog.exec():
            return

        text = os.path.normpath(dialog.inputText.value())
        files = glob.glob(text)

        current_group = self.list2.data
        current_files = [d[0] for d in current_group]
        for file in files:
            if file in current_files:
                print("File already in list", file)
                continue
            try:
                print("Add file", file)
                res = self.getPandasData(file)
                if self.list2.data is current_group:
                    self.list2.addData(file, True, res)
                    print("replot")
                    self.replot()
                app.processEvents()
            except FileNotFoundError:
                continue

    def getPandasData(self, file):
        res = pd.read_excel(file)
        res["filename"] = file
        res["index"] = res["Unnamed: 0"]
        del res["Unnamed: 0"]
        res["group"] = file
        return res

    def listSelected(self):
        try:
            data = self.data_folders[self.list.currentRow()]
        except IndexError:
            return
        self.update_group_name()
        self.list2.setData(data[2])

    def getAllCurrentPandasData(self):
        results = []
        for name, checked, files in self.data_folders:
            if checked != 0:
                for name2, checked2, res in files:
                    if checked2 != 0:
                        res["group"] = name
                        results.append(res)
        res = pd.concat(results)
        res["t"] = res["index"] * self.dt.value() / 60
        res.to_csv("tmp_pandas.csv")
        return res

    def replot(self):
        if self.current_plot_func is not None:
            self.current_plot_func()

    def get_comparison_index(self):
        if self.input_tbar.value() is None:
            return None
        if self.input_tbar_unit.value() == "steps":
            index = int(np.floor(self.input_tbar.value() + 0.5))
        elif self.input_tbar_unit.value() == "min":
            index = int(np.floor(self.input_tbar.value() / self.dt.value() + 0.5))
        else:
            index = int(np.floor(self.input_tbar.value() * 60 / self.dt.value() + 0.5))
        return index

    def barplot(self):
        for button in self.plot_buttons:
            button.setChecked(False)
        self.button_run3.setChecked(True)
        self.current_plot_func = self.barplot
        plt.cla()
        if self.type.value() == "Contractility":
            mu_name = 'Mean Contractility (µN)'
            y_label = 'Contractility (µN)'
        else:
            mu_name = 'Mean Pressure (Pa)'
            y_label = 'Pressure (Pa)'

        # get all the data as a pandas dataframe
        res = self.getAllCurrentPandasData()

        # limit the dataframe to the comparison time
        index = self.get_comparison_index()
        res = res[res.index == index]

        code_data = [res, ["t", "group", mu_name, "filename"]]

        def plot(res, mu_name, y_label):
            # iterate over the groups
            for name, data in res.groupby("group")[mu_name]:
                # add the bar with the mean value and the standard error as errorbar
                plt.bar(name, data.mean(), yerr=data.sem(), error_kw=dict(capsize=5))
                # add the number of averaged points
                plt.text(name, data.mean() + data.sem(), f"n={data.count()}", ha="center", va="bottom")

            # add ticks and labels
            plt.ylabel(y_label)
            # despine the axes
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
            plt.tight_layout()
            # show the plot
            self.canvas.draw()

        code = execute(plot, code_data[0][code_data[1]], mu_name=mu_name, y_label=y_label)

        self.export_data = [code, code_data]

    def plot_groups(self):
        for button in self.plot_buttons:
            button.setChecked(False)
        self.button_run2.setChecked(True)
        self.current_plot_func = self.plot_groups
        if self.type.value() == "Contractility":
            mu_name = 'Mean Contractility (µN)'
            std_name = 'St.dev. Contractility (µN)'
            y_label = 'Contractility (µN)'
        else:
            mu_name = 'Mean Pressure (Pa)'
            std_name = 'St.dev. Pressure (Pa)'
            y_label = 'Pressure (Pa)'

        plt.cla()
        res = self.getAllCurrentPandasData()

        code_data = [res, ["t", "group", mu_name, "filename"]]

        # add a vertical line where the comparison time is
        if self.input_tbar.value() is not None:
            comp_h = self.get_comparison_index() * self.dt.value() / 60
            plt.axvline(comp_h, color="k")

        def plot(res, mu_name, y_label):
            # iterate over the groups
            for group_name, data in res.groupby("group"):
                # get the mean and sem
                x = data.groupby("t")[mu_name].agg(["mean", "sem", "count"])
                # plot the mean curve
                p, = plt.plot(x.index, x["mean"], lw=2, label=f"{group_name} (n={int(x['count'].mean())})")
                # add a shaded area for the standard error
                plt.fill_between(x.index, x["mean"] - x["sem"], x["mean"] + x["sem"], facecolor=p.get_color(), lw=0,
                                 alpha=0.5)

            # add a grid
            plt.grid(True)
            # add labels
            plt.xlabel('Time (h)')
            plt.ylabel(y_label)
            plt.tight_layout()
            plt.legend()

            # show
            self.canvas.draw()

        code = execute(plot, code_data[0][code_data[1]], mu_name=mu_name, y_label=y_label)

        self.export_data = [code, code_data]
        return

    def run2(self):
        for button in self.plot_buttons:
            button.setChecked(False)
        self.button_run.setChecked(True)
        self.current_plot_func = self.run2
        if self.type.value() == "Contractility":
            mu_name = 'Mean Contractility (µN)'
            std_name = 'St.dev. Contractility (µN)'
            y_label = 'Contractility (µN)'
        else:
            mu_name = 'Mean Pressure (Pa)'
            std_name = 'St.dev. Pressure (Pa)'
            y_label = 'Pressure (Pa)'

        try:
            res = self.data_folders[self.list.currentRow()][2][self.list2.currentRow()][2]
        except IndexError:
            return

        #plt.figure(figsize=(6, 3))
        code_data = [res, ["t", mu_name, std_name]]

        res["t"] = res.index * self.dt.value() / 60

        plt.cla()

        def plot(res, mu_name, std_name, y_label, plot_color):
            mu = res[mu_name]
            std = res[std_name]

            # plot time course of mean values
            p, = plt.plot(res.t, mu, lw=2, color=plot_color)
            # add standard deviation area
            plt.fill_between(res.t, mu - std, mu + std, facecolor=p.get_color(), lw=0, alpha=0.5)

            # add grid
            plt.grid(True)
            # add labels
            plt.xlabel('Time (h)')
            plt.ylabel(y_label)
            plt.tight_layout()

            # show the plot
            self.canvas.draw()

        code = execute(plot, code_data[0][code_data[1]], mu_name=mu_name, std_name=std_name, y_label=y_label, plot_color=f"C{self.list.currentRow()}")

        self.export_data = [code, code_data]

    def export(self):
        settings = self.settings
        class AddFilesDialog(QtWidgets.QDialog):
            def __init__(self, parent):
                super().__init__(parent)
                self.setWindowTitle("Export Plot")
                with QtShortCuts.QVBoxLayout(self) as layout:
                    self.label = QtWidgets.QLabel("Select a path to export the plot script with the data.")
                    layout.addWidget(self.label)
                    self.inputText = QtShortCuts.QInputFilename(None, None, file_type="Python Script (*.py)", settings=settings,
                                                                settings_key="batch_eval/export_plot", existing=False)
                    self.strip_data = QtShortCuts.QInputBool(None, "export only essential data columns", True, settings=settings, settings_key="batch_eval/export_complete_df")
                    self.include_df = QtShortCuts.QInputBool(None, "include dataframe in script", True, settings=settings, settings_key="batch_eval/export_include_df")
                    with QtShortCuts.QHBoxLayout() as layout3:
                        # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                        layout3.addStretch()
                        self.button_addList = QtShortCuts.QPushButton(None, "cancel", self.reject)
                        self.button_addList = QtShortCuts.QPushButton(None, "ok", self.accept)

        dialog = AddFilesDialog(self)
        if not dialog.exec():
            return

        with open(str(dialog.inputText.value()), "wb") as fp:
            code = ""
            code += "import matplotlib.pyplot as plt\n"
            code += "import pandas as pd\n"
            code += "import io\n"
            code += "\n"
            code += "# the data for the plot\n"
            res, columns = self.export_data[1]
            if dialog.strip_data.value() is False:
                columns = None
            if dialog.include_df.value() is True:
                code += "csv_data = r'''" + res.to_csv(columns=columns) + "'''\n"
                code += "# load the data as a DataFrame\n"
                code += "res = pd.read_csv(io.StringIO(csv_data))\n\n"
            else:
                csv_file = str(dialog.inputText.value()).replace(".py", "_data.csv")
                res.to_csv(csv_file, columns=columns)
                code += "# load the data from file\n"
                code += f"res = pd.read_csv('{csv_file}')\n\n"
            code += self.export_data[0]
            fp.write(code.encode("utf8"))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    print(sys.argv)
    window = MainWindow()
    if len(sys.argv) >= 2:
        window.loadFile(sys.argv[1])
    window.show()
    sys.exit(app.exec_())
