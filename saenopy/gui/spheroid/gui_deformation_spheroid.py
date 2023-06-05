import sys

# Setting the Qt bindings for QtPy
import os

import pandas as pd
import qtawesome as qta

os.environ["QT_API"] = "pyqt5"

from qtpy import QtCore, QtWidgets, QtGui

import numpy as np
from natsort import natsorted


from saenopy.gui.common import QtShortCuts, QExtendedGraphicsView
from qimage2ndarray import array2qimage
import matplotlib.pyplot as plt
import imageio
import threading
import glob


from matplotlib.figure import Figure
import jointforces as jf
import urllib
from pathlib import Path

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

settings = QtCore.QSettings("Saenopy", "Saenopy")


class MatplotlibWidget(Canvas):

    def __init__(self, parent=None, width=4, height=3, dpi=100):
        from matplotlib import _pylab_helpers
        plt.ioff()
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.figure.patch.set_facecolor([0, 1, 0, 0])
        self.axes = self.figure.add_subplot(111)

        Canvas.__init__(self, self.figure)
        self.setStyleSheet("background-color:transparent;")
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

    def setActive(self):
        from matplotlib import _pylab_helpers
        self.manager._cidgcf = self.figure
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
            code = code.replace(key, str(value))
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


class QVLine(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.VLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


class Spoiler(QtWidgets.QWidget):
    def __init__(self, parent=None, title='', animationDuration=300):
        """
        References:
            # Adapted from c++ version
            http://stackoverflow.com/questions/32476006/how-to-make-an-expandable-collapsable-section-widget-in-qt
        """
        super(Spoiler, self).__init__(parent=parent)

        self.animationDuration = animationDuration
        self.toggleAnimation = QtCore.QParallelAnimationGroup()
        self.contentArea = QtWidgets.QScrollArea()
        self.headerLine = QtWidgets.QFrame()
        self.toggleButton = QtWidgets.QToolButton()
        self.mainLayout = QtWidgets.QGridLayout()

        toggleButton = self.toggleButton
        toggleButton.setStyleSheet("QToolButton { border: none; }")
        toggleButton.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        toggleButton.setArrowType(QtCore.Qt.RightArrow)
        toggleButton.setText(str(title))
        toggleButton.setCheckable(True)
        toggleButton.setChecked(False)

        headerLine = self.headerLine
        headerLine.setFrameShape(QtWidgets.QFrame.HLine)
        headerLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        headerLine.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)

        self.contentArea.setStyleSheet("QScrollArea { background-color: white; border: none; }")
        self.contentArea.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        # start out collapsed
        self.contentArea.setMaximumHeight(0)
        self.contentArea.setMinimumHeight(0)
        # let the entire widget grow and shrink with its content
        toggleAnimation = self.toggleAnimation
        toggleAnimation.addAnimation(QtCore.QPropertyAnimation(self, b"minimumHeight"))
        toggleAnimation.addAnimation(QtCore.QPropertyAnimation(self, b"maximumHeight"))
        toggleAnimation.addAnimation(QtCore.QPropertyAnimation(self.contentArea, b"maximumHeight"))
        # don't waste space
        mainLayout = self.mainLayout
        mainLayout.setVerticalSpacing(0)
        mainLayout.setContentsMargins(0, 0, 0, 0)
        row = 0
        mainLayout.addWidget(self.toggleButton, row, 0, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(self.headerLine, row, 2, 1, 1)
        row += 1
        mainLayout.addWidget(self.contentArea, row, 0, 1, 3)
        self.setLayout(self.mainLayout)

        def start_animation(checked):
            arrow_type = QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow
            direction = QtCore.QAbstractAnimation.Forward if checked else QtCore.QAbstractAnimation.Backward
            toggleButton.setArrowType(arrow_type)
            self.toggleAnimation.setDirection(direction)
            self.toggleAnimation.start()

        self.toggleButton.clicked.connect(start_animation)

    def setContentLayout(self, contentLayout):
        # Not sure if this is equivalent to self.contentArea.destroy()
        self.contentArea.destroy()
        self.contentArea.setLayout(contentLayout)
        collapsedHeight = self.sizeHint().height() - self.contentArea.maximumHeight()
        contentHeight = contentLayout.sizeHint().height()
        for i in range(self.toggleAnimation.animationCount()-1):
            spoilerAnimation = self.toggleAnimation.animationAt(i)
            spoilerAnimation.setDuration(self.animationDuration)
            spoilerAnimation.setStartValue(collapsedHeight)
            spoilerAnimation.setEndValue(collapsedHeight + contentHeight)
        contentAnimation = self.toggleAnimation.animationAt(self.toggleAnimation.animationCount() - 1)
        contentAnimation.setDuration(self.animationDuration)
        contentAnimation.setStartValue(0)
        contentAnimation.setEndValue(contentHeight)



class CheckAbleGroup(QtWidgets.QWidget, QtShortCuts.EnterableLayout):
    value_changed = QtCore.Signal(bool)
    main_layout = None
    def __init__(self, parent=None, title='', animationDuration=300):
        super().__init__(parent=parent)

        self.headerLine = QtWidgets.QFrame()
        self.checkbox = QtWidgets.QCheckBox()
        self.toggleButton = QtWidgets.QToolButton()
        self.mainLayout = QtWidgets.QGridLayout()

        with QtShortCuts.QVBoxLayout(self) as self.main_layout:
            self.main_layout.setContentsMargins(0, 0, 0, 0)
            with QtShortCuts.QHBoxLayout() as layout:
                layout.setContentsMargins(12, 0, 0, 0)
                self.toggleButton = QtShortCuts.QInputBool(None, "", True)
                self.label = QtWidgets.QPushButton(title).addToLayout()
                self.label.setStyleSheet("QPushButton { border: none; }")
                self.label.clicked.connect(self.toggle)

                headerLine = QtWidgets.QFrame().addToLayout()
                headerLine.setFrameShape(QtWidgets.QFrame.HLine)
                headerLine.setFrameShadow(QtWidgets.QFrame.Sunken)
                headerLine.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
            self.child_widget = QtWidgets.QWidget().addToLayout()
        self.layout = self
        self.value = self.toggleButton.value
        #self.setValue = self.toggleButton.setValue
        self.valueChanged = self.toggleButton.valueChanged
        self.toggleButton.valueChanged.connect(self.changedActive)

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setPen(QtGui.QPen(QtGui.QColor("gray")))
        #p.setBrush(QtGui.QBrush(QtGui.QColor("gray")))
        top = 5
        p.drawRect(0, self.height()-1, self.width(), 0)
        p.drawRect(0, top, 0, self.height())
        p.drawRect(0, top, 7, 0)
        p.drawRect(self.width()-1, top, 0, self.height())
        super().paintEvent(ev)
                    
    def toggle(self):
        self.setValue(not self.value())
        self.changedActive()

    def setValue(self, value):
        self.toggleButton.setValue(value)
        self.changedActive()

    def changedActive(self):
        self.value_changed.emit(self.value())
        self.child_widget.setEnabled(self.value())

    def addLayout(self, layout):
        if self.main_layout is None:
            self.setLayout(layout)
        else:
            self.child_widget.setLayout(layout)
        return layout


class LookUpTable(QtWidgets.QDialog):
    progress_signal = QtCore.Signal(int, int)
    finished_signal = QtCore.Signal()
    thread = None

    def __init__(self):
        super().__init__()
        self.settings = QtCore.QSettings("Saenopy", "Saenopy")

        with QtShortCuts.QHBoxLayout(self):
            with QtShortCuts.QVBoxLayout() as main_layout:
                with QtShortCuts.QGroupBox(None, "Material Parameters") as (self.material_parameters, layout):
                    with QtShortCuts.QHBoxLayout():
                        self.input_k = QtShortCuts.QInputString(None, "k", "1449", type=float)
                        self.input_d_0 = QtShortCuts.QInputString(None, "d_0", "0.00215", type=float)
                    with QtShortCuts.QHBoxLayout():
                        self.input_lamda_s = QtShortCuts.QInputString(None, "lamdba_s", "0.032", type=float)
                        self.input_ds = QtShortCuts.QInputString(None, "ds", "0.055", type=float)

                with QtShortCuts.QGroupBox(None, "Pressure Range") as (self.material_parameters, layout):
                    with QtShortCuts.QHBoxLayout():
                        self.start = QtShortCuts.QInputString(None, "min", "0.1", type=float)
                        self.end = QtShortCuts.QInputString(None, "max", "1000", type=float)
                        self.n = QtShortCuts.QInputString(None, "count", "150", type=int)

                with QtShortCuts.QGroupBox(None, "Iteration Parameters") as (self.material_parameters, layout):
                    with QtShortCuts.QHBoxLayout():
                        self.max_iter = QtShortCuts.QInputString(None, "max_iter", "600", type=int)
                        self.step = QtShortCuts.QInputString(None, "step", "0.0033", type=float)

                with QtShortCuts.QGroupBox(None, "Run Parameters") as (self.material_parameters, layout):
                    with QtShortCuts.QVBoxLayout():
                        self.n_cores = QtShortCuts.QInputNumber(None, "n_cores", 1, float=False)
                        #layout=None, name=None, value=None, dialog_title="Choose File", file_type="All", filename_checker=None, existing=False, **kwargs):
                        self.output = QtShortCuts.QInputFolder(None, "Output Folder")

                main_layout.addStretch()

                with QtShortCuts.QHBoxLayout() as layout2:

                    self.button_run = QtWidgets.QPushButton("run").addToLayout()
                    self.button_run.clicked.connect(self.run)
                    layout2.addStretch()
                    layout2.addWidget(self.button_run)

                self.progressbar = QtWidgets.QProgressBar().addToLayout()

            self.description = QtWidgets.QTextBrowser().addToLayout()
            self.description.setStyleSheet("QTextEdit { background: #f0f0f0}")
            self.description.setText("""
            <h1>Material Simulations</h1><br/>

            To calculate the contractile forces that multicellular aggregates excert on the surrounding
            matrix, we generate material lookup-tables that predict the contractile pressure
            from the size of the matrix deformations as a function of the distance to the spheroid center as 
            described in Mark et al. (2020, <a href="https://elifesciences.org/articles/51912">click here for more details</a>). <br/>
            <br/>

            To generate a material lookup-table, we model the nonlinear fiber material according to the
            given material properties <b>k</b>, <b>d_0</b>, <b>ds</b> and <b>lambda_s</b> 
            (<a href="https://saenopy.readthedocs.io/en/latest/">click here for more details</a>).<br/>
            <i> Default values are taken from a collagen I hydrogel (1.2mg/ml) with k=1449, d_0=0.00215, ds=0.055, lambda_s=0.032.</i><br/>
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
            to be conducted only a single time for a certain material. Additionally, lookuptables for purely linear fiber material with 
            arbitrary Young's modulus can be created without conducting simulations
            in the next step.
            """)
            self.description.setOpenExternalLinks(True)

        self.input_list = [
            self.input_k,
            self.input_d_0,
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

        self.accept()

    def progress_callback(self, i, n):
        self.progressbar.setRange(0, n)
        self.progressbar.setValue(i)

    def run_thread(self):
        out_table = Path(self.output.value())
        out_folder = out_table.parent / out_table.stem

        material = jf.materials.custom(self.input_k.value(),
                                       self.input_d_0.value(),
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

################



class LookupTableGenerator(QtWidgets.QDialog):
    def loadExisting(self):
        last_folder = settings.value("batch", "batch/simulations_path")
        filename = QtWidgets.QFileDialog.getExistingDirectory(None, "Open Simulations Folder", last_folder)
        filename = filename[0] if isinstance(filename, tuple) else str(filename) if filename is not None else None

        if filename is None:
            return

        self.output.setValue(filename)

    def generate(self):
        dialog = LookUpTable()

        if not dialog.exec():
            return

        self.result = dialog.output.value()
        self.label_input.setText(self.result)
        self.path_changed()

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Generate Material Lookup Table")
        with QtShortCuts.QHBoxLayout(self):
            with QtShortCuts.QGroupBox(None, "Generate Material Lookup Table") as (_, layout):
                with QtShortCuts.QHBoxLayout() as layout3:
                    self.button_addList = QtShortCuts.QPushButton(None, "Load existing\nSimulations",
                                                                  self.loadExisting)
                    self.button_addList.setMinimumSize(100, 100)
                    QVLine().addToLayout()
                    self.button_addList = QtShortCuts.QPushButton(None, "Generate New Simulations",
                                                                  self.generate)
                    self.button_addList.setMinimumSize(100, 100)

                self.output = QtShortCuts.QInputString(layout, "Input Folder")
                self.output.setDisabled(True)

                with QtShortCuts.QHBoxLayout(layout) as layout2:
                    self.x0 = QtShortCuts.QInputString(layout2, "x0", "1", type=float)
                    self.x1 = QtShortCuts.QInputString(layout2, "x1", "50", type=float)
                    self.n = QtShortCuts.QInputString(layout2, "n", "100", type=int)

                self.lookup_table = QtShortCuts.QInputFilename(layout, "Output Lookup Table", 'lookup_example.pkl',
                                                               file_type="Pickle Lookup Table (*.pkl)")

                with QtShortCuts.QHBoxLayout(layout) as layout2:
                    layout2.addStretch()
                    self.button_run = QtShortCuts.QPushButton(layout2, "generate", self.run)

            self.description = QtWidgets.QTextBrowser().addToLayout()
            self.description.setStyleSheet("QTextEdit { background: #f0f0f0}")
            self.description.setText("""
            <h1>Material Lookup-Table</h1>

            From the individual simulations we generate our material lookup-table. <br/>
            <br/>

            Therefore, we load the <b>Input folder</b> where the individual simulations are stored.<b>Generate</b> will then create and save the material-lookuptable 
            as a *.pkl-file for further force analysis in the specified <b>Output Lookup Table</b> location.<br/>
            <br/>
            <i>If desired the interpolation of data points from the FE-simulations to different relative distances can be changed.  
            By default a grid with <b>n</b>=150 logarithmically spaced intervals is generated between a
            distance of <b>x0</b>=1 effective spheroid radii to a distance of <b>x1</b>=50 effective spheroid radii away from the center.</i><br/>
            <br/>
            Additionally, lookuptables for purely linear fiber material (assuming poission ratio of v=0.25) with 
            arbitrary Young's modulus can be created by interpolation without conducting furhter simulations. 
            using the XXXX function. <br/>
            <br/>

            The material lookup-table can be visualized by reading in the *.pkl-file as <b>Input Lookup Table</b> and adjust the desired range
            of pressures (from <b>p0</b> to <b>p1</b> Pascal) and distances (from <b>d1</b> to <b>d2</b> in 
            effective spheroid radii). Further, the measured deformation propagation from experiments can be added and
            directly compared to the material lookup-table using XXXX.<br/>
            <br/>




            """.strip())

    def run(self):
        lookup_table = jf.simulation.create_lookup_table_solver(str(self.output.value()),
                                                                x0=self.x0.value(),
                                                                x1=self.x1.value(),
                                                                n=self.n.value())
        get_displacement, get_pressure = jf.simulation.create_lookup_functions(lookup_table)
        jf.simulation.save_lookup_functions(get_displacement, get_pressure, str(self.lookup_table.value()))
        self.accept()

class SelectLookup(QtWidgets.QDialog):
    progress_signal = QtCore.Signal(int, int)

    def loadExisting(self):
        last_folder = settings.value("batch", "batch/lookuptable_path")
        filename = QtWidgets.QFileDialog.getOpenFileName(None, "Open Lookup Table", last_folder, "Pickle Lookup Table (*.pkl)")
        filename = filename[0] if isinstance(filename, tuple) else str(filename) if filename is not None else None

        if filename == "":
            return

        self.result = filename
        self.label_input.setText(filename)
        self.path_changed()
        #self.accept()

    def generateNewLookup(self):
        dialog = LookupTableGenerator(self)

        if not dialog.exec():
            return

        self.result = dialog.lookup_table.value()
        self.label_input.setText(self.result)
        self.path_changed()

    def loadLinear(self):
        class LookupTableDialog(QtWidgets.QDialog):
            def __init__(self, parent):
                super().__init__(parent)
                self.setWindowTitle("Generate a Linear Material Lookup Table")
                with QtShortCuts.QVBoxLayout(self):
                    self.label = QtWidgets.QLabel("Interpolate a pre-calculated lookup table to a new Young's Modulus.<br>Note that the poisson ratio for the fiber material model is always 0.25 for the linear case.").addToLayout()
                    self.label = QtWidgets.QLabel("Select a path where to save the Lookup Table.").addToLayout()
                    self.inputText = QtShortCuts.QInputFilename(None, "Output Lookup Table", 'lookup_example.pkl',
                                                                file_type="Pickle Lookup Table (*.pkl)",
                                                                settings=settings,
                                                                settings_key="batch/lookuptable_path",
                                                                existing=False)
                    # self.inputText.valueChanged.connect(self.path_changed)
                    with QtShortCuts.QHBoxLayout() as layout2:
                        layout2.addStretch()
                        self.youngs = QtShortCuts.QInputString(None, "Young's Modulus", "250", type=float)
                        self.button_run = QtShortCuts.QPushButton(layout2, "generate", self.run_linear)

            def run_linear(self):
                jf.simulation.linear_lookup_interpolator(emodulus=self.youngs.value(),
                                                         output_newtable=str(self.inputText.value()))
                QtWidgets.QMessageBox.information(self, "Lookup complete",
                                                  f"A lookuptable file for a Young's Modulus {self.youngs.value()} has been written to {self.inputText.value()}.")
                self.accept()
        dialog = LookupTableDialog(self)

        if not dialog.exec():
            return

        self.result = dialog.inputText.value()
        self.label_input.setText(self.result)
        self.path_changed()

    def path_changed(self):
        try:
            [w.setDisabled(False) for w in self.to_disabled]
            self.canvas.setActive()
            plt.clf()
            figure = jf.simulation.plot_lookup_table(str(self.result),
                                                     pressure=[float(self.p0.value()), float(self.p1.value())],
                                                     distance=[float(self.d1.value()), float(self.d2.value())],
                                                     figure=self.canvas.figure, show=False)
            self.canvas.draw()
        except Exception as err:
            [w.setDisabled(True) for w in self.to_disabled]
            raise

    result = None
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Lookup Table")
        with QtShortCuts.QVBoxLayout(self) as layout:
            QtWidgets.QLabel("Choose").addToLayout()
            with QtShortCuts.QHBoxLayout() as layout3:
                self.button_addList = QtShortCuts.QPushButton(None, "Load existing\nLookup Table", self.loadExisting)
                self.button_addList.setMinimumSize(100, 100)
                QVLine().addToLayout()
                self.button_addList = QtShortCuts.QPushButton(None, "Generate\nLookup Table", self.generateNewLookup)
                self.button_addList.setMinimumSize(100, 100)
                self.button_addList = QtShortCuts.QPushButton(None, "Generate\nLinear Lookup Table", self.loadLinear)
                self.button_addList.setMinimumSize(100, 100)
            self.label_input = QtWidgets.QLabel("").addToLayout()
            with QtShortCuts.QGroupBox(None, "Lookup Table Preview Plot") as (self.plot_preview, _):
                with QtShortCuts.QVBoxLayout():
                    self.canvas = MatplotlibWidget(self).addToLayout()
                    self.toolbar = NavigationToolbar(self.canvas, self).addToLayout()
                    with QtShortCuts.QHBoxLayout(layout):
                        self.p0 = QtShortCuts.QInputString(None, "p_min", "0", type=float)
                        self.p1 = QtShortCuts.QInputString(None, "p_max", "1000", type=float)
                        self.d1 = QtShortCuts.QInputString(None, "r_min", "2", type=float)
                        self.d2 = QtShortCuts.QInputString(None, "r_max", "50", type=float)
                        self.replot = QtShortCuts.QPushButton(None, "replot", self.path_changed)

            with QtShortCuts.QHBoxLayout() as layout3:
                layout3.addStretch()
                self.button_cancel = QtShortCuts.QPushButton(None, "cancel", self.reject)
                self.button_ok = QtShortCuts.QPushButton(None, "ok", self.accept)

        self.to_disabled = [self.canvas, self.toolbar, self.p0, self.p1, self.d1, self.d2, self.replot, self.button_ok]
        [w.setDisabled(True) for w in self.to_disabled]



class SelectOrGenerateLookupTable(QtWidgets.QWidget):
    def __init__(self):
        pass


class MainWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # QSettings
        self.settings = QtCore.QSettings("Saenopy", "Saenopy")

        self.setMinimumWidth(800)
        self.setMinimumHeight(400)
        self.setWindowTitle("Saenopy Viewer")

        main_layout = QtWidgets.QHBoxLayout(self)

        with QtShortCuts.QTabWidget(main_layout) as self.tabs:

            """ """
            with self.tabs.createTab("Analyse Measurements") as v_layout:
                with QtShortCuts.QHBoxLayout() as h_layout:
                    #self.deformations = Deformation(h_layout, self)
                    self.deformations = BatchEvaluate(self)
                    h_layout.addWidget(self.deformations)
                    self.description = QtWidgets.QTextEdit()
                    self.description.setReadOnly(True)
                    self.description.setStyleSheet("QTextEdit { background: #f0f0f0}")
                    h_layout.addWidget(self.description)
                    self.description.setMaximumWidth(300)
                    self.description.setText("""
                    <h1>Deformation Detection</h1>
                    Now we need to extract the deformations from the recorded image data. <br/>
                    <br/>
                    Therefore we use the 2D OpenPIV (particle image velocimetry) algorithm to determine the movement 
                    of beads that surround the spheroid in the equatorial plane. By exploiting spherical symmetry this deformation 
                    can then simply compared to the 3D simulations by using our material lookup-table. <br/>
                    <br/>
                                        
                    
                     <h1>Force Reconstruction</h1>
                     For all matrix deformation a pressure & force value can be assigned by the relative deformation and
                     the relative distance to the center with regards to the used material lookup-table.The overall force is then 
                     calculated from all matrix deformations that lie in the specified distance between <b>r_min</b> and <b>r_max</b> away
                     to the spheroid center (default is r_min=2 and r_max=None). <br/>
                    <br/>
                    
                     Both steps can be executed individually or joint.<br/>
                    <br/>
                     
                    <h2>Parameters</h2>
                    <ul>
                    <br/>Deformation<br/>
                    <li><b>Raw Images</b>: Path to the folder containing the raw image data.</li>
                    <li><b>Wildcard</b>: Wildcard to selection elements within this folder (Example: Pos01_t*.tif; star will allow all timesteps). </li>
                    <li><b>n_min, n_max</b>: Set a minimum or maximum element if first or last time steps from this selection should be ignored (default is None). </li>
                    <li><b>thres_segmentation</b>: Factor to change the threshold for spheroid segmentation (default is 0.9). </li>
                    <li><b>continous_segmentation</b>: If active, the segmentation is repeated for every timestep individually.
                    By default we use the segmentation of the first time step (less sensitive to fluctuations)  </li>
                    <li><b>thres_segmentation</b>: Factor to change the threshold for spheroid segmentation (default is 0.9).</li>
                    <br/>Force<br/>
                    <li><b>r_min, r_max</b>:Distance range (relativ radii towards center) in which deformations are included for the force caluclation/li>
                    </ul>
                 
                     """.strip())

            """ """
            with self.tabs.createTab("Data Analysis") as v_layout:
                with QtShortCuts.QHBoxLayout() as h_layout:

                    #self.deformations = Force(h_layout, self)
                    self.deformations = PlottingWindow(self)
                    h_layout.addWidget(self.deformations)

                    self.description = QtWidgets.QTextEdit()
                    self.description.setReadOnly(True)
                    self.description.setStyleSheet("QTextEdit { background: #f0f0f0}")
                    h_layout.addWidget(self.description)
                    self.description.setText("""
                            <h1>Data Analysis</h1>
                           
                            In this step we can analyze our results. <br/>
                            <br/>                    
                            
                            For each  <b>individually</b> spheroid/organoid the <b>force</b> or the <b>pressure</b> development 
                            can be visualized over time. In addition, different spheroids or organoids
                            can be merged in <b>groups</b> , for which then the
                            mean value and standard error can be visualized groupwise. <br/>
                            <br/>
                                
                            Different groups can be created and to each group individual experiments 
                            are added by using a path-placeholder combination. In particular,                                            
                            the "*" is used to scan across sub-directories to find different "result.xlsx"-files.<br/>                  
                            <i> Example: "C:/User/ExperimentA/well1/Pos*/result.xlsx" to read 
                            in all positions in certain directory </i><br/>
                            <br/>
                            
                            Finally, we need to specify the time between consecutive images in <b>dt</b>. 
                            To display a bar chart for a particular time point, we select the desired
                            time point using <b>Comparison time</b>.<br/>
                            <br/>
                            
                            Export allows to store the data as CSV file or an embedded table within a python script 
                            allowing to re-plot and adjust the figures later on. <br/>
                            """)

    def load(self):
        files = glob.glob(self.input_filename.value())
        self.input_label.setText("\n".join(files))
#        self.input_filename



class QSlider(QtWidgets.QSlider):
    min = None
    max = None
    evaluated = 0

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        if self.maximum()-self.minimum():
           
            if (self.min is not None) and (self.min != "None"):
                self.drawRect(0, self.min, "gray", 0.2)
            if (self.max is not None) and (self.max != "None"):
                value = self.max
                if self.max < 0:
                    value = self.maximum()+self.max
                self.drawRect(value, self.maximum(), "gray", 0.2)
            if (self.evaluated is not None) and (self.min != "None"):
                self.drawRect(self.min if self.min is not None else 0, self.evaluated, "lightgreen", 0.3)
        super().paintEvent(ev)

    def drawRect(self, start, end, color, border):
        p = QtGui.QPainter(self)
        p.setPen(QtGui.QPen(QtGui.QColor("transparent")))
        p.setBrush(QtGui.QBrush(QtGui.QColor(color)))  
            
        if (self.min is not None) and (end != "None") and (start != "None"):
            s = self.width() * (start - self.minimum()) / (self.maximum() - self.minimum() + 1e-5)
            e = self.width() * (end - self.minimum()) / (self.maximum() - self.minimum() + 1e-5)
            p.drawRect(int(s), int(self.height()*border),
                       int(e-s), int(self.height()*(1-border*2)))

    def setEvaluated(self, value):
        self.evaluated = value
        self.update()


class BatchEvaluate(QtWidgets.QWidget):
    progress_signal = QtCore.Signal(int, int, int, int)
    measurement_evaluated_signal = QtCore.Signal(int, int)
    finished_signal = QtCore.Signal()
    thread = None

    def __init__(self, parent=None):
        super().__init__(parent)

        # QSettings
        self.settings = QtCore.QSettings("Saenopy", "Saenopy")

        self.setMinimumWidth(800)
        self.setMinimumHeight(400)
        self.setWindowTitle("Saenopy Viewer")

        with QtShortCuts.QHBoxLayout(self) as main_layout:
            with QtShortCuts.QSplitter() as lay:
                with QtShortCuts.QVBoxLayout() as layout:
                    layout.setContentsMargins(0, 0, 0, 0)
                    self.list = ListWidget(layout, add_item_button="add measurements")
                    self.list.addItemClicked.connect(self.show_files)
                    self.list.itemSelectionChanged.connect(self.listSelected)
                    self.progress1 = QtWidgets.QProgressBar()
                    layout.addWidget(self.progress1)

                with QtShortCuts.QVBoxLayout() as layout:
                    self.slider = QSlider().addToLayout()
                    self.slider.setRange(0, 0)
                    self.slider.valueChanged.connect(self.slider_changed)
                    self.slider.setOrientation(QtCore.Qt.Horizontal)
                    #layout.addWidget(self.slider)

                    self.label_text = QtWidgets.QLabel()
                    layout.addWidget(self.label_text)

                    self.label = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
                    self.label.setMinimumWidth(300)
                    self.pixmap = QtWidgets.QGraphicsPixmapItem(self.label.origin)
                    self.contour = QtWidgets.QGraphicsPathItem(self.label.origin)
                    pen = QtGui.QPen(QtGui.QColor(255, 0, 0))
                    pen.setCosmetic(True)
                    self.contour.setPen(pen)

                    self.label2 = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
                    #self.label2.setMinimumWidth(300)
                    self.pixmap2 = QtWidgets.QGraphicsPixmapItem(self.label2.origin)

                    self.label_text2 = QtWidgets.QLabel().addToLayout()
                    self.progress2 = QtWidgets.QProgressBar().addToLayout()

                frame = QtWidgets.QFrame().addToLayout()
                frame.setMaximumWidth(300)
                with QtShortCuts.QVBoxLayout(frame) as layout:
                    with CheckAbleGroup(self, "Detect Deformations").addToLayout() as self.deformation_data:
                     with QtShortCuts.QVBoxLayout() as layout2:
                        self.window_size = QtShortCuts.QInputNumber(layout2, "window size", 50,
                                                                    float=False, name_post='px',
                                                                    settings=self.settings,
                                                                    settings_key="spheriod/deformation/window_size")

                        QHLine().addToLayout()
                        with QtShortCuts.QHBoxLayout(None):
                            self.n_min = QtShortCuts.QInputString(None, "n_min", "None", allow_none=True, type=int,
                                                                  settings=self.settings,
                                                                  settings_key="spheriod/deformation/n_min")
                            self.n_max = QtShortCuts.QInputString(None, "n_max", "None", allow_none=True, type=int,
                                                                  settings=self.settings, name_post='frames',
                                                                  settings_key="spheriod/deformation/n_max")

                        self.thres_segmentation = QtShortCuts.QInputNumber(None, "segmentation threshold", 0.9, float=True,
                                                                           min=0.2, max=1.5, step=0.1,
                                                                           use_slider=False,
                                                                           settings=self.settings,
                                                                           settings_key="spheriod/deformation/thres_segmentation2")
                        self.thres_segmentation.valueChanged.connect(lambda: self.param_changed("thres_segmentation", True))
                        self.continous_segmentation = QtShortCuts.QInputBool(None, "continous_segmentation", False,
                                                                             settings=self.settings,
                                                                             settings_key="spheriod/deformation/continous_segemntation")
                        self.continous_segmentation.valueChanged.connect(lambda: self.param_changed("continous_segmentation", True))
                        self.n_min.valueChanged.connect(lambda: self.param_changed("n_min"))
                        self.n_max.valueChanged.connect(lambda: self.param_changed("n_max"))


                        with CheckAbleGroup(self, "individual segmentation").addToLayout() as self.individual_data:
                            with QtShortCuts.QVBoxLayout() as layout2:
                                self.segmention_thres_indi = QtShortCuts.QInputString(None, "segmention threshold", None,
                                                                                      type=float, allow_none=True)
                                self.segmention_thres_indi.valueChanged.connect(self.listSelected)

                        self.individual_data.value_changed.connect(self.changedCheckBox)

                    #QHLine().addToLayout()
                    if 1:
                        with CheckAbleGroup(self, "Plot").addToLayout() as self.plot_data:
                         with QtShortCuts.QVBoxLayout() as layout2:
                            with QtShortCuts.QHBoxLayout() as layout2:
                                self.color_norm = QtShortCuts.QInputString(None, "color norm", 75., type=float,
                                                                           settings=self.settings, name_post='µm',
                                                                           settings_key="spheriod/deformation/color_norm")

                                self.cbar_um_scale = QtShortCuts.QInputString(None, "pixel_size", None, allow_none=True,
                                                                              type=float, settings=self.settings, name_post='µm/px',
                                                                              settings_key="spheriod/deformation/cbar_um_scale")
                            with QtShortCuts.QHBoxLayout() as layout2:
                                self.quiver_scale = QtShortCuts.QInputString(None, "quiver_scale", 1, type=float,
                                                                             settings=self.settings, name_post='a.u.',
                                                                             settings_key="spheriod/deformation/quiver_scale")

                            with QtShortCuts.QHBoxLayout() as layout2:
                                self.dpi = QtShortCuts.QInputString(None, "dpi", 150, allow_none=True, type=int,
                                                                    settings=self.settings, settings_key="spheriod/deformation/dpi")
                                self.dt_min = QtShortCuts.QInputString(None, "dt", None, allow_none=True, type=float,
                                                                       settings=self.settings, name_post='min',
                                                                       settings_key="spheriod/deformation/dt_min")

                    with CheckAbleGroup(self, "Calculate Forces").addToLayout() as self.force_data:
                        with QtShortCuts.QVBoxLayout():
                            with QtShortCuts.QHBoxLayout():
                                self.lookup_table = QtShortCuts.QInputString(None, "Lookup Table")
                                self.lookup_table.line_edit.setDisabled(True)
                                self.button_lookup = QtShortCuts.QPushButton(None, "choose file", self.choose_lookup)
                            #self.output = QtShortCuts.QInputFolder(None, "Result Folder")
                            #self.lookup_table = QtShortCuts.QInputFilename(None, "Lookup Table", 'lookup_example.pkl',
                            #                                               file_type="Pickle Lookup Table (*.pkl)",
                            #                                               existing=True)


                            self.pixel_size = QtShortCuts.QInputString(None, "pixel_size", "1.29", name_post='µm/px', type=float)

                            with QtShortCuts.QHBoxLayout():
                                self.x0 = QtShortCuts.QInputString(None, "r_min", "2", type=float)
                                self.x1 = QtShortCuts.QInputString(None, "r_max", "None", type=float, name_post='spheriod radii', allow_none=True)

                    layout.addStretch()

                    self.button_run = QtShortCuts.QPushButton(None, "run", self.run)
        self.images = []
        self.data = []
        self.list.setData(self.data)

        self.input_list = [
            #self.inputText,
            #self.outputText,
            #self.button_clear,
            #self.button_addList,
            self.force_data,
            self.deformation_data,
            self.plot_data,
        ]

        self.progress_signal.connect(self.progress_callback)
        self.measurement_evaluated_signal.connect(self.measurement_evaluated)
        self.finished_signal.connect(self.finished)


    def changedCheckBox(self):
        for widget in [self.thres_segmentation]:
            widget.setDisabled(self.individual_data.value())
        if not self.individual_data.value():
            for widget in [self.segmention_thres_indi]:
                widget.setValue("None")

    def choose_lookup(self):

        self.lookup_gui = SelectLookup()
        self.lookup_gui.exec()

        if self.lookup_gui.result is not None:
            self.lookup_table.setValue(self.lookup_gui.result)

    def show_files(self):
        settings = self.settings

        class AddFilesDialog(QtWidgets.QDialog):
            def __init__(self, parent):
                super().__init__(parent)
                self.setWindowTitle("Add Files")
                with QtShortCuts.QVBoxLayout(self) as layout:
                    self.label = QtWidgets.QLabel(
                        "Select a path as an input wildcard.<br/>Use a <b>? placeholder</b> to specify <b>different timestamps</b> of a measurement.<br/>Optionally use a <b>* placeholder</b> to specify <b>different measurements</b> to load.")
                    layout.addWidget(self.label)

                    self.inputText = QtShortCuts.QInputFilename(None, "input", file_type="Image (*.tif *.png *.jpg)", settings=settings,
                                                                settings_key="batch/wildcard", existing=True,
                                                                allow_edit=True)
                    self.outputText = QtShortCuts.QInputFolder(None, "output", settings=settings,
                                                               settings_key="batch/wildcard2", allow_edit=True)


                    def changed():
                        import glob, re
                        text = os.path.normpath(self.inputText.value())

                        glob_string = text.replace("?", "*")
                        files =natsorted( glob.glob(glob_string))
                                              
                        regex_string = re.escape(text).replace(r"\*", "(.*)").replace(r"\?", ".*")

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
                                data[reconstructed_file] = 0
                            data[reconstructed_file] += 1

                        counts = [v for v in data.values()]
                        if len(counts):
                            min_count = np.min(counts)
                            max_count = np.max(counts)
                        else:
                            min_count = 0
                            max_count = 0

                        if text == "":
                            self.label2.setText("")
                            self.label2.setStyleSheet("QLabel { color : red; }")
                            self.button_addList1.setDisabled(True)
                        elif max_count == 0:
                            self.label2.setText("No images found.")
                            self.label2.setStyleSheet("QLabel { color : red; }")
                            self.button_addList1.setDisabled(True)
                        elif max_count == 1:
                            self.label2.setText(f"Found {len(counts)} measurements with {max_count} images.<br>Maybe check the wildcard placeholder. Did you forget the : placeholder?")
                            self.label2.setStyleSheet("QLabel { color : orange; }")
                            self.button_addList1.setDisabled(True)
                        elif min_count == max_count:
                            self.label2.setText(
                                f"Found {len(counts)} measurements with each {min_count} images.")
                            self.label2.setStyleSheet("QLabel { color : green; }")
                            self.button_addList1.setDisabled(False)
                        else:
                            self.label2.setText(
                                f"Found {len(counts)} measurements with {counts} images.")
                            self.label2.setStyleSheet("QLabel { color : green; }")
                            self.button_addList1.setDisabled(False)


                    self.inputText.line.textChanged.connect(changed)

                    self.label2 = QtWidgets.QLabel()#.addToLayout()
                    layout.addWidget(self.label2)

                    with QtShortCuts.QHBoxLayout() as layout3:
                        # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                        layout3.addStretch()
                        self.button_addList2 = QtShortCuts.QPushButton(None, "cancel", self.reject)
                        self.button_addList1 = QtShortCuts.QPushButton(None, "ok", self.accept)
                    changed()

        dialog = AddFilesDialog(self)
        if not dialog.exec():
            return

        import glob
        import re
        text = os.path.normpath(dialog.inputText.value())
        glob_string = text.replace("?", "*")
        #print("globbing", glob_string)
        files = natsorted(glob.glob(glob_string))

        output_base = glob_string
        while "*" in str(output_base):
            output_base = Path(output_base).parent

        regex_string = re.escape(text).replace(r"\*", "(.*)").replace(r"\?", ".*")

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
                output = Path(dialog.outputText.value()) / os.path.relpath(file, output_base)
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
        #data.update(self.data)
        #self.data = data
        #self.list.clear()
        #self.list.addItems(list(data.keys()))
        import matplotlib as mpl
        for reconstructed_file, d in data.items():
            self.list.addData(reconstructed_file, True, d, mpl.colors.to_hex(f"gray"))

    def clear_files(self):
        self.list.clear()
        self.data = {}

    last_cell = None
    def listSelected(self):
        if len(self.list.selectedItems()):
            data = self.data[self.list.currentRow()][2]

            attr = data#[3]
            if self.last_cell == self.list.currentRow():
                attr["thres_segmentation"] = self.segmention_thres_indi.value()
                #attr["seg_gaus1"] = self.seg_gaus1_indi.value()
                #attr["seg_gaus2"] = self.seg_gaus2_indi.value()
            else:
                self.segmention_thres_indi.setValue(attr["thres_segmentation"])
                #self.seg_gaus1_indi.setValue(attr["seg_gaus1"])
                #self.seg_gaus2_indi.setValue(attr["seg_gaus2"])
                #print("->", [attr[v] is None for v in ["thres_segmentation"]])
                if np.all([attr[v] is None for v in ["thres_segmentation"]]):
                    self.individual_data.setValue(False)
                else:
                    self.individual_data.setValue(True)
            self.last_cell = self.list.currentRow()
            self.images = data["images"]
            self.last_image = None
            self.last_seg = None
            #for name in ["thres_segmentation", "continous_segmentation", "n_min", "n_max"]:
            #    if data[name] is not None:
            #        getattr(self, name).setValue(data[name])
            self.slider.setRange(0, len(self.images)-1)
            self.slider_changed(self.slider.value())
            self.label_text2.setText(str(data["output"]))
            self.slider.min = self.n_min.value()
            self.slider.max = self.n_max.value()
            self.slider.update()

    def param_changed(self, name, update_image=False):
        if len(self.list.selectedItems()):
            data = self.data[self.list.currentRow()][2]
            data[name] = getattr(self, name).value()
            if update_image:
                self.slider_changed(self.slider.value())
            self.slider.min = self.n_min.value()
            self.slider.max = self.n_max.value()
            self.slider.update()

    last_image = None
    last_seg = None
    def slider_changed(self, i):
        data = self.data[self.list.currentRow()][2]

        thres_segmentation = self.thres_segmentation.value() if data["thres_segmentation"] is None else data["thres_segmentation"]

        if self.last_image is not None and self.last_image[0] == i:
            i, im, im0 = self.last_image
            #print("cached")
        else:
            im = imageio.v2.imread(self.images[i]).astype(float)
            if self.continous_segmentation.value() is True:
                im0 = im
            else:
                im0 = imageio.v2.imread(self.images[0]).astype(float)
            self.last_image = [i, im, im0]

        if self.last_seg is not None and \
                self.last_seg[1] == thres_segmentation and \
                self.continous_segmentation.value() is False:
            pass
            #print("cached")
        else:
            print(self.last_seg, i, thres_segmentation)
            seg0 = jf.piv.segment_spheroid(im0, True, thres_segmentation)
            from skimage import measure
            # Find contours at a constant value of 0.8
            contours = measure.find_contours(seg0["mask"], 0.5)

            path = QtGui.QPainterPath()
            for c in contours:
                path.moveTo(c[0][1], im.shape[0]-c[0][0])
                for cc in c:
                    path.lineTo(cc[1], im.shape[0]-cc[0])
            self.contour.setPath(path)
            self.last_seg = [i, thres_segmentation, seg0]
        im = im - im.min()
        im = (im/im.max()*255).astype(np.uint8)

        self.pixmap.setPixmap(QtGui.QPixmap(array2qimage(im)))
        self.label.setExtend(im.shape[1], im.shape[0])
        self.label_text.setText(f"{i+1}/{len(self.images)} {self.images[i]}")



        #from jointforces.piv import save_displacement_plot
        #import io
        #buf = io.BytesIO()
        #import time
        #t = time.time()
        #dis_sum = np.load(str(data["output"]) + '/def' + str(i).zfill(6) + '.npy', allow_pickle=True).item()
        #print("loadtime", time.time()-t)

        try:
            #im = imageio.v2.imread(buf)
            im = imageio.v2.imread(str(data["output"]) + '/plot' + str(i).zfill(6) + '.png')
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
        if self.lookup_table.value() == '' and self.force_data.value() is True:
            QtWidgets.QMessageBox.critical(self, 'Error - Saenopy',
                                           'No lookup table for force reconstruction specified. Either provide one or disable force calculation.',
                                           QtWidgets.QMessageBox.Ok)
            return
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
                self.list.item(j).setIcon(qta.icon("fa5s.check", options=[dict(color="darkgreen")]))
            else:
                self.list.item(j).setIcon(qta.icon("fa5.circle", options=[dict(color="white")]))
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
            #print("compute displacements")
            n = self.list.count() - 1
            for i in range(n):
                try:
                    if not self.data[i][1]:
                        continue
                    data = self.data[i][2]
                    self.progress_signal.emit(i, n, 0, len(data["images"]))
                    folder, file = os.path.split(self.data[i][0])

                    continous_segmentation = self.continous_segmentation.value()
                    thres_segmentation = data["thres_segmentation"] or self.thres_segmentation.value()
                    
                    # set proper None values if no number set
                    try:
                        n_min = int(self.n_min.value())
                    except:
                        n_min = None
                    try:
                        n_max = int(self.n_max.value())
                    except:
                        n_max = None
                    try:
                        cbar_um_scale = float(self.cbar_um_scale.value())
                    except:
                        cbar_um_scale = None    
                    try:
                        r_max = float(self.x1.value())
                    except:
                        r_max = None    
                    try:
                        r_min = float(self.x0.value())
                    except:
                        r_min = None  
                        
                    if self.deformation_data.value() is True:
                        jf.piv.compute_displacement_series(str(folder),
                                                       str(file),
                                                       str(data["output"]),
                                                       n_max=n_max,
                                                       n_min=n_min,
                                                       plot=self.plot_data.value(),
                                                       #plot=self.plot.value(),
                                                       draw_mask=False,
                                                       color_norm=self.color_norm.value(),
                                                       cbar_um_scale= cbar_um_scale ,
                                                       quiver_scale=self.quiver_scale.value(),
                                                       dpi=(self.dpi.value()),
                                                       continous_segmentation=continous_segmentation,
                                                       thres_segmentation=thres_segmentation,
                                                       window_size=(self.window_size.value()),
                                                       dt_min=(self.dt_min.value()),
                                                       cutoff=None, cmap="turbo",
                                                       callback=lambda ii, nn: self.progress_signal.emit(i, n, ii, nn))

                    elif self.plot_data.value() is True:
                        images = data["images"]
                        for ii in range(0, len(images)):
                            im = imageio.v2.imread(images[i]).astype(float)
                            if ii == 0 or self.continous_segmentation.value() is True:
                                seg0 = jf.piv.segment_spheroid(im, True, self.thres_segmentation.value())
                            if ii > 0:
                                #print("self.dt_min.value()*ii if self.dt_min.value() is not None else None", self.dt_min.value()*ii if self.dt_min.value() is not None else None)
                                from jointforces.piv import save_displacement_plot
                                dis_sum = np.load(str(data["output"]) + '/def' + str(ii).zfill(6) + '.npy', allow_pickle=True).item()
                                save_displacement_plot(str(data["output"]) + '/plot' + str(ii).zfill(6) + '.png', im,
                                                       seg0, dis_sum,
                                                           quiver_scale=(self.quiver_scale.value()),
                                                       color_norm=self.color_norm.value(), cbar_um_scale=(self.cbar_um_scale.value()), dpi=(self.dpi.value()), t=self.dt_min.value()*ii if self.dt_min.value() is not None else None)
                            self.progress_signal.emit(i, n, ii, len(images))

                    if self.force_data.value() is True:
                        jf.force.reconstruct(str(data["output"]),  # PIV output folder
                                             str(self.lookup_table.value()),  # lookup table
                                             self.pixel_size.value(),  # pixel size (µm)
                                             None, r_min=r_min, r_max=r_max)
                except Exception as err:
                    import traceback
                    traceback.print_exc()
                    self.measurement_evaluated_signal.emit(i, -1)
                self.progress_signal.emit(i+1, n, 0, len(data["images"]))
        finally:
            self.finished_signal.emit()

    def measurement_evaluated(self, index, state):
        if state == 1:
            self.list.item(index).setIcon(qta.icon("fa5s.check", options=[dict(color="darkgreen")]))
        elif state == -1:
            self.list.item(index).setIcon(qta.icon("fa5s.times", options=[dict(color="red")]))
        else:
            self.list.item(index).setIcon(qta.icon("fa5.circle", options=[dict(color="white")]))

class ListWidget(QtWidgets.QListWidget):
    itemSelectionChanged2 = QtCore.Signal()
    itemChanged2 = QtCore.Signal()
    addItemClicked = QtCore.Signal()

    data = []
    def __init__(self, layout, editable=False, add_item_button=False, color_picker=False):
        super().__init__()
        layout.addWidget(self)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.list2_context_menu)
        self.itemChanged.connect(self.list2_checked_changed)
        self.itemChanged = self.itemChanged2
        self.act_delete = QtWidgets.QAction(qta.icon("fa5s.trash-alt"), "Delete", self)
        self.act_delete.triggered.connect(self.delete_item)

        self.act_color = None
        if color_picker is True:
            self.act_color = QtWidgets.QAction(qta.icon("fa5s.paint-brush"), "Change Color", self)
            self.act_color.triggered.connect(self.change_color)

        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)

        self.flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable
        if editable:
            self.flags |= QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsDragEnabled

        self.add_item_button = add_item_button
        self.addAddItem()
        self.itemSelectionChanged.connect(self.listSelected)
        self.itemSelectionChanged = self.itemSelectionChanged2

        self.itemClicked.connect(self.item_clicked)
        self.model().rowsMoved.connect(self.rowsMoved)

    def rowsMoved(self, parent, start, end, dest, row):
        if row == self.count():
            if self.add_item is not None:
                self.takeItem(self.count() - 2)
            self.addAddItem()
            return
        if row > start:
            row -= 1
        self.data.insert(row, self.data.pop(start))

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
        self.add_item = QtWidgets.QListWidgetItem(qta.icon("fa5s.plus"), self.add_item_button, self)
        self.add_item.setFlags(QtCore.Qt.ItemIsEnabled)

    def list2_context_menu(self, position):
        if self.currentItem() and self.currentItem() != self.add_item:
            # context menu
            menu = QtWidgets.QMenu()

            if self.act_color is not None:
                menu.addAction(self.act_color)

            menu.addAction(self.act_delete)

            # open menu at mouse click position
            if menu:
                menu.exec_(self.viewport().mapToGlobal(position))

    def change_color(self):
        import matplotlib as mpl
        index = self.currentRow()

        # get new color from color picker
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(*mpl.colors.to_rgb(self.data[index][3])))
        # if a color is set, apply it
        if color.isValid():
            self.data[index][3] = "#%02x%02x%02x" % color.getRgb()[:3]
            self.item(index).setIcon(qta.icon("fa5.circle", options=[dict(color=color)]))

    def delete_item(self):
        index = self.currentRow()
        self.data.pop(index)
        self.takeItem(index)

    def setData(self, data):
        self.no_list_change = True
        self.data = data
        self.clear()
        for d, checked, _, color in data:
            self.customAddItem(d, checked, color)

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

    def addData(self, d, checked, extra=None, color=None):
        self.no_list_change = True
        if self.add_item is not None:
            self.takeItem(self.count()-1)
        self.data.append([d, checked, extra, color])
        item = self.customAddItem(d, checked, color)
        self.addAddItem()
        self.no_list_change = False
        return item

    def customAddItem(self, d, checked, color):
        item = QtWidgets.QListWidgetItem(d, self)
        if color is not None:
            item.setIcon(qta.icon("fa5.circle", options=[dict(color=color)]))
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
        self.settings = QtCore.QSettings("Saenopy", "Saenopy")

        self.setMinimumWidth(800)
        self.setMinimumHeight(400)
        self.setWindowTitle("Saenopy Evaluation")

        self.images = []
        self.data_folders = []
        self.current_plot_func = lambda: None

        with QtShortCuts.QVBoxLayout(self) as main_layout0:
         with QtShortCuts.QHBoxLayout() as main_layout00:
             self.button_save = QtShortCuts.QPushButton(None, "save", self.save)
             self.button_load = QtShortCuts.QPushButton(None, "load", self.load)
             main_layout00.addStretch()
         with QtShortCuts.QHBoxLayout() as main_layout:
            with QtShortCuts.QVBoxLayout() as layout:
                with QtShortCuts.QGroupBox(None, "Groups") as (_, layout2):
                    layout2.setContentsMargins(0, 3, 0, 1)
                    self.list = ListWidget(layout2, True, add_item_button="add group", color_picker=True)
                    self.list.setStyleSheet("QListWidget{border: none}")
                    self.list.itemSelectionChanged.connect(self.listSelected)
                    self.list.itemChanged.connect(self.replot)
                    self.list.itemChanged.connect(self.update_group_name)
                    self.list.addItemClicked.connect(self.addGroup)

                with QtShortCuts.QGroupBox(layout, "Group") as (self.box_group, layout2):
                    layout2.setContentsMargins(0, 3, 0, 1)
                    self.list2 = ListWidget(layout2, add_item_button="add files")
                    self.list2.setStyleSheet("QListWidget{border: none}")
                    self.list2.itemSelectionChanged.connect(self.run2)
                    self.list2.itemChanged.connect(self.replot)
                    self.list2.addItemClicked.connect(self.addFiles)

                    self.setAcceptDrops(True)

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

    def save(self):
        new_path = QtWidgets.QFileDialog.getSaveFileName(None, "Save Session", os.getcwd(), "JSON File (*.json)")
        if new_path:
            if isinstance(new_path, tuple):
                new_path = new_path[0]
            else:
                new_path = str(new_path)
            list_new = []
            for item in self.list.data:
                list_new.append({"name": item[0], "selected": item[1], "color": item[3], "paths": []})
                for item2 in item[2]:
                    list_new[-1]["paths"].append({"path": item2[0], "selected": item[1]})
            import json
            with open(new_path, "w") as fp:
                json.dump(list_new, fp, indent=2)

    def load(self):
        new_path = QtWidgets.QFileDialog.getOpenFileName(None, "Save Session", os.getcwd(), "JSON File (*.json)")
        if new_path:
            if isinstance(new_path, tuple):
                new_path = new_path[0]
            else:
                new_path = str(new_path)
            import json
            with open(new_path, "r") as fp:
                list_new = json.load(fp)
            self.list.clear()
            self.list.setData([[i["name"], i["selected"], [], i["color"]] for i in list_new])
            self.data_folders = self.list.data

            for i, d in enumerate(list_new):
                self.list.setCurrentRow(i)
                self.list.listSelected()
                self.listSelected()
                self.list2.data = self.list.data[i][2]
                self.add_files([d_0["path"] for d_0 in d["paths"]])

                for ii, d_0 in enumerate(d["paths"]):
                    self.list2.data[ii][1] = d_0["selected"]

    def update_group_name(self):
        if self.list.currentItem() is not None:
            self.box_group.setTitle(f"Files for '{self.list.currentItem().text()}'")
            self.box_group.setEnabled(True)
        else:
            self.box_group.setEnabled(False)

    def addGroup(self):
        import matplotlib as mpl
        text = f"Group{1+len(self.data_folders)}"
        item = self.list.addData(text, True, [], mpl.colors.to_hex(f"C{len(self.data_folders)}"))
        self.list.setCurrentItem(item)
        self.list.editItem(item)


    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        # accept url lists (files by drag and drop)
        for url in event.mimeData().urls():
            # if str(url.toString()).strip().endswith(".npz"):
            event.accept()
            return
        event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        event.acceptProposedAction()

    def dropEvent(self, event: QtCore.QEvent):
        urls = []
        for url in event.mimeData().urls():
            print(url)
            url = url.toLocalFile()
            if url[0] == "/" and url[2] == ":":
                url = url[1:]
            print(url)
            if url.endswith("result.xlsx"):
                urls += [url]
            else:
                urls += glob.glob(url + "/**/result.xlsx", recursive=True)
        self.add_files(urls)


    def addFiles(self):
        settings = self.settings
        class AddFilesDialog(QtWidgets.QDialog):
            def __init__(self, parent):
                super().__init__(parent)
                self.setWindowTitle("Add Files")
                with QtShortCuts.QVBoxLayout(self) as layout:
                    self.label = QtWidgets.QLabel("Select a path as an input wildcard. Use * to specify a placeholder. All paths that match the wildcard will be added.")
                    layout.addWidget(self.label)
                    def checker(filename):
                        return filename + "/**/result.xlsx"
                    self.inputText = QtShortCuts.QInputFolder(None, None, settings=settings, filename_checker=checker,
                                                                settings_key="batch_eval/wildcard", allow_edit=True)
                    with QtShortCuts.QHBoxLayout() as layout3:
                        # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                        layout3.addStretch()
                        self.button_addList = QtShortCuts.QPushButton(None, "cancel", self.reject)
                        self.button_addList = QtShortCuts.QPushButton(None, "ok", self.accept)

        dialog = AddFilesDialog(self)
        if not dialog.exec():
            return

        text = os.path.normpath(dialog.inputText.value())
        print(text)
        files = glob.glob(text, recursive=True)
        print(files)

        self.add_files(files)

    def add_files(self, files):
        current_group = self.list2.data
        current_files = [d[0] for d in current_group]
        for file in files:
            if file in current_files:
                #print("File already in list", file)
                continue
            try:
                #print("Add file", file)
                res = self.getPandasData(file)
                if self.list2.data is current_group:
                    self.list2.addData(file, True, res)
                    #print("replot")
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
        for name, checked, files, color in self.data_folders:
            if checked != 0:
                for name2, checked2, res, color in files:
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
        self.canvas.setActive()
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

        color_dict = {d[0]: d[3] for d in self.data_folders}

        def plot(res, mu_name, y_label, color_dict2):
            # define the colors
            color_dict = color_dict2

            # iterate over the groups
            for name, data in res.groupby("group", sort=False)[mu_name]:
                # add the bar with the mean value and the standard error as errorbar
                plt.bar(name, data.mean(), yerr=data.sem(), error_kw=dict(capsize=5), color=color_dict[name])
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

        code = execute(plot, code_data[0][code_data[1]], mu_name=mu_name, y_label=y_label, color_dict2=color_dict)

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

        self.canvas.setActive()
        plt.cla()
        res = self.getAllCurrentPandasData()

        code_data = [res, ["t", "group", mu_name, "filename"]]

        # add a vertical line where the comparison time is
        if self.input_tbar.value() is not None:
            comp_h = self.get_comparison_index() * self.dt.value() / 60
            plt.axvline(comp_h, color="k")

        color_dict = {d[0]: d[3] for d in self.data_folders}

        def plot(res, mu_name, y_label, color_dict2):
            # define the colors
            color_dict = color_dict2

            # iterate over the groups
            for group_name, data in res.groupby("group", sort=False):
                # get the mean and sem
                x = data.groupby("t")[mu_name].agg(["mean", "sem", "count"])
                # plot the mean curve
                p, = plt.plot(x.index, x["mean"], color=color_dict[group_name], lw=2, label=f"{group_name} (n={int(x['count'].mean())})")
                # add a shaded area for the standard error
                plt.fill_between(x.index, x["mean"] - x["sem"], x["mean"] + x["sem"], facecolor=p.get_color(), lw=0, alpha=0.5)

            # add a grid
            plt.grid(True)
            # add labels
            plt.xlabel('Time (h)')
            plt.ylabel(y_label)
            plt.tight_layout()
            plt.legend()

            # show
            self.canvas.draw()

        code = execute(plot, code_data[0][code_data[1]], mu_name=mu_name, y_label=y_label, color_dict2=color_dict)

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

        self.canvas.setActive()
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

        code = execute(plot, code_data[0][code_data[1]], mu_name=mu_name, std_name=std_name, y_label=y_label, plot_color=self.data_folders[self.list.currentRow()][3])

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
