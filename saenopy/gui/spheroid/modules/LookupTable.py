from qtpy import QtCore, QtWidgets, QtGui

from saenopy.gui.common import QtShortCuts
import matplotlib.pyplot as plt
import threading

import jointforces as jf
import urllib
from pathlib import Path

from saenopy.gui.common.gui_classes import QVLine
from .MatplotlibWidget import MatplotlibWidget, NavigationToolbar
from .helper import kill_thread

settings = QtCore.QSettings("Saenopy", "Saenopy")


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

                self.lookup_table = QtShortCuts.QInputFilename(layout, "Output Lookup Table", 'lookup_example.npy',
                                                               file_type="Numpy File (*.npy)")

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
        filename = str(self.output.value())
        if not filename.endswith(".npy"):
            filename = filename + ".npy"
        lookup_table = jf.simulation.create_lookup_table_solver(filename,
                                                                x0=self.x0.value(),
                                                                x1=self.x1.value(),
                                                                n=self.n.value())
        jf.simulation.save_lookup_table(lookup_table, filename)
        #get_displacement, get_pressure = jf.simulation.create_lookup_functions(lookup_table)
        #jf.simulation.save_lookup_functions(get_displacement, get_pressure, str(self.lookup_table.value()))
        self.accept()

class SelectLookup(QtWidgets.QDialog):
    progress_signal = QtCore.Signal(int, int)

    def loadExisting(self):
        last_folder = settings.value("batch", "batch/lookuptable_path")
        filename = QtWidgets.QFileDialog.getOpenFileName(None, "Open Lookup Table", last_folder, "Numpy Lookup Table (*.npy);; Pickle Lookup Table (*.pkl)")
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
                    self.inputText = QtShortCuts.QInputFilename(None, "Output Lookup Table", 'lookup_example.npy',
                                                                file_type="Numpy Lookup Table (*.npy)",
                                                                settings=settings,
                                                                settings_key="batch/lookuptable_path",
                                                                existing=False)
                    # self.inputText.valueChanged.connect(self.path_changed)
                    with QtShortCuts.QHBoxLayout() as layout2:
                        layout2.addStretch()
                        self.youngs = QtShortCuts.QInputString(None, "Young's Modulus", "250", type=float)
                        self.button_run = QtShortCuts.QPushButton(layout2, "generate", self.run_linear)

            def run_linear(self):
                filename = str(self.inputText.value())
                if not filename.endswith(".npy"):
                    filename = filename + ".npy"
                    self.inputText.setValue(filename)
                jf.simulation.linear_lookup_interpolator(emodulus=self.youngs.value(),
                                                         output_newtable=filename)
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
