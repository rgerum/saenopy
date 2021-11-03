import sys

# Setting the Qt bindings for QtPy
import os
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
from saenopy.gui.gui_classes import Spoiler, CheckAbleGroup, QHLine, QVLine, MatplotlibWidget, NavigationToolbar, execute, kill_thread, ListWidget
import imageio
from qimage2ndarray import array2qimage
import matplotlib.pyplot as plt
import glob
import imageio
import threading
import saenopy.getDeformations
import saenopy.multigridHelper
import saenopy.materials
from saenopy.gui.stack_selector import StackSelector
from saenopy.getDeformations import getStack, Stack
import matplotlib as mpl
from saenopy.solver import Solver
from pathlib import Path
import re
from saenopy.loadHelpers import Saveable

""" some magic to prevent PyQt5 from swallowing exceptions """
# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook
# Set the exception hook to our wrapping function
sys.excepthook = lambda *args: sys._excepthook(*args)



def showVectorField(plotter, obj, name, show_nan=True, show_all_points=False, factor=5):
    try:
        field = getattr(obj, name)
    except AttributeError:
        field = obj.getNodeVar(name)
    nan_values = np.isnan(field[:, 0])

    point_cloud = pv.PolyData(obj.R)
    point_cloud.point_arrays[name] = field
    point_cloud.point_arrays[name + "_mag"] = np.linalg.norm(field, axis=1)
    point_cloud.point_arrays[name + "_mag2"] = point_cloud.point_arrays[name + "_mag"].copy()
    point_cloud.point_arrays[name + "_mag2"][nan_values] = 0
    if show_all_points:
        plotter.add_mesh(point_cloud, colormap="turbo", scalars=name + "_mag2")
    elif show_nan:
        R = obj.R[nan_values]
        if R.shape[0]:
            point_cloud2 = pv.PolyData(R)
            point_cloud2.point_arrays["nan"] = obj.R[nan_values, 0] * np.nan
            plotter.add_mesh(point_cloud2, colormap="turbo", scalars="nan", show_scalar_bar=False)

    # generate the arrows
    arrows = point_cloud.glyph(orient=name, scale=name + "_mag2", factor=factor)
    plotter.add_mesh(arrows, colormap="turbo", name="arrows")

    plotter.update_scalar_bar_range([0, np.nanpercentile(point_cloud[name + "_mag2"], 99.9)])

    plotter.show_grid()
    plotter.show()


class Result(Saveable):
    __save_parameters__ = ['output', 'stack_deformed', 'stack_relaxed', 'piv_parameter', 'mesh_piv',
                           'interpolate_parameter', 'solve_parameter', 'solver']
    output: str = None

    stack_deformed: Stack = None
    stack_relaxed: Stack = None

    piv_parameter: dict = None
    mesh_piv: saenopy.solver.Mesh = None

    interpolate_parameter: dict = None
    solve_parameter: dict = None
    solver: saenopy.solver.Solver = None

    def __init__(self, output, stack_deformed, stack_relaxed, **kwargs):
        self.output = output

        self.stack_deformed = stack_deformed
        self.stack_relaxed = stack_relaxed

        super().__init__(**kwargs)

    def __str__(self):
        return f"""
from saenopy.getDeformations import Stack
stack_deformed = Stack({self.deformed})
stack_relaxed = Stack({self.relaxed})
"""
    def save(self):
        super().save(self.output)


def double_glob(text):
    glob_string = text.replace("?", "*")
    print("globbing", glob_string)
    files = glob.glob(glob_string)

    output_base = glob_string
    while "*" in str(output_base):
        output_base = Path(output_base).parent

    regex_string = re.escape(text).replace("\*", "(.*)").replace("\?", ".*")

    results = []
    for file in files:
        file = os.path.normpath(file)
        print(file, regex_string)
        match = re.match(regex_string, file).groups()
        reconstructed_file = regex_string
        for element in match:
            reconstructed_file = reconstructed_file.replace("(.*)", element, 1)
        reconstructed_file = reconstructed_file.replace(".*", "*")
        reconstructed_file = re.sub(r'\\(.)', r'\1', reconstructed_file)
        if reconstructed_file not in results:
            results.append(reconstructed_file)
    return results, output_base


class QProgressBar(QtWidgets.QProgressBar):
    signal_start = QtCore.Signal(int)
    signal_progress = QtCore.Signal(int)

    def __init__(self, layout):
        super().__init__()
        self.setOrientation(QtCore.Qt.Horizontal)
        layout.addWidget(self)
        self.signal_start.connect(lambda i: self.setRange(0, i))
        self.signal_progress.connect(lambda i: self.setValue(i))

    def iterator(self, iter):
        print("iterator", iter)
        self.signal_start.emit(len(iter))
        for i, v in enumerate(iter):
            yield i
            print("emit", i)
            self.signal_progress.emit(i+1)


class PipelineModule(QtWidgets.QWidget):
    processing_finished = QtCore.Signal()
    processing_error = QtCore.Signal(str)
    result: Result = None
    tab: QtWidgets.QTabWidget = None

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__()
        if layout is not None:
            layout.addWidget(self)
        self.parent = parent
        self.settings = self.parent.settings

        self.processing_finished.connect(self.finished_process)
        self.processing_error.connect(self.errored_process)

        self.parent.result_changed.connect(self.resultChanged)
        self.parent.set_current_result.connect(self.setResult)

    def setParameterMapping(self, params_name: str, parameter_dict: dict):
        self.params_name = params_name
        self.parameter_dict = parameter_dict
        for name, widget in self.parameter_dict.items():
            widget.valueChanged.connect(lambda x, name=name: self.setParameter(name, x))

        self.setResult(None)

    def check_available(self, result: Result) -> bool:
        return False

    def resultChanged(self, result: Result):
        """ called when the contents of result changed. Only update view if its the currently displayed one. """
        if result is self.result:
            self.setResult(result)

    def setResult(self, result: Result):
        """ set a new active result object """
        self.result = result
        # check if the results instance can be evaluated currently with this module
        if self.check_available(result) is False:
            # if not disable all the widgets
            for name, widget in self.parameter_dict.items():
                widget.setDisabled(True)
        else:
            # if the results instance does not have the parameter dictionary yet, create it
            if getattr(result, self.params_name) is None:
                setattr(result, self.params_name, {})
            # iterate over the parameters
            for name, widget in self.parameter_dict.items():
                # enable them
                widget.setDisabled(False)
                # set the widgets to the value if the value exits
                params = getattr(result, self.params_name)
                if name in params:
                    widget.setValue(params[name])
                else:
                    params[name] = widget.value()
            for name in list(params.keys()):
                if name not in self.parameter_dict:
                    del params[name]
            self.valueChanged()
        self.update_display()

    def update_display(self):
        pass

    def setParameter(self, name: str, value):
        if self.result is not None:
            getattr(self.result, self.params_name)[name] = value

    def valueChanged(self):
        pass

    def start_process(self):
        self.input_button.setEnabled(False)
        self.progressbar.setRange(0, 0)
        self.thread = threading.Thread(target=self.process_thread, args=(self.result,))
        self.thread.start()

    def process_thread(self, result: Result):
        params = getattr(result, self.params_name)
        try:
            self.process(result, params)
            result.save()
            self.parent.result_changed.emit(result)
            self.processing_finished.emit()
        except Exception as err:
            import traceback
            traceback.print_exc()
            self.processing_error.emit(str(err))

    def process(self, result: Result, params: dict):
        pass

    def finished_process(self):
        self.input_button.setEnabled(True)
        self.progressbar.setRange(0, 1)

    def errored_process(self, text: str):
        QtWidgets.QMessageBox.critical(self, "Deformation Detector", text)
        self.input_button.setEnabled(True)
        self.progressbar.setRange(0, 1)


class StackDisplay(PipelineModule):
    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with self.parent.tabs.createTab("Stacks") as self.tab:
            with QtShortCuts.QHBoxLayout() as layout:
                with QtShortCuts.QVBoxLayout() as layout:
                    with QtShortCuts.QHBoxLayout() as layout:
                        self.label1 = QtWidgets.QLabel("deformed").addToLayout()
                        layout.addStretch()
                        self.button = QtWidgets.QPushButton(qta.icon("fa5s.undo"), "").addToLayout()
                        self.button.setToolTip("reset view")
                        self.button.clicked.connect(lambda x: (self.view1.fitInView(), self.view2.fitInView()))
                    self.view1 = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
                    self.view1.setMinimumWidth(300)
                    self.pixmap1 = QtWidgets.QGraphicsPixmapItem(self.view1.origin)

                    self.label2 = QtWidgets.QLabel("relaxed").addToLayout()
                    self.view2 = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
                    # self.label2.setMinimumWidth(300)
                    self.pixmap2 = QtWidgets.QGraphicsPixmapItem(self.view2.origin)
                self.z_slider = QtWidgets.QSlider(QtCore.Qt.Vertical).addToLayout()
                self.z_slider.valueChanged.connect(self.z_slider_value_changed)
                self.z_slider.setToolTip("set z position")

        self.view1.link(self.view2)

        self.setParameterMapping(None, {})

    def update_display(self):
        if self.result is not None:
            self.parent.tabs.setTabEnabled(0, True)
            self.view1.setToolTip(f"deformed stack\n{self.result.stack_deformed.description()}")
            self.view2.setToolTip(f"relaxed stack\n{self.result.stack_relaxed.description()}")
            self.z_slider.setRange(0, self.result.stack_deformed.shape[2] - 1)
            self.z_slider.setValue(self.result.stack_deformed.shape[2] // 2)
        else:
            self.parent.tabs.setTabEnabled(0, False)

    def z_slider_value_changed(self):
        if self.result is not None:
            im = self.result.stack_deformed[:, :, self.z_slider.value()]
            self.pixmap1.setPixmap(QtGui.QPixmap(array2qimage(im)))
            self.view1.setExtend(im.shape[1], im.shape[0])

            im = self.result.stack_relaxed[:, :, self.z_slider.value()]
            self.pixmap2.setPixmap(QtGui.QPixmap(array2qimage(im)))
            self.view2.setExtend(im.shape[1], im.shape[0])
            self.z_slider.setToolTip(f"set z position\ncurrent position {self.z_slider.value()}")


class DeformationDetector(PipelineModule):

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with QtShortCuts.QVBoxLayout(self) as layout:
            with QtShortCuts.QHBoxLayout():
                self.input_overlap = QtShortCuts.QInputNumber(None, "overlap", 0.6, step=0.1, value_changed=self.valueChanged)
                self.input_win = QtShortCuts.QInputNumber(None, "window size", 30, value_changed=self.valueChanged, unit="μm")
            with QtShortCuts.QHBoxLayout():
                self.input_signoise = QtShortCuts.QInputNumber(None, "signoise", 1.3, step=0.1)
                self.input_driftcorrection = QtShortCuts.QInputBool(None, "driftcorrection", True)
            self.label = QtWidgets.QLabel().addToLayout()
            self.input_button = QtShortCuts.QPushButton(None, "detect deformations", self.start_process)

            self.progressbar = QProgressBar(layout).addToLayout()

        with self.parent.tabs.createTab("Deformations") as self.tab:
            with QtShortCuts.QHBoxLayout() as layout:
                self.plotter = QtInteractor(self)
                self.plotter.set_background("black")
                layout.addWidget(self.plotter.interactor)

        self.setParameterMapping("piv_parameter", {
            "win_um": self.input_win,
            "fac_overlap": self.input_overlap,
            "signoise_filter": self.input_signoise,
            "drift_correction": self.input_driftcorrection,
        })

    def check_available(self, result: Result):
        return result is not None and result.stack_deformed is not None and result.stack_relaxed is not None

    def update_display(self):
        if self.result is not None and self.result.mesh_piv is not None:
            self.parent.tabs.setTabEnabled(1, True)
            showVectorField(self.plotter, self.result.mesh_piv, "U_measured")
        else:
            self.parent.tabs.setTabEnabled(1, False)

    def valueChanged(self):
        if self.check_available(self.result):
            voxel_size1 = self.result.stack_deformed.voxel_size
            voxel_size2 = self.result.stack_relaxed.voxel_size
            stack_deformed = self.result.stack_deformed
            stack_relaxed = self.result.stack_relaxed

            unit_size = (1-self.input_overlap.value())*self.input_win.value()
            stack_size = np.array(stack_deformed.shape)*voxel_size1 - self.input_win.value()
            self.label.setText(f"Deformation grid with {unit_size:.1f}μm elements. Total region is {stack_size}.")
        else:
            self.label.setText("")

    def process(self, result: Result, params: dict):
        import tqdm
        t = tqdm.tqdm
        n = tqdm.tqdm.__new__
        tqdm.tqdm.__new__ = lambda cls, iter: self.progressbar.iterator(iter)
        result.mesh_piv = saenopy.getDeformations.getDisplacementsFromStacks2(result.stack_deformed, result.stack_relaxed,
                                   params["win_um"], params["fac_overlap"], params["signoise_filter"],
                                   params["drift_correction"])



class MeshCreator(PipelineModule):
    mesh_size = [200, 200, 200]

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with QtShortCuts.QVBoxLayout(self) as layout:
            with QtShortCuts.QHBoxLayout():
                self.input_element_size = QtShortCuts.QInputNumber(None, "element_size", 7, unit="μm")
            with QtShortCuts.QHBoxLayout() as layout2:
                self.input_inner_region = QtShortCuts.QInputNumber(None, "inner_region", 100, unit="μm")
                self.input_thinning_factor = QtShortCuts.QInputNumber(None, "thinning factor", 0.2, step=0.1)
            with QtShortCuts.QHBoxLayout() as layout2:

                self.input_mesh_size_same = QtShortCuts.QInputBool(None, "same as stack", True, value_changed=self.valueChanged)
                self.input_mesh_size_x = QtShortCuts.QInputNumber(None, "mesh size x", 200, step=1)
                self.input_mesh_size_y = QtShortCuts.QInputNumber(None, "y", 200, step=1)
                self.input_mesh_size_z = QtShortCuts.QInputNumber(None, "z", 200, step=1)
                self.input_mesh_size_label = QtWidgets.QLabel("μm").addToLayout()
                self.valueChanged()

            self.input_button = QtWidgets.QPushButton("interpolate mesh").addToLayout()
            self.input_button.clicked.connect(self.start_process)

            self.progressbar = QProgressBar(layout).addToLayout()

        with self.parent.tabs.createTab("Mesh") as self.tab:
            with QtShortCuts.QHBoxLayout() as layout:
                self.plotter = QtInteractor(self)
                self.plotter.set_background("black")
                layout.addWidget(self.plotter.interactor)

        self.setParameterMapping("interpolate_parameter", {
            "element_size": self.input_element_size,
            "inner_region": self.input_inner_region,
            "thinning_factor": self.input_thinning_factor,
            "mesh_size_same": self.input_mesh_size_same,
            "mesh_size_x": self.input_mesh_size_x,
            "mesh_size_y": self.input_mesh_size_y,
            "mesh_size_z": self.input_mesh_size_z,
        })

    def valueChanged(self):
        self.input_mesh_size_x.setDisabled(self.input_mesh_size_same.value())
        self.input_mesh_size_y.setDisabled(self.input_mesh_size_same.value())
        self.input_mesh_size_z.setDisabled(self.input_mesh_size_same.value())
        self.deformation_detector_mesh_size_changed()

    def deformation_detector_mesh_size_changed(self):
        if self.input_mesh_size_same.value():
            if self.result is not None and self.result.mesh_piv is not None:
                x, y, z = (self.result.mesh_piv.R.max(axis=0) - self.result.mesh_piv.R.min(axis=0))*1e6
                self.input_mesh_size_x.setValue(x)
                self.setParameter("mesh_size_x", x)
                self.input_mesh_size_y.setValue(y)
                self.setParameter("mesh_size_y", y)
                self.input_mesh_size_z.setValue(z)
                self.setParameter("mesh_size_z", z)

    def check_available(self, result: Result):
        return result is not None and result.mesh_piv is not None

    def update_display(self):
        if self.result is not None and self.result.mesh_piv is not None:
            self.parent.tabs.setTabEnabled(2, True)
            showVectorField(self.plotter, self.result.solver, "U_target", factor=5)
        else:
            self.parent.tabs.setTabEnabled(2, False)

    def process(self, result: Result, params: dict):
        M = result.mesh_piv
        points, cells = saenopy.multigridHelper.getScaledMesh(params["element_size"]*1e-6,
                                      params["inner_region"]*1e-6,
                                      np.array([params["mesh_size_x"], params["mesh_size_y"],
                                                 params["mesh_size_z"]])*1e-6 / 2,
                                      [0, 0, 0], params["thinning_factor"])
        print(np.max(points, axis=0)*1e6, np.min(points, axis=0)*1e6)
        print(np.max(points, axis=0)*1e6 - np.min(points, axis=0)*1e6)
        print(np.max(M.R, axis=0) * 1e6, np.min(M.R, axis=0) * 1e6)
        print(np.max(M.R, axis=0)*1e6 - np.min(M.R, axis=0)*1e6)

        R = (M.R - np.min(M.R, axis=0)) - (np.max(M.R, axis=0) - np.min(M.R, axis=0)) / 2
        U_target = saenopy.getDeformations.interpolate_different_mesh(R, M.getNodeVar("U_measured"), points)
        print(np.min(U_target), np.max(U_target), np.mean(np.isnan(U_target)))
        print(np.min(M.getNodeVar("U_measured")), np.max(M.getNodeVar("U_measured")), np.mean(np.isnan(M.getNodeVar("U_measured"))))
        from saenopy.multigridHelper import getScaledMesh, getNodesWithOneFace

        border_idx = getNodesWithOneFace(cells)
        inside_mask = np.ones(points.shape[0], dtype=bool)
        inside_mask[border_idx] = False

        M = saenopy.Solver()
        M.setNodes(points)
        M.setTetrahedra(cells)
        M.setTargetDisplacements(U_target, inside_mask)

        result.solver = M


class Regularizer(PipelineModule):
    iteration_finished = QtCore.Signal(object)

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with QtShortCuts.QVBoxLayout(self) as main_layout:
            with QtShortCuts.QGroupBox(None, "Material Parameters") as self.material_parameters:
                with QtShortCuts.QVBoxLayout() as layout:
                    with QtShortCuts.QHBoxLayout() as layout2:
                        self.input_k = QtShortCuts.QInputString(None, "k", "1645", type=float)
                        self.input_d0 = QtShortCuts.QInputString(None, "d0", "0.0008", type=float)
                    with QtShortCuts.QHBoxLayout() as layout2:
                        self.input_lamda_s = QtShortCuts.QInputString(None, "lamdba_s", "0.0075", type=float)
                        self.input_ds = QtShortCuts.QInputString(None, "ds", "0.033", type=float)

            with QtShortCuts.QGroupBox(None, "Regularisation Parameters") as self.material_parameters:
                with QtShortCuts.QVBoxLayout() as layout:
                    self.input_alpha = QtShortCuts.QInputString(None, "alpha", "9", type=float)
                    with QtShortCuts.QHBoxLayout(None) as layout:
                        self.input_stepper = QtShortCuts.QInputString(None, "stepper", "0.33", type=float)
                        self.input_imax = QtShortCuts.QInputNumber(None, "i_max", 100, float=False)

            self.input_button = QtShortCuts.QPushButton(None, "calculate forces", self.start_process)

            self.progressbar = QProgressBar(main_layout)

        with self.parent.tabs.createTab("Forces") as self.tab:
            with QtShortCuts.QVBoxLayout() as layout:
                self.canvas = MatplotlibWidget(self)
                layout.addWidget(self.canvas)
                layout.addWidget(NavigationToolbar(self.canvas, self))

                self.plotter = QtInteractor(self)
                self.plotter.set_background("black")
                layout.addWidget(self.plotter.interactor)

        self.setParameterMapping("solve_parameter", {
            "k": self.input_k,
            "d0": self.input_d0,
            "lambda_s": self.input_lamda_s,
            "ds": self.input_ds,
            "alpha": self.input_alpha,
            "stepper": self.input_stepper,
            "i_max": self.input_imax,
        })

        self.iteration_finished.emit(np.ones([10, 3]))

    def check_available(self, result: Result):
        return result is not None and result.solver is not None

    def iteration_callback(self, relrec):
        relrec = np.array(relrec)
        self.canvas.figure.axes[0].cla()
        self.canvas.figure.axes[0].semilogy(relrec[:, 0])
        self.canvas.figure.axes[0].semilogy(relrec[:, 1])
        self.canvas.figure.axes[0].semilogy(relrec[:, 2])
        self.canvas.figure.axes[0].set_xlabel("iteration")
        self.canvas.figure.axes[0].set_ylabel("error")
        self.canvas.draw()

    def process(self, result: Result, params: dict):
        M = result.solver

        def callback(M, relrec):
            self.iteration_finished.emit(relrec)

        M.setMaterialModel(saenopy.materials.SemiAffineFiberMaterial(
                           params["k"],
                           params["d0"],
                           params["lambda_s"],
                           params["ds"],
                           ))

        M.solve_regularized(stepper=params["stepper"], i_max=params["i_max"],
                            alpha=params["alpha"], callback=callback, verbose=True)

    def update_display(self):
        if self.result is not None and self.result.mesh_piv is not None:
            self.parent.tabs.setTabEnabled(3, True)
            showVectorField(self.plotter, self.result.solver, "f", factor=3e4)
        else:
            self.parent.tabs.setTabEnabled(3, False)



class BatchEvaluate(QtWidgets.QWidget):
    result_changed = QtCore.Signal(object)
    set_current_result = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.settings = QtCore.QSettings("Saenopy", "Seanopy_deformation")

        with QtShortCuts.QHBoxLayout(self) as main_layout:
            with QtShortCuts.QSplitter() as lay:
                with QtShortCuts.QVBoxLayout() as layout:
                    layout.setContentsMargins(0, 0, 0, 0)
                    self.list = ListWidget(layout, add_item_button="add measurements")
                    self.list.addItemClicked.connect(self.show_files)
                    self.list.itemSelectionChanged.connect(self.listSelected)
                    self.progress1 = QtWidgets.QProgressBar()
                    layout.addWidget(self.progress1)
         #           self.label = QtWidgets.QLabel(
         #               "Select a path as an input wildcard. Use * to specify a placeholder. All paths that match the wildcard will be added.").addToLayout()
                with QtShortCuts.QHBoxLayout() as layout:
                    layout.setContentsMargins(0, 0, 0, 0)
                    with QtShortCuts.QTabWidget(layout) as self.tabs:
                        pass
                with QtShortCuts.QVBoxLayout() as layout0:
                    self.sub_module_stacks = StackDisplay(self, layout0)
                    with CheckAbleGroup(self, "find deformations (piv)").addToLayout() as self.find_deformations:
                        with QtShortCuts.QVBoxLayout() as layout:
                            layout.setContentsMargins(0, 0, 0, 0)
                            self.sub_module_deformation = DeformationDetector(self, layout)
                    with CheckAbleGroup(self, "interpolate mesh").addToLayout() as self.find_deformations:
                        with QtShortCuts.QVBoxLayout() as layout:
                            layout.setContentsMargins(0, 0, 0, 0)
                            self.sub_module_mesh = MeshCreator(self, layout)
                    with CheckAbleGroup(self, "fit forces (regularize)").addToLayout() as self.find_deformations:
                        with QtShortCuts.QVBoxLayout() as layout:
                            layout.setContentsMargins(0, 0, 0, 0)
                            self.sub_module_regularize = Regularizer(self, layout)
                    layout0.addStretch()

        self.data = []
        self.list.setData(self.data)

        #self.list.addData("foo", True, [], mpl.colors.to_hex(f"C0"))

        data = Result.load(r"E:\saenopy\test\TestData\output\Mark_and_Find_001_Pos001_S001_z_ch00.npz")
        self.list.addData("test", True, data, mpl.colors.to_hex(f"gray"))

    def show_files(self):
        settings = self.settings

        class AddFilesDialog(QtWidgets.QDialog):
            def __init__(self, parent):
                super().__init__(parent)
                self.setMinimumWidth(800)
                self.setMinimumHeight(600)
                self.setWindowTitle("Add Files")
                with QtShortCuts.QVBoxLayout(self) as layout:
                    self.outputText = QtShortCuts.QInputFolder(None, "output", settings=settings,
                                                               settings_key="batch/wildcard2", allow_edit=True)
                    with QtShortCuts.QHBoxLayout() as layout3:
                        self.stack_before = StackSelector(layout3, "deformed")
                        self.stack_before.glob_string_changed.connect(lambda x, y: self.input_deformed.setText(y.replace("*", "?")))
                        self.stack_after = StackSelector(layout3, "relaxed", self.stack_before)
                        self.stack_after.glob_string_changed.connect(lambda x, y: self.input_relaxed.setText(y.replace("*", "?")))
                    with QtShortCuts.QHBoxLayout() as layout3:
                        self.input_deformed = QtWidgets.QLineEdit().addToLayout()
                        self.input_relaxed = QtWidgets.QLineEdit().addToLayout()
                        self.input_relaxed.text()
                    with QtShortCuts.QHBoxLayout() as layout3:
                        # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                        layout3.addStretch()
                        self.button_addList2 = QtShortCuts.QPushButton(None, "cancel", self.reject)
                        self.button_addList1 = QtShortCuts.QPushButton(None, "ok", self.accept)
        #getStack
        dialog = AddFilesDialog(self)
        if not dialog.exec():
            return

        results1, output_base = double_glob(dialog.input_deformed.text())
        results2, _ = double_glob(dialog.input_relaxed.text())
        voxel_size1 = dialog.stack_before.getVoxelSize()
        voxel_size2 = dialog.stack_after.getVoxelSize()

        for r1, r2 in zip(results1, results2):
            output = Path(dialog.outputText.value()) / os.path.relpath(r1, output_base)
            output = output.parent / output.stem
            output = Path(str(output).replace("*", "")+".npz")

            if output.exists():
                print('exists')
                data = Result.load(output)
            else:
                print("new")
                data = Result(
                    output=output,
                    stack_deformed=Stack(r1, voxel_size1),
                    stack_relaxed=Stack(r2, voxel_size2),
                )
            print(r1, r2, output)
            self.list.addData(r1, True, data, mpl.colors.to_hex(f"gray"))

        #import matplotlib as mpl
        #for fiber, cell, out in zip(fiber_list, cell_list, out_list):
        #    self.list.addData(fiber, True, [fiber, cell, out, {"segmention_thres": None, "seg_gaus1": None, "seg_gaus2": None}], mpl.colors.to_hex(f"gray"))

    def listSelected(self):
        if self.list.currentRow() is not None:
            pipe = self.data[self.list.currentRow()][2]
            self.set_current_result.emit(pipe)



class MainWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # QSettings
        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

        self.setMinimumWidth(800)
        self.setMinimumHeight(800)
        self.setWindowTitle("Saenopy Viewer")

        main_layout = QtWidgets.QHBoxLayout(self)

        with QtShortCuts.QTabWidget(main_layout) as self.tabs:
            """ """
            with self.tabs.createTab("Compaction") as v_layout:
                with QtShortCuts.QHBoxLayout() as h_layout:
                    # self.deformations = Deformation(h_layout, self)
                    self.deformations = BatchEvaluate(self)
                    h_layout.addWidget(self.deformations)
                    self.description = QtWidgets.QTextEdit()
                    self.description.setDisabled(True)
                    self.description.setMaximumWidth(300)
                    h_layout.addWidget(self.description)
                    self.description.setText("""
                    <h1>Start Evaluation</h1>
                     """.strip())
                v_layout.addWidget(QHLine())
                with QtShortCuts.QHBoxLayout() as h_layout:
                    h_layout.addStretch()
                    #self.button_previous = QtShortCuts.QPushButton(None, "back", self.previous)
                    #self.button_next = QtShortCuts.QPushButton(None, "next", self.next)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    print(sys.argv)
    window = MainWindow()
    if len(sys.argv) >= 2:
        window.loadFile(sys.argv[1])
    window.show()
    sys.exit(app.exec_())
