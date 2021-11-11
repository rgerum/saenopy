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
from saenopy.multigridHelper import getScaledMesh, getNodesWithOneFace
import matplotlib as mpl
from saenopy.solver import Solver
from pathlib import Path
import re
from saenopy.loadHelpers import Saveable
from typing import List

"""REFERENCE FOLDERS"""
#\\131.188.117.96\biophysDS2\dboehringer\Platte_4\SoftwareWorkinProgess\TFM-Example-Data-3D\a127-tom-test-set\20170914_A172_rep1-bispos3\Before
#\\131.188.117.96\biophysDS\lbischof\tif_and_analysis_backup\2021-06-02-NK92-Blebb-Rock\Blebb-round1\Mark_and_Find_001


def showVectorField(plotter, obj, name, show_nan=True, show_all_points=False, factor=5):
    try:
        field = getattr(obj, name)
    except AttributeError:
        field = obj.getNodeVar(name)
    nan_values = np.isnan(field[:, 0])

    plotter.clear()

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

    title = name
    if name == "U_measured" or name == "U_target":
        title = "Deformations (m)"
    elif name == "f":
        title = "Forces (N)"

    sargs = dict(#position_x=0.05, position_y=0.95,
                 title_font_size=15,
                 label_font_size=9,
                 n_labels=3,
                 title=title,
                 #italic=True,  ##height=0.25, #vertical=True,
                 fmt="%.1e",
                 color=plotter._theme.font.color,
                 font_family="arial")
    plotter.add_mesh(arrows, scalar_bar_args=sargs, colormap="turbo", name="arrows")

    plotter.update_scalar_bar_range([0, np.nanpercentile(point_cloud[name + "_mag2"], 99.9)])

    print(plotter._theme.font.color)
    plotter.show_grid(color=plotter._theme.font.color)
    #plotter.renderer.show_bounds(color=plotter._theme.font.color)
    plotter.show()


class Result(Saveable):
    __save_parameters__ = ['output', 'stack', 'piv_parameter', 'mesh_piv',
                           'interpolate_parameter', 'solve_parameter', 'solver',
                           '___save_name__', '___save_version__']
    ___save_name__ = "Result"
    ___save_version__ = "1.0"
    output: str = None
    state: False

    stack: List[Stack] = None

    piv_parameter: dict = None
    mesh_piv: List[saenopy.solver.Mesh] = None

    interpolate_parameter: dict = None
    solve_parameter: dict = None
    solver: List[saenopy.solver.Solver] = None

    def __init__(self, output, stack, **kwargs):
        self.output = str(output)

        self.stack = stack
        self.state = False

        super().__init__(**kwargs)

    def __strX__(self):
        return f"""
from saenopy.getDeformations import Stack
stack_deformed = Stack({self.deformed}) 
stack_relaxed = Stack({self.relaxed})
"""

    def save(self):
        Path(self.output).parent.mkdir(exist_ok=True, parents=True)
        super().save(self.output)


if 0:
    result = Result("test.npz", [Stack(r"../test/TestData/20170914_A172_rep1/Before/*.tif", [1, 1, 1])])
    result.mesh_piv = saenopy.solver.Mesh()
    result.mesh_piv.setNodes(np.zeros([4, 3]))
    print(result.to_dict())
    result.save()

    Result.load("test.npz")
    exit()
#result = Result.load("/home/richard/Dropbox/Projects/2020 Sanopy/saenopy/test/TestData/output4/Mark_and_Find_001_Pos001_S001_z_ch00.npz")
#pprint(result.stack[0].shape)
#print(result.stack[0][:, :, 1])
#exit()

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



class ModuleScaleBar(QtWidgets.QGroupBox):
    pixtomu = None

    def __init__(self, parent, view):
        QtWidgets.QWidget.__init__(self)
        self.parent = parent

        self.font = QtGui.QFont()
        self.font.setPointSize(16)

        self.scale = 1

        self.scalebar = QtWidgets.QGraphicsRectItem(0, 0, 1, 1, view.hud_lowerRight)
        self.scalebar.setBrush(QtGui.QBrush(QtGui.QColor("white")))
        self.scalebar.setPen(QtGui.QPen(QtGui.QColor("white")))
        self.scalebar.setPos(-20, -20)
        self.scalebar_text = QtWidgets.QGraphicsTextItem("", view.hud_lowerRight)
        self.scalebar_text.setFont(self.font)
        self.scalebar_text.setDefaultTextColor(QtGui.QColor("white"))

        view.signal_zoom.connect(self.zoomEvent)
        #self.parent.view.zoomEvent = lambda scale, pos: self.zoomEvent(scale, pos)
        #self.parent.signal_objective_changed.connect(self.updateStatus)
        #self.parent.signal_coupler_changed.connect(self.updateStatus)

        self.updateStatus()

    def updateStatus(self):
        self.updateBar()

    def zoomEvent(self, scale, pos):
        self.scale = scale
        self.updateBar()

    def setScale(self, voxel_size):
        self.pixtomu = voxel_size[0]
        self.updateBar()

    def updateBar(self):
        if self.scale == 0 or self.pixtomu is None:
            return
        mu = 100*self.pixtomu/self.scale
        values = [1, 5, 10, 25, 50, 75, 100, 150, 200, 250, 500, 1000, 1500, 2000, 2500, 5000, 10000]
        old_v = mu
        for v in values:
            if mu < v:
                mu = old_v
                break
            old_v = v
        pixel = mu/(self.pixtomu)*self.scale
        self.scalebar.setRect(0, 0, -pixel, 5)
        self.scalebar_text.setPos(-pixel-20-25, -20-30)
        self.scalebar_text.setTextWidth(pixel+50)
        self.scalebar_text.setHtml(u"<center>%d&thinsp;µm</center>" % mu)



class QProgressBar(QtWidgets.QProgressBar):
    signal_start = QtCore.Signal(int)
    signal_progress = QtCore.Signal(int)

    def __init__(self, layout=None):
        super().__init__()
        self.setOrientation(QtCore.Qt.Horizontal)
        if layout is not None:
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
    processing_state_changed = QtCore.Signal(object)
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
        self.processing_state_changed.connect(self.state_changed)

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

    def state_changed(self, result: Result):
        if result is self.result and getattr(self, "group", None) is not None:
            if getattr(result, self.params_name + "_state", "") == "scheduled":
                self.group.label.setIcon(qta.icon("fa.hourglass-o", options=[dict(color="gray")]))
                self.group.label.setToolTip("scheduled")
            elif getattr(result, self.params_name + "_state", "") == "running":
                self.group.label.setIcon(qta.icon("fa.hourglass", options=[dict(color="orange")]))
                self.group.label.setToolTip("running")
            elif getattr(result, self.params_name + "_state", "") == "finished":
                self.group.label.setIcon(qta.icon("fa.check", options=[dict(color="green")]))
                self.group.label.setToolTip("finished")
            elif getattr(result, self.params_name + "_state", "") == "failed":
                self.group.label.setIcon(qta.icon("fa.cross", options=[dict(color="red")]))
                self.group.label.setToolTip("failed")
            else:
                self.group.label.setIcon(qta.icon("fa.circle", options=[dict(color="gray")]))
                self.group.label.setToolTip("")

    def setResult(self, result: Result):
        """ set a new active result object """
        self.result = result

        self.state_changed(result)

        # check if the results instance can be evaluated currently with this module
        if self.check_available(result) is False:
            # if not disable all the widgets
            for name, widget in self.parameter_dict.items():
                widget.setDisabled(True)
        else:
            self.ensure_tmp_params_initialized(result)
            # iterate over the parameters
            for name, widget in self.parameter_dict.items():
                # enable them
                widget.setDisabled(False)
                # set the widgets to the value if the value exits
                params_tmp = getattr(result, self.params_name + "_tmp")
                widget.setValue(params_tmp[name])
            self.valueChanged()
        self.update_display()

    def update_display(self):
        pass

    def setParameter(self, name: str, value):
        if self.result is not None:
            getattr(self.result, self.params_name + "_tmp")[name] = value

    def valueChanged(self):
        pass

    def ensure_tmp_params_initialized(self, result):
        # if the results instance does not have the parameter dictionary yet, create it
        if getattr(result, self.params_name + "_tmp", None) is None:
            setattr(result, self.params_name + "_tmp", {})
        # iterate over the parameters
        for name, widget in self.parameter_dict.items():
            # set the widgets to the value if the value exits
            params = getattr(result, self.params_name, None)
            params_tmp = getattr(result, self.params_name + "_tmp")
            if name not in params_tmp:
                if params is not None and name in params:
                    params_tmp[name] = params[name]
                else:
                    params_tmp[name] = widget.value()

    def start_process(self, x=None, result=None):
        if result is None:
            result = self.result
        if result is None:
            return
        if getattr(result, self.params_name + "_state", "") == "scheduled" or \
            getattr(result, self.params_name + "_state", "") == "running":
            return
        self.ensure_tmp_params_initialized(result)
        params = getattr(result, self.params_name + "_tmp")
        setattr(result, self.params_name + "_state", "scheduled")
        self.processing_state_changed.emit(result)
        return self.parent.addTask(self.process_thread, result, params, "xx")

    def process_thread(self, result: Result, params: dict):
        #params = getattr(result, self.params_name + "_tmp")
        setattr(result, self.params_name + "_state", "running")
        self.processing_state_changed.emit(result)
        try:
            self.process(result, params)
            # store the parameters that have been used for evaluation
            setattr(result, self.params_name, params.copy())
            result.save()
            setattr(result, self.params_name + "_state", "finished")
            self.parent.result_changed.emit(result)
            self.processing_finished.emit()
        except Exception as err:
            import traceback
            traceback.print_exc()
            setattr(result, self.params_name + "_state", "failed")
            self.processing_state_changed.emit(result)
            self.processing_error.emit(str(err))

    def process(self, result: Result, params: dict):
        pass

    def finished_process(self):
        self.input_button.setEnabled(True)
        self.parent.progressbar.setRange(0, 1)

    def errored_process(self, text: str):
        QtWidgets.QMessageBox.critical(self, "Deformation Detector", text)
        self.input_button.setEnabled(True)
        self.parent.progressbar.setRange(0, 1)


class StackDisplay(PipelineModule):
    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with self.parent.tabs.createTab("Stacks") as self.tab:
            with QtShortCuts.QHBoxLayout() as layout:
                with QtShortCuts.QVBoxLayout() as layout:
                    with QtShortCuts.QHBoxLayout() as layout:
                        self.label1 = QtWidgets.QLabel("deformed").addToLayout()
                        layout.addStretch()
                        self.button = QtWidgets.QPushButton(qta.icon("fa5s.home"), "").addToLayout()
                        self.button.setToolTip("reset view")
                        self.button.clicked.connect(lambda x: (self.view1.fitInView(), self.view2.fitInView()))
                        self.button2 = QtWidgets.QPushButton(qta.icon("fa5s.save"), "").addToLayout()
                        self.button2.setToolTip("save image")
                        self.button2.clicked.connect(self.export)
                    self.view1 = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
                    self.view1.setMinimumWidth(300)
                    self.pixmap1 = QtWidgets.QGraphicsPixmapItem(self.view1.origin)
                    self.scale1 = ModuleScaleBar(self, self.view1)

                    self.label2 = QtWidgets.QLabel("relaxed").addToLayout()
                    self.view2 = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
                    # self.label2.setMinimumWidth(300)
                    self.pixmap2 = QtWidgets.QGraphicsPixmapItem(self.view2.origin)
                    self.scale2 = ModuleScaleBar(self, self.view2)
                self.z_slider = QtWidgets.QSlider(QtCore.Qt.Vertical).addToLayout()
                self.z_slider.valueChanged.connect(self.z_slider_value_changed)
                self.z_slider.setToolTip("set z position")

        self.view1.link(self.view2)

        self.setParameterMapping(None, {})

    def export(self):
        if self.result is None:
            return
        import tifffile
        new_path = QtWidgets.QFileDialog.getSaveFileName(None, "Save Images", os.getcwd())
        # if we got one, set it
        if new_path:
            if isinstance(new_path, tuple):
                new_path = new_path[0]
            else:
                new_path = str(new_path)
            new_path = new_path.strip(".gif").strip("_relaxed.tif").strip("_deformed.tif")
            new_path = Path(new_path)
            print(new_path.parent / (new_path.stem + "_deformed.tif"))
            tifffile.imsave(new_path.parent / (new_path.stem + "_deformed.tif"), self.result.stack[0][:, :, self.z_slider.value()])
            tifffile.imsave(new_path.parent / (new_path.stem + "_relaxed.tif"), self.result.stack[1][:, :, self.z_slider.value()])
            imageio.mimsave(new_path.parent / (new_path.stem + ".gif"), [self.result.stack[0][:, :, self.z_slider.value()], self.result.stack[1][:, :, self.z_slider.value()]], fps=2)

    def update_display(self):
        if self.result is not None:
            self.parent.tabs.setTabEnabled(0, True)
            self.view1.setToolTip(f"deformed stack\n{self.result.stack[0].description()}")
            self.view2.setToolTip(f"relaxed stack\n{self.result.stack[1].description()}")
            self.scale1.setScale(self.result.stack[0].voxel_size)
            self.scale2.setScale(self.result.stack[1].voxel_size)
            self.z_slider.setRange(0, self.result.stack[0].shape[2] - 1)
            self.z_slider.setValue(self.result.stack[0].shape[2] // 2)
            self.z_slider_value_changed()
        else:
            self.parent.tabs.setTabEnabled(0, False)

    def z_slider_value_changed(self):
        if self.result is not None:
            im = self.result.stack[0][:, :, self.z_slider.value()]
            self.pixmap1.setPixmap(QtGui.QPixmap(array2qimage(im)))
            self.view1.setExtend(im.shape[0], im.shape[0])

            im = self.result.stack[1][:, :, self.z_slider.value()]
            self.pixmap2.setPixmap(QtGui.QPixmap(array2qimage(im)))
            self.view2.setExtend(im.shape[1], im.shape[0])
            self.z_slider.setToolTip(f"set z position\ncurrent position {self.z_slider.value()}")


vtk_toolbars = []
class VTK_Toolbar(QtWidgets.QWidget):
    theme_values = [pv.themes.DefaultTheme(), pv.themes.ParaViewTheme(),
                                                          pv.themes.DarkTheme(), pv.themes.DocumentTheme()]
    def __init__(self, plotter, update_display):
        super().__init__()
        self.plotter = plotter
        self.update_display = update_display
        vtk_toolbars.append(self)

        with QtShortCuts.QHBoxLayout(self) as layout0:
            layout0.setContentsMargins(0, 0, 0, 0)
            self.theme = QtShortCuts.QInputChoice(None, "Theme", value=self.theme_values[2],
                                                  values=self.theme_values,
                                                  value_names=["default", "paraview", "dark", "document"])



            self.theme.valueChanged.connect(lambda x: self.new_plotter(x))

            layout0.addStretch()
            self.button = QtWidgets.QPushButton(qta.icon("fa5s.home"), "").addToLayout()
            self.button.setToolTip("reset view")
            self.button.clicked.connect(lambda x: self.plotter.isometric_view())

            def save():
                outer_self = self

                class PlotDialog(QtWidgets.QDialog):
                    def __init__(self, parent):
                        super().__init__(parent)
                        with QtShortCuts.QVBoxLayout(self) as layout:
                            self.plotter = QtInteractor(self, theme=outer_self.plotter.theme)
                            layout.addWidget(self.plotter)
                            showVectorField(self.plotter, outer_self.result.mesh_piv, "U_measured")
                            self.button2 = QtWidgets.QPushButton(qta.icon("fa5s.save"), "").addToLayout()
                            self.button2.setToolTip("save")
                            self.button2.clicked.connect(self.save)

                    def save(self):
                        new_path = QtWidgets.QFileDialog.getSaveFileName(None, "Save Images", os.getcwd())
                        # if we got one, set it
                        if new_path:
                            if isinstance(new_path, tuple):
                                new_path = new_path[0]
                            else:
                                new_path = str(new_path)
                            print(new_path)
                            self.plotter.screenshot(new_path)
                        self.close()

                plot_diaolog = PlotDialog(self)
                plot_diaolog.show()

            self.button2 = QtWidgets.QPushButton(qta.icon("fa5s.save"), "").addToLayout()
            self.button2.setToolTip("save")
            self.button2.clicked.connect(save)

    def new_plotter(self, x, no_recursion=False):
        # layout0.removeWidget(self.plotter.interactor)
        # self.plotter = QtInteractor(self, theme=x)
        # self.plotter.close()
        # layout0.addWidget(self.plotter.interactor)
        if self.plotter.theme == x:
            return
        if no_recursion is False:
            for widget in vtk_toolbars:
                if widget is not self:
                    widget.theme.setValue(x)
                    widget.new_plotter(x, no_recursion=True)
        self.plotter.theme = x
        self.plotter.set_background(self.plotter._theme.background)
        print(self.plotter._theme.font.color)
        self.update_display()


class DeformationDetector(PipelineModule):
    pipeline_name = "find deformations"

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with QtShortCuts.QVBoxLayout(self) as layout:
            layout.setContentsMargins(0, 0, 0, 0)
            with CheckAbleGroup(self, "find deformations (piv)").addToLayout() as self.group:

                with QtShortCuts.QVBoxLayout() as layout:
                    with QtShortCuts.QHBoxLayout():
                        self.input_overlap = QtShortCuts.QInputNumber(None, "overlap", 0.6, step=0.1, value_changed=self.valueChanged)
                        self.input_win = QtShortCuts.QInputNumber(None, "window size", 30, value_changed=self.valueChanged, unit="μm")
                    with QtShortCuts.QHBoxLayout():
                        self.input_signoise = QtShortCuts.QInputNumber(None, "signoise", 1.3, step=0.1)
                        self.input_driftcorrection = QtShortCuts.QInputBool(None, "driftcorrection", True)
                    self.label = QtWidgets.QLabel().addToLayout()
                    self.input_button = QtShortCuts.QPushButton(None, "detect deformations", self.start_process)

        with self.parent.tabs.createTab("Deformations") as self.tab:
            with QtShortCuts.QVBoxLayout() as layout:
                self.plotter = QtInteractor(self)#, theme=pv.themes.DocumentTheme())
                self.tab.parent().plotter = self.plotter
                self.plotter.set_background("black")
                #self.plotter.theme = pv.themes.DocumentTheme()
                self.plotter.reset_camera()
                VTK_Toolbar(self.plotter, self.update_display).addToLayout()

                layout.addWidget(self.plotter.interactor)

                self.t_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal).addToLayout()
                self.t_slider.valueChanged.connect(self.update_display)
                self.t_slider.setToolTip("set time")

        self.setParameterMapping("piv_parameter", {
            "win_um": self.input_win,
            "fac_overlap": self.input_overlap,
            "signoise_filter": self.input_signoise,
            "drift_correction": self.input_driftcorrection,
        })

    def check_available(self, result: Result):
        return result is not None and result.stack is not None

    def update_display(self):
        if self.result is not None and self.result.mesh_piv is not None:
            self.parent.tabs.setTabEnabled(1, True)
            self.plotter.interactor.setToolTip(str(self.result.piv_parameter)+f"\nNodes {self.result.mesh_piv[0].R.shape[0]}\nTets {self.result.mesh_piv[0].T.shape[0]}")
            showVectorField(self.plotter, self.result.mesh_piv[self.t_slider.value()], "U_measured")
        else:
            self.parent.tabs.setTabEnabled(1, False)

    def valueChanged(self):
        if self.check_available(self.result):
            self.t_slider.setRange(0, len(self.result.stack)-2)
            voxel_size1 = self.result.stack[0].voxel_size
            stack_deformed = self.result.stack[0]

            unit_size = (1-self.input_overlap.value())*self.input_win.value()
            stack_size = np.array(stack_deformed.shape)*voxel_size1 - self.input_win.value()
            self.label.setText(f"Deformation grid with {unit_size:.1f}μm elements.\nTotal region is {stack_size}.")
        else:
            self.label.setText("")

    def process(self, result: Result, params: dict):
        import tqdm
        t = tqdm.tqdm
        n = tqdm.tqdm.__new__
        tqdm.tqdm.__new__ = lambda cls, iter: self.parent.progressbar.iterator(iter)
        if not isinstance(result.mesh_piv, list):
            result.mesh_piv = [None]*(len(result.stack)-1)
        for i in range(len(result.stack)-1):
            result.mesh_piv[i] = saenopy.getDeformations.getDisplacementsFromStacks2(result.stack[i], result.stack[i+1],
                                       params["win_um"], params["fac_overlap"], params["signoise_filter"],
                                       params["drift_correction"])



class MeshCreator(PipelineModule):
    mesh_size = [200, 200, 200]
    pipeline_name = "interpolate mesh"

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with QtShortCuts.QVBoxLayout(self) as layout:
            layout.setContentsMargins(0, 0, 0, 0)
            with CheckAbleGroup(self, "interpolate mesh").addToLayout() as self.group:
                with QtShortCuts.QVBoxLayout() as layout:
                    with QtShortCuts.QHBoxLayout():
                        with QtShortCuts.QVBoxLayout() as layout2:
                            self.input_element_size = QtShortCuts.QInputNumber(None, "element_size", 7, unit="μm")
                            #with QtShortCuts.QHBoxLayout() as layout2:
                            self.input_inner_region = QtShortCuts.QInputNumber(None, "inner_region", 100, unit="μm")
                            self.input_thinning_factor = QtShortCuts.QInputNumber(None, "thinning factor", 0.2, step=0.1)
                            layout2.addStretch()
                        with QtShortCuts.QVBoxLayout() as layout2:
                            self.input_mesh_size_same = QtShortCuts.QInputBool(None, "mesh size same as stack", True, value_changed=self.valueChanged)
                            self.input_mesh_size_x = QtShortCuts.QInputNumber(None, "x", 200, step=1, name_post="μm")
                            self.input_mesh_size_y = QtShortCuts.QInputNumber(None, "y", 200, step=1, name_post="μm")
                            self.input_mesh_size_z = QtShortCuts.QInputNumber(None, "z", 200, step=1, name_post="μm")
                            #self.input_mesh_size_label = QtWidgets.QLabel("μm").addToLayout()
                        self.valueChanged()

                    self.input_button = QtWidgets.QPushButton("interpolate mesh").addToLayout()
                    self.input_button.clicked.connect(self.start_process)

        with self.parent.tabs.createTab("Mesh") as self.tab:
            with QtShortCuts.QVBoxLayout() as layout:
                self.plotter = QtInteractor(self)
                self.tab.parent().plotter = self.plotter
                self.plotter.set_background("black")
                VTK_Toolbar(self.plotter, self.update_display).addToLayout()
                layout.addWidget(self.plotter.interactor)

                self.t_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal).addToLayout()
                self.t_slider.valueChanged.connect(self.update_display)
                self.t_slider.setToolTip("set time")

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
                x, y, z = (self.result.mesh_piv[0].R.max(axis=0) - self.result.mesh_piv[0].R.min(axis=0))*1e6
                self.input_mesh_size_x.setValue(x)
                self.setParameter("mesh_size_x", x)
                self.input_mesh_size_y.setValue(y)
                self.setParameter("mesh_size_y", y)
                self.input_mesh_size_z.setValue(z)
                self.setParameter("mesh_size_z", z)

    def check_available(self, result: Result):
        return result is not None and result.mesh_piv is not None

    def update_display(self):
        if self.result is not None and self.result.solver is not None:
            self.parent.tabs.setTabEnabled(2, True)
            self.t_slider.setRange(0, len(self.result.mesh_piv) - 1)
            self.plotter.interactor.setToolTip(str(self.result.interpolate_parameter)+f"\nNodes {self.result.solver[self.t_slider.value()].R.shape[0]}\nTets {self.result.solver[self.t_slider.value()].T.shape[0]}")
            showVectorField(self.plotter, self.result.solver[self.t_slider.value()], "U_target", factor=5)
        else:
            self.plotter.interactor.setToolTip("")
            self.parent.tabs.setTabEnabled(2, False)

    def process(self, result: Result, params: dict):
        solvers = []
        for i in range(len(result.mesh_piv)):
            M = result.mesh_piv[i]
            points, cells = saenopy.multigridHelper.getScaledMesh(params["element_size"]*1e-6,
                                          params["inner_region"]*1e-6,
                                          np.array([params["mesh_size_x"], params["mesh_size_y"],
                                                     params["mesh_size_z"]])*1e-6 / 2,
                                          [0, 0, 0], params["thinning_factor"])

            R = (M.R - np.min(M.R, axis=0)) - (np.max(M.R, axis=0) - np.min(M.R, axis=0)) / 2
            U_target = saenopy.getDeformations.interpolate_different_mesh(R, M.getNodeVar("U_measured"), points)

            border_idx = getNodesWithOneFace(cells)
            inside_mask = np.ones(points.shape[0], dtype=bool)
            inside_mask[border_idx] = False

            M = saenopy.Solver()
            M.setNodes(points)
            M.setTetrahedra(cells)
            M.setTargetDisplacements(U_target, inside_mask)

            solvers.append(M)
        result.solver = solvers


class Regularizer(PipelineModule):
    pipeline_name = "fit forces"
    iteration_finished = QtCore.Signal(object)

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with QtShortCuts.QVBoxLayout(self) as layout:
            layout.setContentsMargins(0, 0, 0, 0)
            with CheckAbleGroup(self, "fit forces (regularize)").addToLayout() as self.group:

                with QtShortCuts.QVBoxLayout() as main_layout:
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

        with self.parent.tabs.createTab("Forces") as self.tab:
            with QtShortCuts.QVBoxLayout() as layout:
                self.canvas = MatplotlibWidget(self)
                layout.addWidget(self.canvas)
                layout.addWidget(NavigationToolbar(self.canvas, self))

                self.plotter = QtInteractor(self)
                self.plotter.set_background("black")
                self.tab.parent().plotter = self.plotter
                VTK_Toolbar(self.plotter, self.update_display).addToLayout()
                layout.addWidget(self.plotter.interactor)

                self.t_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal).addToLayout()
                self.t_slider.valueChanged.connect(self.update_display)
                self.t_slider.setToolTip("set time")

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
        for i in range(len(result.solver)):
            M = result.solver[i]

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

    def valueChanged(self):
        if self.check_available(self.result):
            self.t_slider.setRange(0, len(self.result.mesh_piv)-1)

    def update_display(self):
        if self.result is not None and self.result.solver is not None and getattr(self.result.solver[self.t_slider.value()], "regularisation_results", None) is not None:
            self.parent.tabs.setTabEnabled(3, True)
            self.plotter.interactor.setToolTip(str(self.result.solve_parameter)+f"\nNodes {self.result.solver[self.t_slider.value()].R.shape[0]}\nTets {self.result.solver[self.t_slider.value()].T.shape[0]}")
            showVectorField(self.plotter, self.result.solver[self.t_slider.value()], "f", factor=3e4)
            self.iteration_callback(self.result.solver[self.t_slider.value()].regularisation_results)
        else:
            self.plotter.interactor.setToolTip("")
            self.parent.tabs.setTabEnabled(3, False)


class ResultView(PipelineModule):

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with self.parent.tabs.createTab("View") as self.tab:
            with QtShortCuts.QVBoxLayout() as vlayout:
                with QtShortCuts.QHBoxLayout() as layout_vert_plot:
                    self.input_checks = {}
                    for name in ["U_target", "U", "f", "stiffness"]:
                        input = QtShortCuts.QInputBool(layout_vert_plot, name, name != "stiffness")
                        input.valueChanged.connect(self.replot)
                        self.input_checks[name] = input
                    layout_vert_plot.addStretch()
                    self.button_export = QtWidgets.QPushButton(qta.icon("fa5s.save"), "")
                    self.button_export.setToolTip("save image")
                    layout_vert_plot.addWidget(self.button_export)
                    self.button_export.clicked.connect(self.saveScreenshot)

                # add the pyvista interactor object
                self.plotter_layout = QtWidgets.QHBoxLayout()
                self.plotter_layout.setContentsMargins(0, 0, 0, 0)
                self.frame = QtWidgets.QFrame().addToLayout()
                self.frame.setLayout(self.plotter_layout)

                self.plotter = QtInteractor(self.frame)
                self.plotter_layout.addWidget(self.plotter.interactor)
                vlayout.addLayout(self.plotter_layout)

                self.t_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal).addToLayout()
                self.t_slider.valueChanged.connect(self.update_display)
                self.t_slider.setToolTip("set time")
        self.setParameterMapping(None, {})

    def update_display(self):
        if self.result is not None and self.result.solver is not None and getattr(self.result.solver[0], "regularisation_results", None) is not None:
            self.parent.tabs.setTabEnabled(4, True)
            self.t_slider.setRange(0, len(self.result.solver)-1)
            self.M = self.result.solver[self.t_slider.value()]

            def scale(m):
                vmin, vmax = np.nanpercentile(m, [1, 99.9])
                return np.clip((m - vmin) / (vmax - vmin), 0, 1) * (vmax - vmin)

            R = self.M.R
            minR = np.min(R, axis=0)
            maxR = np.max(R, axis=0)

            if self.M.reg_mask is None:
                border = (R[:, 0] < minR[0] + 0.5e-6) | (R[:, 0] > maxR[0] - 0.5e-6) | \
                         (R[:, 1] < minR[1] + 0.5e-6) | (R[:, 1] > maxR[1] - 0.5e-6) | \
                         (R[:, 2] < minR[2] + 0.5e-6) | (R[:, 2] > maxR[2] - 0.5e-6)
                self.M.reg_mask = ~border

            self.point_cloud = pv.PolyData(self.M.R)
            self.point_cloud.point_arrays["f"] = -self.M.f * self.M.reg_mask[:, None]
            self.point_cloud["f_mag"] = np.linalg.norm(self.M.f * self.M.reg_mask[:, None], axis=1)
            self.point_cloud.point_arrays["U"] = self.M.U
            self.point_cloud["U_mag"] = np.linalg.norm(self.M.U, axis=1)
            self.point_cloud.point_arrays["U_target"] = self.M.U_target
            self.point_cloud["U_target_mag"] = np.linalg.norm(self.M.U_target, axis=1)

            self.point_cloud2 = None

            self.offset = np.min(self.M.R, axis=0)
            self.replot()
        else:
            self.plotter.interactor.setToolTip("")
            self.parent.tabs.setTabEnabled(4, False)

    def calculateStiffness(self):
        self.point_cloud2 = pv.PolyData(np.mean(self.M.R[self.M.T], axis=1))
        from saenopy.materials import SemiAffineFiberMaterial
        # self.M.setMaterialModel(SemiAffineFiberMaterial(1645, 0.0008, 0.0075, 0.033), generate_lookup=False)
        if self.M.material_parameters is not None:
            print("loading material")
            self.M.setMaterialModel(SemiAffineFiberMaterial(*self.M.material_parameters[1:]), generate_lookup=False)
        else:
            print("Warning using default material parameters")
            self.M.setMaterialModel(SemiAffineFiberMaterial(1645, 0.0008, 0.0075, 0.033), generate_lookup=False)
        self.M._check_relax_ready()
        self.M._prepare_temporary_quantities()
        self.point_cloud2["stiffness"] = self.M.getMaxTetStiffness() / 6

    point_cloud = None

    def replot(self):
        names = [name for name, input in self.input_checks.items() if input.value()]
        if len(names) == 0:
            return
        if len(names) <= 3:
            shape = (len(names), 1)
        else:
            shape = (2, 2)
        if self.plotter.shape != shape:
            self.plotter_layout.removeWidget(self.plotter)
            self.plotter.close()

            self.plotter = QtInteractor(self.frame, shape=shape, border=False)
            self.plotter.set_background("black")
            # pv.set_plot_theme("document")
            self.plotter_layout.addWidget(self.plotter.interactor)

        plotter = self.plotter
        # color bar design properties
        # Set a custom position and size
        sargs = dict(#position_x=0.05, position_y=0.95,
                     title_font_size=15,
                     label_font_size=9,
                     n_labels=3,
                     #italic=True,  ##height=0.25, #vertical=True,
                     fmt="%.1e",
                     font_family="arial")

        for i, name in enumerate(names):
            plotter.subplot(i // plotter.shape[1], i % plotter.shape[1])
            # scale plot with axis length later
            norm_stack_size = np.abs(np.max(self.M.R) - np.min(self.M.R))

            if name == "stiffness":
                if self.point_cloud2 is None:
                    self.calculateStiffness()
                # clim =  np.nanpercentile(self.point_cloud2["stiffness"], [50, 99.9])
                sargs2 = sargs.copy()
                sargs2["title"] = "Stiffness (Pa)"
                plotter.add_mesh(self.point_cloud2, colormap="turbo", point_size=4., render_points_as_spheres=True,
                                 scalar_bar_args=sargs2)
                plotter.update_scalar_bar_range(np.nanpercentile(self.point_cloud2["stiffness"], [50, 99.9]))
            elif name == "f":
                arrows = self.point_cloud.glyph(orient="f", scale="f_mag", \
                                                # Automatically scale maximal force to 15% of axis length
                                                factor=0.15 * norm_stack_size / np.nanmax(
                                                    np.linalg.norm(self.M.f * self.M.reg_mask[:, None], axis=1)))
                sargs2 = sargs.copy()
                sargs2["title"] = "Force (N)"
                plotter.add_mesh(arrows, colormap='turbo', name="arrows", scalar_bar_args=sargs2)
                plotter.update_scalar_bar_range(np.nanpercentile(self.point_cloud["f_mag"], [50, 99.9]))
                # plot center points if desired
                # plotter.add_points(np.array([self.M.getCenter(mode="Force")]), color='r')

            elif name == "U_target":
                arrows2 = self.point_cloud.glyph(orient="U_target", scale="U_target_mag", \
                                                 # Automatically scale maximal force to 10% of axis length
                                                 factor=0.1 * norm_stack_size / np.nanmax(
                                                     np.linalg.norm(self.M.U_target, axis=1)))
                sargs2 = sargs.copy()
                sargs2["title"] = "Deformations (m)"
                plotter.add_mesh(arrows2, colormap='turbo', name="arrows2", scalar_bar_args=sargs2)  #

                # plot center if desired
                # plotter.add_points(np.array([self.M.getCenter(mode="deformation_target")]), color='w')

                plotter.update_scalar_bar_range(np.nanpercentile(self.point_cloud["U_target_mag"], [50, 99.9]))
                # plotter.update_scalar_bar_range([0,1.5e-6])
            elif name == "U":
                arrows3 = self.point_cloud.glyph(orient=name, scale=name + "_mag", \
                                                 # Automatically scale maximal force to 10% of axis length
                                                 factor=0.1 * norm_stack_size / np.nanmax(
                                                     np.linalg.norm(self.M.U, axis=1)))
                sargs2 = sargs.copy()
                sargs2["title"] = "Rec. Deformations [m]"
                plotter.add_mesh(arrows3, colormap='turbo', name="arrows3", scalar_bar_args=sargs2)
                plotter.update_scalar_bar_range(np.nanpercentile(self.point_cloud["U_mag"], [50, 99.9]))
                # plotter.update_scalar_bar_range([0,1.5e-6])

            # plot center points if desired
            # plotter.add_points(np.array([self.M.getCenter(mode="Deformation")]), color='w')
            # plotter.add_points(np.array([self.M.getCenter(mode="Force")]), color='r')

            plotter.show_grid()
        # print(names)
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



class BatchEvaluate(QtWidgets.QWidget):
    result_changed = QtCore.Signal(object)
    set_current_result = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.settings = QtCore.QSettings("Saenopy", "Seanopy_deformation")

        with QtShortCuts.QHBoxLayout(self) as main_layout:
            main_layout.setContentsMargins(0, 0, 0, 0)
            with QtShortCuts.QSplitter() as lay:
                with QtShortCuts.QVBoxLayout() as layout:
                    layout.setContentsMargins(0, 0, 0, 0)
                    self.list = ListWidget(layout, add_item_button="add measurements")
                    self.list.addItemClicked.connect(self.show_files)
                    self.list.itemSelectionChanged.connect(self.listSelected)
                    self.progressbar = QProgressBar().addToLayout()
         #           self.label = QtWidgets.QLabel(
         #               "Select a path as an input wildcard. Use * to specify a placeholder. All paths that match the wildcard will be added.").addToLayout()
                with QtShortCuts.QHBoxLayout() as layout:
                    layout.setContentsMargins(0, 0, 0, 0)
                    with QtShortCuts.QTabWidget(layout) as self.tabs:
                        self.tabs.setMinimumWidth(500)
                        old_tab = None
                        cam_pos = None
                        def tab_changed(x):
                            print(x)
                            nonlocal old_tab, cam_pos
                            tab = self.tabs.currentWidget()
                            print(tab, old_tab, getattr(old_tab, "plotter", None), cam_pos)
                            if old_tab is not None and getattr(old_tab, "plotter", None):
                                print("set cam pos")
                                cam_pos = old_tab.plotter.camera_position
                            if cam_pos is not None and getattr(tab, "plotter", None):
                                tab.plotter.camera_position = cam_pos
                            old_tab = tab


                        self.tabs.currentChanged.connect(tab_changed)
                        pass
                with QtShortCuts.QVBoxLayout() as layout0:
                    layout0.parent().setMaximumWidth(420)
                    layout0.setContentsMargins(0, 0, 0, 0)
                    self.sub_module_stacks = StackDisplay(self, layout0)
                    self.sub_module_deformation = DeformationDetector(self, layout0)
                    self.sub_module_mesh = MeshCreator(self, layout0)
                    self.sub_module_regularize = Regularizer(self, layout0)
                    self.sub_module_view = ResultView(self, layout0)
                    layout0.addStretch()
                    self.button_start_all = QtShortCuts.QPushButton(None, "run all", self.run_all)

        self.data = []
        self.list.setData(self.data)

        #self.list.addData("foo", True, [], mpl.colors.to_hex(f"C0"))

        #data = Result.load(r"..\test\TestData\output3\Mark_and_Find_001_Pos001_S001_z_ch00.npz")
        #self.list.addData("test", True, data, mpl.colors.to_hex(f"gray"))

        self.setAcceptDrops(True)

        self.tasks = []
        self.current_task_id = 0
        self.thread = None
        self.signal_task_finished.connect(self.run_finished)

    def run_all(self):
        for i in range(len(self.data)):
            if not self.data[i][1]:
                continue
            result = self.data[i][2]
            if self.sub_module_deformation.group.value() is True:
                self.sub_module_deformation.start_process(result=result)
            if self.sub_module_mesh.group.value() is True:
                self.sub_module_mesh.start_process(result=result)
            if self.sub_module_regularize.group.value() is True:
                self.sub_module_regularize.start_process(result=result)

    def addTask(self, task, result, params, name):
        print("add task", task, result, params, name)
        self.tasks.append([task, result, params, name])
        if self.thread is None:
            self.run_next()

    signal_task_finished = QtCore.Signal()

    def run_next(self):
        task, result, params, name = self.tasks[self.current_task_id]
        self.thread = threading.Thread(target=self.run_thread, args=(task, result, params, name))
        self.thread.start()

    def run_thread(self, task, result, params, name):
        result.state = True
        self.update_icons()
        task(result, params)
        self.signal_task_finished.emit()
        result.state = False
        self.update_icons()

    def run_finished(self):
        self.current_task_id += 1
        self.thread = None
        if self.current_task_id < len(self.tasks):
            self.run_next()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        # accept url lists (files by drag and drop)
        for url in event.mimeData().urls():
            #if str(url.toString()).strip().endswith(".npz"):
            event.accept()
            return
        event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        event.acceptProposedAction()

    def dropEvent(self, event: QtCore.QEvent):
        for url in event.mimeData().urls():
            print(url)
            url = url.path()
            if url.endswith(".npz"):
                urls = [url]
            else:
                urls = glob.glob(url+"/**/*.npz", recursive=True)
            for url in urls:
                data = Result.load(url)
                self.list.addData(data.stack[0].filename, True, data, mpl.colors.to_hex(f"gray"))
        self.update_icons()

    def update_icons(self):
        for j in range(self.list.count()-1):
            if self.data[j][2].state is True:
                self.list.item(j).setIcon(qta.icon("fa.hourglass", options=[dict(color="orange")]))
            else:
                self.list.item(j).setIcon(qta.icon("fa.circle", options=[dict(color="gray")]))

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
                print('exists', output)
                data = Result.load(output)
            else:
                print("new")
                data = Result(
                    output=output,
                    stack=[Stack(r1, voxel_size1), Stack(r2, voxel_size2)],
                )
                data.save()
            print(r1, r2, output)
            self.list.addData(r1, True, data, mpl.colors.to_hex(f"gray"))
        self.update_icons()
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

        self.setMinimumWidth(1400)
        self.setMinimumHeight(900)
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
