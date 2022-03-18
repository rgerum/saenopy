import sys
import os
import qtawesome as qta
from qtpy import QtCore, QtWidgets, QtGui
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

import imageio
from qimage2ndarray import array2qimage
import matplotlib.pyplot as plt
import glob
import imageio
import threading
import inspect

import natsort

from pathlib import Path
import re
import pandas as pd
import matplotlib as mpl

import saenopy
import saenopy.multigridHelper
from saenopy.gui import QtShortCuts, QExtendedGraphicsView
from saenopy.gui.gui_classes import Spoiler, CheckAbleGroup, QHLine, QVLine, MatplotlibWidget, NavigationToolbar, execute, kill_thread, ListWidget, QProcess, ProcessSimple
import saenopy.getDeformations
import saenopy.multigridHelper
import saenopy.materials
from saenopy.gui.stack_selector import StackSelector
from saenopy.getDeformations import getStack, Stack, format_glob
from saenopy.multigridHelper import getScaledMesh, getNodesWithOneFace
from saenopy.loadHelpers import Saveable
from saenopy.solver import Result

from typing import List, Tuple

"""REFERENCE FOLDERS"""
#\\131.188.117.96\biophysDS2\dboehringer\Platte_4\SoftwareWorkinProgess\TFM-Example-Data-3D\a127-tom-test-set\20170914_A172_rep1-bispos3\Before
#\\131.188.117.96\biophysDS\lbischof\tif_and_analysis_backup\2021-06-02-NK92-Blebb-Rock\Blebb-round1\Mark_and_Find_001


def showVectorField(plotter, obj, field, name, center=None, show_nan=True, show_all_points=False, factor=.1, scalebar_max=None):
    #try:
    #    field = getattr(obj, name)
    #except AttributeError:
    #    field = obj.getNodeVar(name)
    nan_values = np.isnan(field[:, 0])

    #plotter.clear()

    point_cloud = pv.PolyData(obj.R)
    point_cloud.point_data[name] = field
    point_cloud.point_data[name + "_mag"] = np.linalg.norm(field, axis=1)
    point_cloud.point_data[name + "_mag2"] = point_cloud.point_data[name + "_mag"].copy()
    point_cloud.point_data[name + "_mag2"][nan_values] = 0
    if getattr(plotter, "nan_actor", None) is not None:
        plotter.remove_actor(plotter.nan_actor)
    if show_all_points:
        plotter.add_mesh(point_cloud, colormap="turbo", scalars=name + "_mag2")
    elif show_nan:
        R = obj.R[nan_values]
        if R.shape[0]:
            point_cloud2 = pv.PolyData(R)
            point_cloud2.point_data["nan"] = obj.R[nan_values, 0] * np.nan
            plotter.nan_actor = plotter.add_mesh(point_cloud2, colormap="turbo", scalars="nan", show_scalar_bar=False)

    norm_stack_size = np.abs(np.max(obj.R) - np.min(obj.R))
    if scalebar_max is None:
        factor = factor * norm_stack_size / np.nanmax(point_cloud[name + "_mag2"])#np.nanpercentile(point_cloud[name + "_mag2"], 99.9)
    else:
        factor = factor * norm_stack_size / scalebar_max

    # generate the arrows
    arrows = point_cloud.glyph(orient=name, scale=name + "_mag2", factor=factor)

    title = name
    if name == "U_measured" or name == "U_target" or name == "U":
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

    plotter.auto_value = np.nanpercentile(point_cloud[name + "_mag2"], 99.9)
    if scalebar_max is None:
        plotter.update_scalar_bar_range([0, np.nanpercentile(point_cloud[name + "_mag2"], 99.9)])
    else:
        plotter.update_scalar_bar_range([0, scalebar_max])

    if getattr(plotter, "center_actor", None) is not None:
        plotter.remove_actor(plotter.center_actor)
    if center is not None:
        # plot center points if desired
        plotter.center_actor = plotter.add_points(np.array([center]), color='m', point_size=10)

    plotter.show_grid(color=plotter._theme.font.color)
    #plotter.renderer.show_bounds(color=plotter._theme.font.color)
    plotter.show()



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

        self.time_text = QtWidgets.QGraphicsTextItem("", view.hud_upperRight)
        self.time_text.setFont(self.font)
        self.time_text.setDefaultTextColor(QtGui.QColor("white"))

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

class QTimeSlider(QtWidgets.QWidget):
    def __init__(self, name="t", connected=None, tooltip="set time", orientation=QtCore.Qt.Horizontal):
        super().__init__()
        self.tooltip_name = tooltip
        with (QtShortCuts.QHBoxLayout(self) if orientation == QtCore.Qt.Horizontal else QtShortCuts.QVBoxLayout(self)) as layout:
            layout.setContentsMargins(0, 0, 0, 0)
            self.label = QtWidgets.QLabel(name).addToLayout()
            self.label.setAlignment(QtCore.Qt.AlignCenter)
            self.t_slider = QtWidgets.QSlider(orientation).addToLayout()
            self.t_slider.valueChanged.connect(connected)
            self.t_slider.valueChanged.connect(self.value_changed)
            self.t_slider.setToolTip(tooltip)
        self.value = self.t_slider.value
        self.setValue = self.t_slider.setValue
        self.setRange = self.t_slider.setRange

    def value_changed(self):
        self.t_slider.setToolTip(self.tooltip_name+f"\n{self.t_slider.value()+1}/{self.t_slider.maximum()}")

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
        self.parent.tab_changed.connect(self.tabChanged)

    def setParameterMapping(self, params_name: str, parameter_dict: dict):
        self.params_name = params_name
        self.parameter_dict = parameter_dict
        for name, widget in self.parameter_dict.items():
            widget.valueChanged.connect(lambda x, name=name: self.setParameter(name, x))

        self.setResult(None)

    current_result_plotted = False
    current_tab_selected = False
    def tabChanged(self, tab):
        if self.tab is not None and self.tab.parent() == tab:
            if self.current_result_plotted is False:
                self.update_display()
                self.current_result_plotted = True
            self.current_tab_selected = True
        else:
            self.current_tab_selected = False

    def check_available(self, result: Result) -> bool:
        return False

    def check_evaluated(self, result: Result) -> bool:
        return False

    def resultChanged(self, result: Result):
        """ called when the contents of result changed. Only update view if its the currently displayed one. """
        if result is self.result:
            if self.tab is not None:
                for i in range(self.parent.tabs.count()):
                    if self.parent.tabs.widget(i) == self.tab.parent():
                       self.parent.tabs.setTabEnabled(i, self.check_evaluated(result))
            if self.current_tab_selected is True:
                self.update_display()
            self.state_changed(result)

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
                self.group.label.setIcon(qta.icon("fa.times", options=[dict(color="red")]))
                self.group.label.setToolTip("failed")
            else:
                self.group.label.setIcon(qta.icon("fa.circle", options=[dict(color="gray")]))
                self.group.label.setToolTip("")

    def setResult(self, result: Result):
        """ set a new active result object """
        #if result == self.result:
        #    return
        self.current_result_plotted = False
        self.result = result

        if result is not None:
            self.t_slider.setRange(0, len(result.stack)-2)

        self.state_changed(result)
        if self.tab is not None:
            for i in range(self.parent.tabs.count()):
                if self.parent.tabs.widget(i) == self.tab.parent():
                    self.parent.tabs.setTabEnabled(i, self.check_evaluated(result))

        # check if the results instance can be evaluated currently with this module
        #if self.check_available(result) is False:
        if result is None:
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
        if self.current_tab_selected is True:
            print(self.__class__.__name__, "Update Display")
            self.update_display()

    def update_display(self):
        pass

    def setParameter(self, name: str, value):
        if self.result is not None:
            getattr(self.result, self.params_name + "_tmp")[name] = value

    def valueChanged(self):
        pass

    def ensure_tmp_params_initialized(self, result):
        if self.params_name is None:
            return
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

    def get_code(self) -> Tuple[str, str]:
        return "", ""


class StackDisplay(PipelineModule):
    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with self.parent.tabs.createTab("Stacks") as self.tab:
            with QtShortCuts.QHBoxLayout() as layout:
                with QtShortCuts.QVBoxLayout() as layout:
                    with QtShortCuts.QHBoxLayout() as layout:
                        self.label1 = QtWidgets.QLabel("relaxed").addToLayout()
                        layout.addStretch()
                        self.contrast_enhance = QtShortCuts.QInputBool(None, "contrast enhance", False, settings=self.parent.settings, settings_key="stack_contrast_enhance")
                        self.contrast_enhance.valueChanged.connect(self.z_slider_value_changed)
                        self.button = QtWidgets.QPushButton(qta.icon("fa5s.home"), "").addToLayout()
                        self.button.setToolTip("reset view")
                        self.button.clicked.connect(lambda x: (self.view1.fitInView(), self.view2.fitInView()))
                        self.button2 = QtWidgets.QPushButton(qta.icon("fa.floppy-o"), "").addToLayout()
                        self.button2.setToolTip("save image")
                        self.button2.clicked.connect(self.export)
                    self.view1 = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
                    self.view1.setMinimumWidth(300)
                    self.pixmap1 = QtWidgets.QGraphicsPixmapItem(self.view1.origin)
                    self.scale1 = ModuleScaleBar(self, self.view1)

                    self.label2 = QtWidgets.QLabel("deformed").addToLayout()
                    self.view2 = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
                    # self.label2.setMinimumWidth(300)
                    self.pixmap2 = QtWidgets.QGraphicsPixmapItem(self.view2.origin)
                    self.scale2 = ModuleScaleBar(self, self.view2)

                    self.views = [self.view1, self.view2]
                    self.pixmaps = [self.pixmap1, self.pixmap2]

                    self.t_slider = QTimeSlider(connected=self.update_display).addToLayout()
                    self.tab.parent().t_slider = self.t_slider
                self.z_slider = QTimeSlider("z", self.z_slider_value_changed, "set z position", QtCore.Qt.Vertical).addToLayout()

        self.view1.link(self.view2)
        self.current_tab_selected = True
        self.setParameterMapping(None, {})

    def check_evaluated(self, result: Result) -> bool:
        return self.result is not None
    
    def check_available(self, result: Result) -> bool:
        if result is not None and result.stack is not None:
            return True            
        return False
      
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
            tifffile.imsave(new_path.parent / (new_path.stem + "_relaxed.tif"), self.result.stack[0][:, :, self.z_slider.value()])
            tifffile.imsave(new_path.parent / (new_path.stem + "_deformed.tif"), self.result.stack[1][:, :, self.z_slider.value()])
            imageio.mimsave(new_path.parent / (new_path.stem + ".gif"), [self.result.stack[0][:, :, self.z_slider.value()], self.result.stack[1][:, :, self.z_slider.value()]], fps=2)

    def update_display(self):   
        if self.check_available(self.result):
            self.scale1.setScale(self.result.stack[0].voxel_size)
            self.scale2.setScale(self.result.stack[1].voxel_size)
            self.z_slider.setRange(0, self.result.stack[0].shape[2] - 1)
            self.z_slider.setValue(self.result.stack[0].shape[2] // 2)
            self.z_slider_value_changed()

    def z_slider_value_changed(self):
        if self.result is not None:
            for i in range(2):
                self.views[i].setToolTip(f"stack\n{self.result.stack[self.t_slider.value()+i].description(self.z_slider.value())}")

                im = self.result.stack[self.t_slider.value()+i][:, :, self.z_slider.value()]
                if self.contrast_enhance.value():
                    im -= im.min()
                    im = (im.astype(np.float64)*255/im.max()).astype(np.uint8)
                self.pixmaps[i].setPixmap(QtGui.QPixmap(array2qimage(im)))
                self.views[i].setExtend(im.shape[0], im.shape[0])

            self.z_slider.setToolTip(f"set z position\ncurrent position {self.z_slider.value()}")

    def get_code(self) -> Tuple[str, str]:
        from saenopy.solver import common_start, common_end
        def filename_to_string(filename, insert="{z}"):
            if isinstance(filename, list):
                return str(Path(common_start(filename) + insert + common_end(filename)))
            return str(Path(filename))
        from saenopy.solver import getStacks
        import_code = "from saenopy.solver import getStacks\n"
        if self.result.time_delta is None:
            def code(stack1, stack2, output, voxel_size1):
                # load the relaxed and the contracted stack, {z} is the placeholder for the z stack
                results = getStacks([
                    stack1,
                    stack2,
                ], output, voxel_size=voxel_size1)
            data = dict(
                stack1=filename_to_string(self.result.stack[0].filename),
                stack2=filename_to_string(self.result.stack[1].filename),
                output=str(Path(self.result.output).parent),
                voxel_size1=self.result.stack[0].voxel_size,
            )
        else:
            def code(stack1, output, voxel_size1, time_delta1):
                # load the time series stack, {z} is the placeholder for the z stack, {t} is the placeholder for the time steps
                results = getStacks(stack1,
                                    output,
                                    voxel_size=voxel_size1, time_delta=time_delta1)
            stack_filenames = filename_to_string([filename_to_string(stack.filename) for stack in self.result.stack], insert="{t}")
            data = dict(
                stack1=stack_filenames,
                output=str(Path(self.result.output).parent),
                voxel_size1=self.result.stack[0].voxel_size,
                time_delta1=self.result.time_delta,
            )

        code_lines = inspect.getsource(code).split("\n")[1:]
        indent = len(code_lines[0]) - len(code_lines[0].lstrip())
        code = "\n".join(line[indent:] for line in code_lines)

        for key, value in data.items():
            if isinstance(value, str):
                if "\\" in value:
                    code = code.replace(key, "r'" + value + "'")
                else:
                    code = code.replace(key, "'" + value + "'")
            else:
                code = code.replace(key, str(value))
        return import_code, code

vtk_toolbars = []
class VTK_Toolbar(QtWidgets.QWidget):
    theme_values = [pv.themes.DefaultTheme(), pv.themes.ParaViewTheme(),
                                                          pv.themes.DarkTheme(), pv.themes.DocumentTheme()]
    def __init__(self, plotter, update_display, scalbar_type="deformation", center=False):
        super().__init__()
        self.plotter = plotter
        self.update_display = update_display
        vtk_toolbars.append(self)

        with QtShortCuts.QHBoxLayout(self) as layout0:
            layout0.setContentsMargins(0, 0, 0, 0)
            self.theme = QtShortCuts.QInputChoice(None, "Theme", value=self.theme_values[2],
                                                  values=self.theme_values,
                                                  value_names=["default", "paraview", "dark", "document"])

            self.auto_scale = QtShortCuts.QInputBool(None, "auto color", True, tooltip="Automatically choose the maximum for the color scale.")
            self.auto_scale.valueChanged.connect(self.scale_max_changed)
            self.scale_max = QtShortCuts.QInputString(None, "max color", 1e-6, type=float, tooltip="Set the maximum of the color scale.")
            self.scale_max.valueChanged.connect(self.scale_max_changed)
            self.use_nans = QtShortCuts.QInputBool(None, "nans", True, tooltip="Display nodes which do not have values associated as gray dots.")
            self.use_nans.valueChanged.connect(self.update_display)

            if center is True:
                self.use_center = QtShortCuts.QInputBool(None, "center", True,
                                                       tooltip="Display the center of the force field.")
                self.use_center.valueChanged.connect(self.update_display)

            self.theme.valueChanged.connect(lambda x: self.new_plotter(x))

            layout0.addStretch()
            self.button = QtWidgets.QPushButton(qta.icon("fa5s.home"), "").addToLayout()
            self.button.setToolTip("reset view")
            self.button.clicked.connect(lambda x: self.plotter.isometric_view())

            def save():
                if 1:
                    new_path = QtWidgets.QFileDialog.getSaveFileName(None, "Save Images", os.getcwd())
                    # if we got one, set it
                    if new_path:
                        if isinstance(new_path, tuple):
                            new_path = new_path[0]
                        else:
                            new_path = str(new_path)
                        print(new_path)
                        self.plotter.screenshot(new_path)
                    return
                outer_self = self

                class PlotDialog(QtWidgets.QDialog):
                    def __init__(self, parent):
                        super().__init__(parent)
                        with QtShortCuts.QVBoxLayout(self) as layout:
                            self.plotter = QtInteractor(self, theme=outer_self.plotter.theme)
                            layout.addWidget(self.plotter)
                            outer_self.update_display(self.plotter)
                            #showVectorField(self.plotter, outer_self.result.mesh_piv, "U_measured")
                            self.button2 = QtWidgets.QPushButton(qta.icon("fa.floppy-o"), "").addToLayout()
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
                        self.plotter.close()
                        self.close()

                    def close(self):
                        self.plotter.close()

                plot_diaolog = PlotDialog(self)
                plot_diaolog.show()

            self.button2 = QtWidgets.QPushButton(qta.icon("fa.floppy-o"), "").addToLayout()
            self.button2.setToolTip("save")
            self.button2.clicked.connect(save)

    def scale_max_changed(self):
        self.scale_max.setDisabled(self.auto_scale.value())
        scalebar_max = self.getScaleMax()
        print(scalebar_max, self.plotter.auto_value, type(self.plotter.auto_value))
        if scalebar_max is None:
            self.plotter.update_scalar_bar_range([0, self.plotter.auto_value])
        else:
            self.plotter.update_scalar_bar_range([0, scalebar_max])

    def getScaleMax(self):
        if self.auto_scale.value():
            return None
        return self.scale_max.value()

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
            result_view.setTheme(x)
        self.plotter.theme = x
        self.plotter.set_background(self.plotter._theme.background)
        print(self.plotter._theme.font.color)
        self.update_display()


cam_pos_initiaized = False
class DeformationDetector(PipelineModule):
    pipeline_name = "find deformations"

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with QtShortCuts.QVBoxLayout(self) as layout:
            layout.setContentsMargins(0, 0, 0, 0)
            with CheckAbleGroup(self, "find deformations (piv)").addToLayout() as self.group:

                with QtShortCuts.QVBoxLayout() as layout:
                    with QtShortCuts.QHBoxLayout():
                        # self.input_overlap = QtShortCuts.QInputNumber(None, "overlap", 0.6, step=0.1, value_changed=self.valueChanged,
                        #                                               tooltip="the fraction of a window size by which two adjacent windows overlap")
                        self.input_elementsize = QtShortCuts.QInputNumber(None, "piv element size", 15.0, step=1, value_changed=self.valueChanged,
                                                                      tooltip="the grid size for deformation detection")
                                    
                        self.input_win = QtShortCuts.QInputNumber(None, "window size", 30, value_changed=self.valueChanged, unit="μm",
                                                                  tooltip="the size of the volume to look for a match")
                    with QtShortCuts.QHBoxLayout():
                        self.input_signoise = QtShortCuts.QInputNumber(None, "signoise", 1.3, step=0.1,
                                                                       tooltip="the signal to noise ratio threshold value, values below are ignore")
                        self.input_driftcorrection = QtShortCuts.QInputBool(None, "driftcorrection", True,
                                                                            tooltip="remove the mean displacement to correct for a global drift")
                    self.label = QtWidgets.QLabel().addToLayout()
                    self.input_button = QtShortCuts.QPushButton(None, "detect deformations", self.start_process)

        with self.parent.tabs.createTab("PIV Deformations") as self.tab:
            with QtShortCuts.QVBoxLayout() as layout:
                self.label_tab = QtWidgets.QLabel("The deformations from the piv algorithm at every window where the crosscorrelation was evaluated.").addToLayout()

                self.plotter = QtInteractor(self)#, theme=pv.themes.DocumentTheme())
                self.tab.parent().plotter = self.plotter
                self.plotter.set_background("black")
                layout.addWidget(self.plotter.interactor)

                self.vtk_toolbar = VTK_Toolbar(self.plotter, self.update_display, "deformation").addToLayout()

                self.t_slider = QTimeSlider(connected=self.update_display).addToLayout()
                self.tab.parent().t_slider = self.t_slider

        self.setParameterMapping("piv_parameter", {
            "win_um": self.input_win,
            "elementsize": self.input_elementsize,
            #"fac_overlap": self.input_overlap,
            "signoise_filter": self.input_signoise,
            "drift_correction": self.input_driftcorrection,
        })

    def check_available(self, result: Result) -> bool:
        return result is not None and result.stack is not None and len(result.stack)

    def check_evaluated(self, result: Result) -> bool:
        return self.result is not None and self.result.mesh_piv is not None

    def update_display(self, *, plotter=None):
        global cam_pos_initiaized
        if plotter is None:
            plotter = self.plotter
        cam_pos = None
        if plotter.camera_position is not None and cam_pos_initiaized is True:
            cam_pos = self.plotter.camera_position
        cam_pos_initiaized = True
        plotter.interactor.setToolTip(str(self.result.piv_parameter)+f"\nNodes {self.result.mesh_piv[0].R.shape[0]}\nTets {self.result.mesh_piv[0].T.shape[0]}")
        M = self.result.mesh_piv[self.t_slider.value()]

        if M.hasNodeVar("U_measured"):
            showVectorField(plotter, M, M.getNodeVar("U_measured"), "U_measured", scalebar_max=self.vtk_toolbar.getScaleMax(), show_nan=self.vtk_toolbar.use_nans.value())

        if cam_pos is not None:
            plotter.camera_position = cam_pos

    def valueChanged(self):
        if self.check_available(self.result):
            voxel_size1 = self.result.stack[0].voxel_size
            stack_deformed = self.result.stack[0]
            overlap = 1 - (self.input_elementsize.value()/self.input_win.value()) 
            stack_size = np.array(stack_deformed.shape)*voxel_size1 - self.input_win.value()
            # self.label.setText(f"Deformation grid with {unit_size:.1f}μm elements.\nTotal region is {stack_size}.")
            self.label.setText(f"""Overlap between neighbouring windows (size={self.input_win.value()}µm or {(self.input_win.value()/np.array(voxel_size1)).astype(int)} px) is choosen \n to {int(overlap*100)}% for an elementsize of {self.input_elementsize.value():.1f}μm elements.\nTotal region is {stack_size}.""")
        else:
            self.label.setText("")

    def process(self, result: Result, params: dict):

        if not isinstance(result.mesh_piv, list):
            result.mesh_piv = [None]*(len(result.stack)-1)

        for i in range(len(result.stack)-1):
            p = ProcessSimple(getDeformation, (i, result, params), {})
            p.start()
            result.mesh_piv[i] = p.join()

            if 0:
                fini = False
                def finished():
                    nonlocal fini
                    result.mesh_piv[i] = process.result
                    fini = True
                process = QProcess(getDeformation, result, params)
                process.finished.connect(finished)
                while fini:
                    pass
                #result.mesh_piv[i] = saenopy.getDeformations.getDisplacementsFromStacks2(result.stack[i], result.stack[i+1],
                #                           params["win_um"], params["fac_overlap"], params["signoise_filter"],
                #                           params["drift_correction"])
        result.solver = None

    def get_code(self) -> Tuple[str, str]:
        from saenopy.getDeformations import getDisplacementsFromStacks2
        import_code = "from saenopy.getDeformations import getDisplacementsFromStacks2\n"
        def code(my_piv_params):
            # define the parameters for the piv deformation detection
            params = my_piv_params

            # iterate over all the results objects
            for result in results:
                # set the parameters
                result.piv_parameter = params
                # iterate over all stack pairs
                for i in range(len(result.stack) - 1):
                    # and calculate the displacement between them
                    result.mesh_piv[i] = getDisplacementsFromStacks2(result.stack[i], result.stack[i + 1],
                                                                     params["win_um"],
                                                                     1 - (params["elementsize"] / params["win_um"]),
                                                                     params["signoise_filter"],
                                                                     params["drift_correction"])
                # save the displacements
                result.save()
        data = {
            "my_piv_params": self.result.piv_parameter_tmp
        }

        code_lines = inspect.getsource(code).split("\n")[1:]
        indent = len(code_lines[0]) - len(code_lines[0].lstrip())
        code = "\n".join(line[indent:] for line in code_lines)

        for key, value in data.items():
            if isinstance(value, str):
                code = code.replace(key, "'" + value + "'")
            else:
                code = code.replace(key, str(value))
        return import_code, code

def getDeformation(progress, i, result, params):
    #import tqdm
    #t = tqdm.tqdm
    #n = tqdm.tqdm.__new__
    #tqdm.tqdm.__new__ = lambda cls, iter: progress.put(iter)

    mesh_piv = saenopy.getDeformations.getDisplacementsFromStacks2(result.stack[i], result.stack[i+1],
                                       params["win_um"],
                                       1-(params["elementsize"]/params["win_um"]), ## calculate overlap from specified element and win-size
                                       params["signoise_filter"],
                                       params["drift_correction"])
    return mesh_piv


class MeshCreator(PipelineModule):
    mesh_size = [200, 200, 200]
    pipeline_name = "interpolate mesh"

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with QtShortCuts.QVBoxLayout(self) as layout:
            layout.setContentsMargins(0, 0, 0, 0)
            with CheckAbleGroup(self, "interpolate mesh").addToLayout() as self.group:
                with QtShortCuts.QVBoxLayout() as layout:
                    self.input_reference = QtShortCuts.QInputChoice(None, "reference stack", "first", ["first", "median", "last"])
                    self.input_reference.setEnabled(False)
                    with QtShortCuts.QHBoxLayout():
                        with QtShortCuts.QVBoxLayout() as layout2:
                            self.input_element_size = QtShortCuts.QInputNumber(None, "mesh element size", 7, unit="μm")
                            #with QtShortCuts.QHBoxLayout() as layout2:
                            self.input_inner_region = QtShortCuts.QInputNumber(None, "inner region", 100, unit="μm")
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

        with self.parent.tabs.createTab("Target Deformations") as self.tab:
            with QtShortCuts.QVBoxLayout() as layout:
                self.label_tab = QtWidgets.QLabel("The deformations from the piv algorithm interpolated on the new mesh for regularisation.").addToLayout()

                self.plotter = QtInteractor(self)
                self.tab.parent().plotter = self.plotter
                self.plotter.set_background("black")
                layout.addWidget(self.plotter.interactor)
                self.vtk_toolbar = VTK_Toolbar(self.plotter, self.update_display).addToLayout()


                self.t_slider = QTimeSlider(connected=self.update_display).addToLayout()
                self.tab.parent().t_slider = self.t_slider

        self.setParameterMapping("interpolate_parameter", {
            "reference_stack": self.input_reference,
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

    def check_evaluated(self, result: Result) -> bool:
        return self.result is not None and self.result.solver is not None

    def update_display(self):
        global cam_pos_initiaized
        if self.result is not None and len(self.result.mesh_piv) > 2:
            self.input_reference.setEnabled(True)
        else:
            self.input_reference.setEnabled(False)
        if self.check_evaluated(self.result):
            cam_pos = None
            if self.plotter.camera_position is not None and cam_pos_initiaized is True:
                cam_pos = self.plotter.camera_position
            cam_pos_initiaized = True
            self.plotter.interactor.setToolTip(str(self.result.interpolate_parameter)+f"\nNodes {self.result.solver[self.t_slider.value()].R.shape[0]}\nTets {self.result.solver[self.t_slider.value()].T.shape[0]}")
            M = self.result.solver[self.t_slider.value()]
            showVectorField(self.plotter, M, M.U_target, "U_target", scalebar_max=self.vtk_toolbar.getScaleMax(), show_nan=self.vtk_toolbar.use_nans.value())
            if cam_pos is not None:
                self.plotter.camera_position = cam_pos
        else:
            self.plotter.interactor.setToolTip("")

    def process(self, result: Result, params: dict):
        solvers = []
        mode = self.input_reference.value()

        U = [M.getNodeVar("U_measured") for M in result.mesh_piv]
        # correct for the median position
        if len(U) > 2:
            xpos2 = np.cumsum(U, axis=0)  # mittlere position
            if mode == "first":
                xpos2 -= xpos2[0]
            elif mode == "median":
                xpos2 -= np.nanmedian(xpos2, axis=0)  # aktuelle abweichung von
            elif mode == "last":
                xpos2 -= xpos2[-1]
        else:
            xpos2 = U
        for i in range(len(result.mesh_piv)):
            M = result.mesh_piv[i]
            points, cells = saenopy.multigridHelper.getScaledMesh(params["element_size"]*1e-6,
                                          params["inner_region"]*1e-6,
                                          np.array([params["mesh_size_x"], params["mesh_size_y"],
                                                     params["mesh_size_z"]])*1e-6 / 2,
                                          [0, 0, 0], params["thinning_factor"])

            R = (M.R - np.min(M.R, axis=0)) - (np.max(M.R, axis=0) - np.min(M.R, axis=0)) / 2
            U_target = saenopy.getDeformations.interpolate_different_mesh(R, xpos2[i], points)

            border_idx = getNodesWithOneFace(cells)
            inside_mask = np.ones(points.shape[0], dtype=bool)
            inside_mask[border_idx] = False

            M = saenopy.Solver()
            M.setNodes(points)
            M.setTetrahedra(cells)
            M.setTargetDisplacements(U_target, inside_mask)

            solvers.append(M)
        result.solver = solvers

    def get_code(self) -> Tuple[str, str]:
        from saenopy.solver import substract_reference_state, interpolate_mesh
        import_code = "from saenopy.solver import substract_reference_state, interpolate_mesh\n"
        def code(my_mesh_params):
            # define the parameters to generate the solver mesh and interpolate the piv mesh onto it
            params = my_mesh_params

            # iterate over all the results objects
            for result in results:
                # correct for the reference state
                displacement_list = substract_reference_state(result.mesh_piv, params["reference_stack"])
                # set the parameters
                result.interpolate_parameter = params
                # iterate over all stack pairs
                for i in range(len(result.mesh_piv)):
                    # and create the interpolated solver mesh
                    result.solver[i] = interpolate_mesh(result.mesh_piv[i], displacement_list[i], params)
                # save the displacements
                result.save()
        data = {
            "my_mesh_params": self.result.interpolate_parameter_tmp,
        }

        code_lines = inspect.getsource(code).split("\n")[1:]
        indent = len(code_lines[0]) - len(code_lines[0].lstrip())
        code = "\n".join(line[indent:] for line in code_lines)

        for key, value in data.items():
            if isinstance(value, str):
                code = code.replace(key, "'" + value + "'")
            else:
                code = code.replace(key, str(value))
        return import_code, code


class Regularizer(PipelineModule):
    pipeline_name = "fit forces"
    iteration_finished = QtCore.Signal(object, object)

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

                    self.canvas = MatplotlibWidget(self).addToLayout()
                    NavigationToolbar(self.canvas, self).addToLayout()

        with self.parent.tabs.createTab("Forces") as self.tab:
            with QtShortCuts.QVBoxLayout() as layout:
                self.label_tab = QtWidgets.QLabel("The fitted regularized forces.").addToLayout()
                if 0:
                    self.canvas = MatplotlibWidget(self)
                    layout.addWidget(self.canvas)
                    layout.addWidget(NavigationToolbar(self.canvas, self))
                else:
                    pass #self.canvas = None

                self.plotter = QtInteractor(self)
                self.plotter.set_background("black")
                self.tab.parent().plotter = self.plotter
                layout.addWidget(self.plotter.interactor)

                self.vtk_toolbar = VTK_Toolbar(self.plotter, self.update_display, center=True).addToLayout()

                self.t_slider = QTimeSlider(connected=self.update_display).addToLayout()
                self.tab.parent().t_slider = self.t_slider

        self.setParameterMapping("solve_parameter", {
            "k": self.input_k,
            "d0": self.input_d0,
            "lambda_s": self.input_lamda_s,
            "ds": self.input_ds,
            "alpha": self.input_alpha,
            "stepper": self.input_stepper,
            "i_max": self.input_imax,
        })

        self.iteration_finished.connect(self.iteration_callback)
        self.iteration_finished.emit(None, np.ones([10, 3]))

    def check_available(self, result: Result):
        return result is not None and result.solver is not None

    def check_evaluated(self, result: Result) -> bool:
        if self.result is not None and self.result.solver is not None:
            relrec = getattr(self.result.solver[self.t_slider.value()], "relrec", None)
            if relrec is not None:
                return True
        return self.result is not None and self.result.solver is not None and getattr(self.result.solver[0], "regularisation_results", None) is not None

    def iteration_callback(self, result, relrec):
        if result is self.result:
            for i in range(self.parent.tabs.count()):
                if self.parent.tabs.widget(i) == self.tab.parent():
                    self.parent.tabs.setTabEnabled(i, self.check_evaluated(result))
            if self.canvas is not None:
                relrec = np.array(relrec).reshape(-1, 3)
                self.canvas.figure.axes[0].cla()
                self.canvas.figure.axes[0].semilogy(relrec[:, 0], label="total loss")
                self.canvas.figure.axes[0].semilogy(relrec[:, 1], ":", label="least squares loss")
                self.canvas.figure.axes[0].semilogy(relrec[:, 2], "--", label="regularize loss")
                self.canvas.figure.axes[0].legend()
                self.canvas.figure.axes[0].set_xlabel("iteration")
                self.canvas.figure.axes[0].set_ylabel("error")
                self.canvas.figure.axes[0].spines["top"].set_visible(False)
                self.canvas.figure.axes[0].spines["right"].set_visible(False)
                self.canvas.figure.tight_layout()
                self.canvas.draw()

    def process(self, result: Result, params: dict):
        for i in range(len(result.solver)):
            M = result.solver[i]

            def callback(M, relrec):
                self.iteration_finished.emit(result, relrec)

            M.setMaterialModel(saenopy.materials.SemiAffineFiberMaterial(
                               params["k"],
                               params["d0"],
                               params["lambda_s"],
                               params["ds"],
                               ))

            M.solve_regularized(stepper=params["stepper"], i_max=params["i_max"],
                                alpha=params["alpha"], callback=callback, verbose=True)

    def update_display(self):
        global cam_pos_initiaized
        if self.check_evaluated(self.result):
            cam_pos = None
            if self.plotter.camera_position is not None and cam_pos_initiaized is True:
                cam_pos = self.plotter.camera_position
            cam_pos_initiaized = True
            self.plotter.interactor.setToolTip(str(self.result.solve_parameter)+f"\nNodes {self.result.solver[self.t_slider.value()].R.shape[0]}\nTets {self.result.solver[self.t_slider.value()].T.shape[0]}")
            M = self.result.solver[self.t_slider.value()]
            center = None
            if self.vtk_toolbar.use_center.value() is True:
                center = M.getCenter(mode="Force")
            showVectorField(self.plotter, M, -M.f * M.reg_mask[:, None], "f", center=center, factor=0.5, scalebar_max=self.vtk_toolbar.getScaleMax(), show_nan=self.vtk_toolbar.use_nans.value())
            if cam_pos is not None:
                self.plotter.camera_position = cam_pos
            relrec = getattr(self.result.solver[self.t_slider.value()], "relrec", None)
            if relrec is None:
                relrec = self.result.solver[self.t_slider.value()].regularisation_results
            self.iteration_callback(self.result, relrec)
        else:
            self.plotter.interactor.setToolTip("")


class FittedMesh(PipelineModule):
    pipeline_name = "fit forces"
    iteration_finished = QtCore.Signal(object, object)

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with self.parent.tabs.createTab("Fitted Deformations") as self.tab:
            with QtShortCuts.QVBoxLayout() as layout:
                self.label_tab = QtWidgets.QLabel("The fitted mesh deformations.").addToLayout()
                if 0:
                    self.canvas = MatplotlibWidget(self)
                    layout.addWidget(self.canvas)
                    layout.addWidget(NavigationToolbar(self.canvas, self))
                else:
                    pass #self.canvas = None

                self.plotter = QtInteractor(self)
                self.plotter.set_background("black")
                self.tab.parent().plotter = self.plotter
                layout.addWidget(self.plotter.interactor)

                self.vtk_toolbar = VTK_Toolbar(self.plotter, self.update_display).addToLayout()

                self.t_slider = QTimeSlider(connected=self.update_display).addToLayout()
                self.tab.parent().t_slider = self.t_slider

        self.setParameterMapping(None, {})

    def check_available(self, result: Result):
        return result is not None and result.solver is not None

    def check_evaluated(self, result: Result) -> bool:
        return self.result is not None and self.result.solver is not None and getattr(self.result.solver[0], "regularisation_results", None) is not None

    def update_display(self):
        global cam_pos_initiaized
        if self.check_evaluated(self.result):
            cam_pos = None
            if self.plotter.camera_position is not None and cam_pos_initiaized is True:
                cam_pos = self.plotter.camera_position
            cam_pos_initiaized = True
            self.plotter.interactor.setToolTip(str(self.result.solve_parameter)+f"\nNodes {self.result.solver[self.t_slider.value()].R.shape[0]}\nTets {self.result.solver[self.t_slider.value()].T.shape[0]}")
            M = self.result.solver[self.t_slider.value()]
            showVectorField(self.plotter, M, M.U, "U", factor=0.1, scalebar_max=self.vtk_toolbar.getScaleMax(), show_nan=self.vtk_toolbar.use_nans.value())
            if cam_pos is not None:
                self.plotter.camera_position = cam_pos
        else:
            self.plotter.interactor.setToolTip("")


result_view = None
class ResultView(PipelineModule):

    def __init__(self, parent: "BatchEvaluate", layout):
        global result_view
        super().__init__(parent, layout)
        result_view = self

        with self.parent.tabs.createTab("View") as self.tab:
            with QtShortCuts.QVBoxLayout() as vlayout:
                with QtShortCuts.QHBoxLayout() as layout_vert_plot:
                    self.input_checks = {}
                    for name, dislay_name in {"U_target": "Target Deformations", "U": "Fitted Deformations", "f": "Forces", "stiffness": "Stiffness"}.items():
                        input = QtShortCuts.QInputBool(layout_vert_plot, dislay_name, name != "stiffness")
                        input.valueChanged.connect(self.replot)
                        self.input_checks[name] = input
                    layout_vert_plot.addStretch()
                    self.button_export = QtWidgets.QPushButton(qta.icon("fa.floppy-o"), "")
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

                self.t_slider = QTimeSlider(connected=self.update_display).addToLayout()
                self.tab.parent().t_slider = self.t_slider
        self.setParameterMapping(None, {})

    def check_evaluated(self, result: Result) -> bool:
        return self.result is not None and self.result.solver is not None and getattr(self.result.solver[0], "regularisation_results", None) is not None

    def update_display(self):
        if self.check_evaluated(self.result):
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
            self.point_cloud.point_data["f"] = -self.M.f * self.M.reg_mask[:, None]
            self.point_cloud["f_mag"] = np.linalg.norm(self.M.f * self.M.reg_mask[:, None], axis=1)
            self.point_cloud.point_data["U"] = self.M.U
            self.point_cloud["U_mag"] = np.linalg.norm(self.M.U, axis=1)
            self.point_cloud.point_data["U_target"] = self.M.U_target
            self.point_cloud["U_target_mag"] = np.linalg.norm(self.M.U_target, axis=1)
            nan_values = np.isnan(self.M.U_target[:, 0])
            self.point_cloud["U_target_mag"][nan_values] = 0

            self.point_cloud2 = None

            self.offset = np.min(self.M.R, axis=0)
            self.replot()
        else:
            self.plotter.interactor.setToolTip("")

    def calculateStiffness(self):
        self.point_cloud2 = pv.PolyData(np.mean(self.M.R[self.M.T], axis=1))
        from saenopy.materials import SemiAffineFiberMaterial
        # self.M.setMaterialModel(SemiAffineFiberMaterial(1645, 0.0008, 0.0075, 0.033), generate_lookup=False)
        if self.M.material_model is None:
            print("Warning using default material parameters")
            self.M.setMaterialModel(SemiAffineFiberMaterial(1449, 0.00215, 0.055, 0.032), generate_lookup=False)
        self.M._check_relax_ready()
        self.M._prepare_temporary_quantities()
        self.point_cloud2["stiffness"] = self.M.getMaxTetStiffness() / 6

    point_cloud = None

    theme = None
    def setTheme(self, x):
        self.theme = x
        self.current_result_plotted = False

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

        if self.theme is not None:
            self.plotter._theme = self.theme
            self.plotter.set_background(self.theme.background)

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
                                 scalar_bar_args=sargs2, opacity="linear")
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
                arrows2 = self.point_cloud.glyph(orient=name, scale=name + "_mag", \
                                                 # Automatically scale maximal force to 10% of axis length
                                                 factor=0.1 * norm_stack_size / np.nanmax(
                                                     np.linalg.norm(self.M.U_target, axis=1)))
                sargs2 = sargs.copy()
                sargs2["title"] = "Deformations (m)"
                plotter.add_mesh(arrows2, colormap='turbo', name="arrows2", scalar_bar_args=sargs2)  #

                # plot center if desired
                # plotter.add_points(np.array([self.M.getCenter(mode="deformation_target")]), color='w')

                plotter.update_scalar_bar_range(np.nanpercentile(self.point_cloud[name + "_mag"], [50, 99.9]))
                # plotter.update_scalar_bar_range([0,1.5e-6])
            elif name == "U":
                arrows3 = self.point_cloud.glyph(orient=name, scale=name + "_mag", \
                                                 # Automatically scale maximal force to 10% of axis length
                                                 factor=0.1 * norm_stack_size / np.nanmax(
                                                     np.linalg.norm(self.M.U, axis=1)))
                sargs2 = sargs.copy()
                sargs2["title"] = "Fitted Deformations [m]"
                plotter.add_mesh(arrows3, colormap='turbo', name="arrows3", scalar_bar_args=sargs2)
                plotter.update_scalar_bar_range(np.nanpercentile(self.point_cloud[name + "_mag"], [50, 99.9]))
                # plotter.update_scalar_bar_range([0,1.5e-6])

            # plot center points if desired
            # plotter.add_points(np.array([self.M.getCenter(mode="Deformation")]), color='w')
            # plotter.add_points(np.array([self.M.getCenter(mode="Force")]), color='r')

            if self.theme is not None:
                plotter.show_grid(color=self.theme.font.color)
            else:
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
    tab_changed = QtCore.Signal(object)
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
                            nonlocal old_tab, cam_pos
                            tab = self.tabs.currentWidget()
                            if old_tab is not None and getattr(old_tab, "plotter", None):
                                cam_pos = old_tab.plotter.camera_position
                            if cam_pos is not None and getattr(tab, "plotter", None):
                                tab.plotter.camera_position = cam_pos
                            if old_tab is not None:
                                tab.t_slider.setValue(old_tab.t_slider.value())
                            old_tab = tab
                            self.tab_changed.emit(tab)
                        self.tabs.currentChanged.connect(tab_changed)
                        pass
                with QtShortCuts.QVBoxLayout() as layout0:
                    layout0.parent().setMaximumWidth(420)
                    layout0.setContentsMargins(0, 0, 0, 0)
                    self.sub_module_stacks = StackDisplay(self, layout0)
                    self.sub_module_deformation = DeformationDetector(self, layout0)
                    self.sub_module_mesh = MeshCreator(self, layout0)
                    self.sub_module_fitted_mesh = FittedMesh(self, layout0)
                    self.sub_module_regularize = Regularizer(self, layout0)
                    self.sub_module_view = ResultView(self, layout0)
                    layout0.addStretch()
                    self.button_start_all = QtShortCuts.QPushButton(None, "run all", self.run_all)
                    self.button_code = QtShortCuts.QPushButton(None, "export code", self.generate_code)

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

    def generate_code(self):
        import_code = ""
        run_code = ""
        for module in [self.sub_module_stacks, self.sub_module_deformation, self.sub_module_mesh, self.sub_module_regularize]:
            code1, code2 = module.get_code()
            import_code += code1
            run_code += code2+"\n"
        run_code = import_code + "\n\n" + run_code
        print(run_code)

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
            
            url = url.toLocalFile()   #path()
            
            if url[0] == "/" and url[2] == ":":
                url = url[1:]
            if url.endswith(".npz"):
                urls = [url]
            else:
                urls = glob.glob(url+"/**/*.npz", recursive=True)
            for url in urls:
                data = Result.load(url)
                self.list.addData(data.output, True, data, mpl.colors.to_hex(f"gray"))
                #app.processEvents()
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
                    with QtShortCuts.QTabWidget(layout) as self.tabs:
                        with self.tabs.createTab("Pair Stacks") as self.tab:
                            self.outputText = QtShortCuts.QInputFolder(None, "output", settings=settings,
                                                                       settings_key="batch/wildcard2", allow_edit=True)
                            with QtShortCuts.QHBoxLayout() as layout3:
                                self.stack_relaxed = StackSelector(layout3, "relaxed")
                                self.stack_relaxed.glob_string_changed.connect(lambda x, y: (print("relaxed, y"), self.input_relaxed.setText(y)))
                                self.stack_deformed = StackSelector(layout3, "deformed", self.stack_relaxed)
                                self.stack_deformed.glob_string_changed.connect(lambda x, y: (print("deformed, y"),self.input_deformed.setText(y)))
                            with QtShortCuts.QHBoxLayout() as layout3:
                                self.input_relaxed = QtWidgets.QLineEdit().addToLayout()
                                self.input_deformed = QtWidgets.QLineEdit().addToLayout()
                            with QtShortCuts.QHBoxLayout() as layout3:
                                # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                                layout3.addStretch()
                                self.button_addList2 = QtShortCuts.QPushButton(None, "cancel", self.reject)
                                def accept():
                                    self.mode = "pair"
                                    if not self.stack_relaxed.validator():
                                        QtWidgets.QMessageBox.critical(self, "Deformation Detector", "Enter a valid voxel size for the relaxed stack.")
                                        return
                                    if not self.stack_deformed.validator():
                                        QtWidgets.QMessageBox.critical(self, "Deformation Detector", "Enter a valid voxel size for the deformed stack.")
                                        return
                                    self.accept()
                                self.button_addList1 = QtShortCuts.QPushButton(None, "ok", accept)
                        with self.tabs.createTab("Time Stacks") as self.tab2:
                            self.outputText2 = QtShortCuts.QInputFolder(None, "output", settings=settings,
                                                                       settings_key="batch/wildcard2", allow_edit=True)
                            with QtShortCuts.QHBoxLayout() as layout3:
                                self.stack_before2 = StackSelector(layout3, "time series", use_time=True)
                                self.stack_before2.glob_string_changed.connect(lambda x, y: self.input_relaxed2.setText(y))
                            with QtShortCuts.QHBoxLayout() as layout3:
                                self.input_relaxed2 = QtWidgets.QLineEdit().addToLayout()
                            with QtShortCuts.QHBoxLayout() as layout3:
                                # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                                layout3.addStretch()
                                self.button_addList3 = QtShortCuts.QPushButton(None, "cancel", self.reject)
                                def accept():
                                    if not self.stack_before2.validator():
                                        QtWidgets.QMessageBox.critical(self, "Deformation Detector", "Enter a valid voxel size for the stack.")
                                        return
                                    if not self.stack_before2.validator_time():
                                        QtWidgets.QMessageBox.critical(self, "Deformation Detector",
                                                                       "Enter a valid time delta.")
                                        return

                                    self.mode = "time"
                                    self.accept()
                                self.button_addList4 = QtShortCuts.QPushButton(None, "ok", accept)

                        with self.tabs.createTab("Existing Files") as self.tab3:
                            self.outputText3 = QtShortCuts.QInputFilename(None, "output", settings=settings, file_type="Results Files (*.npz)",
                                                                       settings_key="batch/wildcard_existing", allow_edit=True, existing=True)
                            with QtShortCuts.QHBoxLayout() as layout3:
                                # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                                layout3.addStretch()
                                self.button_addList6 = QtShortCuts.QPushButton(None, "cancel", self.reject)
                                def accept():
                                    self.mode = "existing"
                                    self.accept()
                                self.button_addList5 = QtShortCuts.QPushButton(None, "ok", accept)

        class FileExistsDialog(QtWidgets.QDialog):
            def __init__(self, parent, filename):
                super().__init__(parent)
                #self.setMinimumWidth(800)
                #self.setMinimumHeight(600)
                self.setWindowTitle("File Exists")
                with QtShortCuts.QVBoxLayout(self) as layout:
                    self.label = QtWidgets.QLabel(f"A file with the name {filename} already exists.").addToLayout()
                    with QtShortCuts.QHBoxLayout() as layout3:
                        layout3.addStretch()
                        self.use_for_all = QtShortCuts.QInputBool(None, "remember decision for all files", False)
                    with QtShortCuts.QHBoxLayout() as layout3:
                        # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                        layout3.addStretch()
                        self.button_addList2 = QtShortCuts.QPushButton(None, "cancel", self.reject)
                        def accept():
                            self.mode = "overwrite"
                            self.accept()
                        self.button_addList1 = QtShortCuts.QPushButton(None, "overwrite", accept)
                        def accept2():
                            self.mode = "read"
                            self.accept()
                        self.button_addList1 = QtShortCuts.QPushButton(None, "read", accept2)

        last_decision = None
        def do_overwrite(filename):
            nonlocal last_decision
            if last_decision is not None:
                return last_decision
            dialog = FileExistsDialog(self, filename)
            result = dialog.exec()
            if not result:
                return 0
            if dialog.use_for_all.value():
                last_decision = dialog.mode
            return dialog.mode

        #getStack
        dialog = AddFilesDialog(self)
        if not dialog.exec():
            return

        from saenopy.solver import getStacks
        if dialog.mode == "pair":
            results = getStacks(
                [dialog.input_relaxed.text(), dialog.input_deformed.text()],
                output_path=dialog.outputText.value(),
                voxel_size=dialog.stack_relaxed.getVoxelSize(),
                exist_overwrite_callback=do_overwrite,
            )
            for data in results:
                self.list.addData(data.output, True, data, mpl.colors.to_hex(f"gray"))
        elif dialog.mode == "time":
            results = getStacks(
                dialog.input_relaxed2.text(),
                output_path=dialog.outputText2.value(),
                voxel_size=dialog.stack_before2.getVoxelSize(),
                time_delta=dialog.stack_before2.getTimeDelta(),
                exist_overwrite_callback=do_overwrite,
            )
            for data in results:
                self.list.addData(data.output, True, data, mpl.colors.to_hex(f"gray"))
        elif dialog.mode == "existing":
            for file in glob.glob(dialog.outputText3.value(), recursive=True):
                data = Result.load(file)
                self.list.addData(data.output, True, data, mpl.colors.to_hex(f"gray"))
        self.update_icons()
        #import matplotlib as mpl
        #for fiber, cell, out in zip(fiber_list, cell_list, out_list):
        #    self.list.addData(fiber, True, [fiber, cell, out, {"segmention_thres": None, "seg_gaus1": None, "seg_gaus2": None}], mpl.colors.to_hex(f"gray"))

    def listSelected(self):
        if self.list.currentRow() is not None:
            pipe = self.data[self.list.currentRow()][2]
            self.set_current_result.emit(pipe)



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
                self.type = QtShortCuts.QInputChoice(None, "type", "strain_energy", ["strain_energy", "contractility", "polarity", "99_percentile_deformation", "99_percentile_force"])
                self.type.valueChanged.connect(self.replot)
                self.agg = QtShortCuts.QInputChoice(None, "aggregate", "mean",
                                                     ["mean", "max", "min", "median"])
                self.agg.valueChanged.connect(self.replot)

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
            print("y", self.list.data)
            for i, d in enumerate(list_new):
                self.list.setCurrentRow(i)
                self.list.listSelected()
                self.listSelected()
                self.list2.data = self.list.data[i][2]
                self.add_files([d0["path"] for d0 in d["paths"]])
                print("xxx", self.list.data)
                for ii, d0 in enumerate(d["paths"]):
                    self.list2.data[ii][1] = d0["selected"]
            print("x", self.list.data)

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
            if url.endswith(".npz"):
                urls += [url]
            else:
                urls += glob.glob(url + "/**/*.npz", recursive=True)
        self.add_files(urls)

    def add_files(self, urls):
        current_group = self.list2.data
        current_files = [d[0] for d in current_group]
        for file in urls:
            if file in current_files:
                print("File already in list", file)
                continue
            try:
                print("Add file", file)
                res = Result.load(file)
                res.resulting_data = []
                for i, M in enumerate(res.solver):
                    res.resulting_data.append({
                        "t": i*res.time_delta if res.time_delta else 0,
                        "strain_energy": M.E_glo,
                        "contractility": M.getContractility(center_mode="force"),
                        "polarity": M.getPolarity(),
                        "99_percentile_deformation": np.nanpercentile(np.linalg.norm(M.U_target[M.reg_mask], axis=1), 99),
                        "99_percentile_force": np.nanpercentile(np.linalg.norm(M.f[M.reg_mask], axis=1), 99),
                        "filename": file,
                    })
                res.resulting_data = pd.DataFrame(res.resulting_data)
                if self.list2.data is current_group:
                    self.list2.addData(file, True, res)
                    print("replot")
                    self.replot()
                app.processEvents()
            except FileNotFoundError:
                continue

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
                        return filename + "/**/*.npz"
                    self.inputText = QtShortCuts.QInputFolder(None, None, settings=settings, filename_checker=checker,
                                                                settings_key="batch_eval/analyse_force_wildcard", allow_edit=True)
                    with QtShortCuts.QHBoxLayout() as layout3:
                        # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                        layout3.addStretch()
                        self.button_addList = QtShortCuts.QPushButton(None, "cancel", self.reject)
                        self.button_addList = QtShortCuts.QPushButton(None, "ok", self.accept)

        dialog = AddFilesDialog(self)
        if not dialog.exec():
            return

        text = os.path.normpath(dialog.inputText.value())
        files = glob.glob(text, recursive=True)

        self.add_files(files)

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
                        res.resulting_data["group"] = name
                        results.append(res.resulting_data)
        res = pd.concat(results)
        #res["t"] = res["index"] * self.dt.value() / 60
        res.to_csv("tmp_pandas.csv")
        return res

    def replot(self):
        if self.current_plot_func is not None:
            self.current_plot_func()

    def barplot(self):
        for button in self.plot_buttons:
            button.setChecked(False)
        self.button_run3.setChecked(True)
        self.current_plot_func = self.barplot
        self.canvas.setActive()
        plt.cla()
        if self.type.value() == "strain_energy":
            mu_name = 'strain_energy'
            y_label = 'Strain Energy'
        elif self.type.value() == "contractility":
            mu_name = 'contractility'
            y_label = 'Contractility'
        elif self.type.value() == "polarity":
            mu_name = 'polarity'
            y_label = 'Polarity'
        elif self.type.value() == "99_percentile_deformation":
            mu_name = '99_percentile_deformation'
            y_label = 'Deformation'
        elif self.type.value() == "99_percentile_force":
            mu_name = '99_percentile_force'
            y_label = 'Force'

        # get all the data as a pandas dataframe
        res = self.getAllCurrentPandasData()

        # limit the dataframe to the comparison time
        print(res)
        print(res.columns)
        res0 = res.groupby("filename").agg("max")
        res = res.groupby("filename").agg(self.agg.value())
        res["group"] = res0["group"]
        #index = self.get_comparison_index()
        #res = res[res.index == index]
        print(res)
        print(res.columns, self.agg.value())

        code_data = [res, ["group", mu_name]]

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
        if self.type.value() == "strain_energy":
            mu_name = 'strain_energy'
            y_label = 'Strain Energy'
        elif self.type.value() == "contractility":
            mu_name = 'contractility'
            y_label = 'Contractility'
        elif self.type.value() == "polarity":
            mu_name = 'polarity'
            y_label = 'Polarity'
        elif self.type.value() == "99_percentile_deformation":
            mu_name = '99_percentile_deformation'
            y_label = 'Deformation'
        elif self.type.value() == "99_percentile_force":
            mu_name = '99_percentile_force'
            y_label = 'Force'

        self.canvas.setActive()
        plt.cla()
        res = self.getAllCurrentPandasData()

        code_data = [res, ["t", "group", mu_name, "filename"]]

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
        #return
        self.current_plot_func = self.run2
        if self.type.value() == "strain_energy":
            mu_name = 'strain_energy'
            y_label = 'Strain Energy'
        elif self.type.value() == "contractility":
            mu_name = 'contractility'
            y_label = 'Contractility'
        elif self.type.value() == "polarity":
            mu_name = 'polarity'
            y_label = 'Polarity'
        elif self.type.value() == "99_percentile_deformation":
            mu_name = '99_percentile_deformation'
            y_label = 'Deformation'
        elif self.type.value() == "99_percentile_force":
            mu_name = '99_percentile_force'
            y_label = 'Force'
        if 0:
            if self.type.value() == "Contractility":
                mu_name = 'Mean Contractility (µN)'
                std_name = 'St.dev. Contractility (µN)'
                y_label = 'Contractility (µN)'
            else:
                mu_name = 'Mean Pressure (Pa)'
                std_name = 'St.dev. Pressure (Pa)'
                y_label = 'Pressure (Pa)'

        try:
            res = self.data_folders[self.list.currentRow()][2][self.list2.currentRow()][2].resulting_data
        except IndexError:
            return

        #plt.figure(figsize=(6, 3))
        code_data = [res, ["t", mu_name]]

        #res["t"] = res.index * self.dt.value() / 60

        self.canvas.setActive()
        plt.cla()

        def plot(res, mu_name, y_label, plot_color):
            mu = res[mu_name]

            # plot time course of mean values
            p, = plt.plot(res.t, mu, lw=2, color=plot_color)

            # add grid
            plt.grid(True)
            # add labels
            plt.xlabel('Time (h)')
            plt.ylabel(y_label)
            plt.tight_layout()

            # show the plot
            self.canvas.draw()

        code = execute(plot, code_data[0][code_data[1]], mu_name=mu_name, y_label=y_label, plot_color=self.data_folders[self.list.currentRow()][3])

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



class MainWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # QSettings
        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

        self.setMinimumWidth(1400)
        self.setMinimumHeight(900)
        self.setWindowTitle("Saenopy Viewer")
        self.setWindowIcon(QtGui.QIcon("img/Icon.ico"))

        main_layout = QtWidgets.QHBoxLayout(self)

        with QtShortCuts.QTabWidget(main_layout) as self.tabs:
            """ """
            with self.tabs.createTab("Analyse Measurements") as v_layout:
                with QtShortCuts.QHBoxLayout() as h_layout:
                    # self.deformations = Deformation(h_layout, self)
                    self.deformations = BatchEvaluate(self)
                    h_layout.addWidget(self.deformations)
                    if 0:
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
            with self.tabs.createTab("Data Analysis") as v_layout:
                with QtShortCuts.QHBoxLayout() as h_layout:
                    # self.deformations = Deformation(h_layout, self)
                    self.deformations = PlottingWindow(self).addToLayout()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    if sys.platform.startswith('win'):
        import ctypes
        myappid = 'fabrylab.saenopy.master'  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    print(sys.argv)
    window = MainWindow()
    if len(sys.argv) >= 2:
        window.loadFile(sys.argv[1])
    window.show()
    sys.exit(app.exec_())
