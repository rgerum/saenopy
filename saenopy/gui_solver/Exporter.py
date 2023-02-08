import sys
import os

import pyvista
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
from saenopy import Result

from typing import List, Tuple

from .PipelineModule import PipelineModule
from .QTimeSlider import QTimeSlider
from .VTK_Toolbar import VTK_Toolbar
from .showVectorField import showVectorField, showVectorField2, getVectorFieldImage
from .DeformationDetector import CamPos

from skimage import exposure
from skimage.filters import meijering, sato, frangi, hessian
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                         denoise_wavelet, estimate_sigma)
from skimage.filters import threshold_yen
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib
from saenopy.gui.gui_classes import CheckAbleGroup, MatplotlibWidget, NavigationToolbar
from saenopy.gui_solver.FiberViewer import ChannelProperties, process_stack, join_stacks
import time
import datetime



def formatTimedelta(t: datetime.timedelta, fmt: str) -> str:
    sign = 1
    if t.total_seconds() < 0:
        sign = -1
        t = -t
    seconds = t.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    parts = {"d": t.days, "H": hours, "M": minutes, "S": seconds,
             "s": t.total_seconds(), "m": t.microseconds // 1000, "f": t.microseconds}

    max_level = None
    if fmt.find("%d") != -1:
        max_level = "d"
    elif fmt.find("%H") != -1:
        max_level = "H"
    elif fmt.find("%M") != -1:
        max_level = "M"
    elif fmt.find("%S") != -1:
        max_level = "S"
    elif fmt.find("%m") != -1:
        max_level = "m"
    elif fmt.find("%f") != -1:
        max_level = "f"

    fmt = fmt.replace("%d", str(parts["d"]))
    if max_level == "H":
        fmt = fmt.replace("%H", "%d" % (parts["H"] + parts["d"] * 24))
    else:
        fmt = fmt.replace("%H", "%02d" % parts["H"])
    if max_level == "M":
        fmt = fmt.replace("%M", "%2d" % (parts["M"] + parts["H"] * 60 + parts["d"] * 60 * 24))
    else:
        fmt = fmt.replace("%M", "%02d" % parts["M"])

    if max_level == "S":
        fmt = fmt.replace("%S", "%d" % parts["s"])
    else:
        fmt = fmt.replace("%S", "%02d" % parts["S"])

    if max_level == "m":
        fmt = fmt.replace("%m", "%3d" % (parts["m"] + parts["s"] * 1000))
    else:
        fmt = fmt.replace("%m", "%03d" % parts["m"])
    if max_level == "f":
        fmt = fmt.replace("%f", "%6d" % (parts["f"] + parts["s"] * 1000 * 1000))
    else:
        fmt = fmt.replace("%f", "%06d" % parts["f"])
    if sign == -1:
        for i in range(len(fmt)):
            if fmt[i] != " ":
                break
        if i == 0:
            fmt = "-" + fmt
        else:
            fmt = fmt[:i - 1] + "-" + fmt[i:]
    return fmt


class ExportViewer(PipelineModule):

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)
        self.export_window = QtWidgets.QWidget()

        if parent is None:
            class Null:
                pass
            self.parent = Null()
            self.parent.shared_properties = Null()
            self.parent.shared_properties.add_property = lambda *args: None
            self.parent.shared_properties.change_property = lambda *args: None
            #self.export_window.show()

        with QtShortCuts.QHBoxLayout(self.export_window) as layout:

            self.plotter = pyvista.Plotter(off_screen=True, multi_samples=4, line_smoothing=True)#QtInteractor(self, auto_update=False)  # , theme=pv.themes.DocumentTheme())
            #self.tab.parent().plotter = self.plotter
            self.plotter.set_background("black")

            self.view1 = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
            self.view1.setMinimumWidth(700)
            self.pixmap1 = QtWidgets.QGraphicsPixmapItem(self.view1.origin)
            self.view1.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            #self.plotter.setMinimumWidth(300)
            #layout.addWidget(self.plotter.interactor)

                #self.z_slider = QTimeSlider("z", self.z_slider_value_changed, "set z position",
                #                            QtCore.Qt.Vertical).addToLayout()
                #self.z_slider.t_slider.valueChanged.connect(
                #    lambda value: parent.shared_properties.change_property("z_slider", value, self))
                #parent.shared_properties.add_property("z_slider", self)\
            self.widget_settings = QtWidgets.QWidget().addToLayout()
            self.widget_settings.setMaximumWidth(700)
            self.widget_settings.setMinimumWidth(700)

            with QtShortCuts.QVBoxLayout(self.widget_settings) as layout:
                self.vtk_toolbar = VTK_Toolbar(self.plotter, self.update_display, shared_properties=self.parent.shared_properties)#.addToLayout()
                self.vtk_toolbar2 = VTK_Toolbar(self.plotter, self.update_display, center=True, shared_properties=self.parent.shared_properties)#.addToLayout()

                with QtShortCuts.QGroupBox(None, "image dimensions") as layout:
                    with QtShortCuts.QHBoxLayout() as layout:
                        self.input_width = QtShortCuts.QInputNumber(None, "width", 1024, float=False)
                        self.input_height = QtShortCuts.QInputNumber(None, "height", 768, float=False)

                        self.input_width.valueChanged.connect(self.update_display)
                        self.input_height.valueChanged.connect(self.update_display)

                with QtShortCuts.QGroupBox(None, "camera") as layout:
                    with QtShortCuts.QHBoxLayout() as layout:
                        self.input_elevation = QtShortCuts.QInputNumber(None, "elevation", 0, min=-360, max=360, use_slider=True)
                        self.input_azimuth = QtShortCuts.QInputNumber(None, "azimuth", 0, min=-360, max=360, use_slider=True)
                        #self.input_zoom = QtShortCuts.QInputNumber(None, "zoom", 1, min=0, max=2, use_slider=True, float=True)


                        self.input_elevation.valueChanged.connect(self.update_display)
                        self.input_azimuth.valueChanged.connect(self.update_display)
                        self.input_azimuth.valueChanged.connect(self.update_display)

                with QtShortCuts.QGroupBox(None, "general") as layout:
                    with QtShortCuts.QHBoxLayout() as layout:
                        self.vtk_toolbar.theme.addToLayout()
                        self.vtk_toolbar.show_grid.addToLayout()
                        self.vtk_toolbar.use_nans.addToLayout()
                        QtShortCuts.current_layout.addStretch()

                with QtShortCuts.QHBoxLayout() as layout:
                    self.input_arrows = QtShortCuts.QInputChoice(None, "arrows", "piv", values=["None", "piv", "target deformations", "fitted deformations", "fitted forces"])
                    self.input_arrows.valueChanged.connect(self.update_display)
                    QtShortCuts.current_layout.addStretch()

                with QtShortCuts.QHBoxLayout() as layout:
                    with QtShortCuts.QGroupBox(None, "deformation arrows") as layout:
                        with QtShortCuts.QHBoxLayout() as layout:
                            self.vtk_toolbar.auto_scale.addToLayout()
                            self.vtk_toolbar.scale_max.addToLayout()
                        self.vtk_toolbar.colormap_chooser.addToLayout()
                        self.vtk_toolbar.arrow_scale.addToLayout()
                        #QtShortCuts.current_layout.addStretch()

                    with QtShortCuts.QGroupBox(None, "force arrows") as layout:
                        with QtShortCuts.QHBoxLayout() as layout:
                            self.vtk_toolbar2.auto_scale.addToLayout()
                            self.vtk_toolbar2.scale_max.addToLayout()
                            self.vtk_toolbar2.use_center.addToLayout()
                        self.vtk_toolbar2.colormap_chooser.addToLayout()
                        self.vtk_toolbar2.arrow_scale.addToLayout()
                        #QtShortCuts.current_layout.addStretch()

                with QtShortCuts.QGroupBox(None, "stack image") as layout:
                    with QtShortCuts.QHBoxLayout() as layout:
                        self.vtk_toolbar.show_image.addToLayout()
                        self.vtk_toolbar.channel_select.addToLayout()
                        self.vtk_toolbar.button_z_proj.addToLayout()
                        self.vtk_toolbar.contrast_enhance.addToLayout()
                        self.vtk_toolbar.colormap_chooser2.addToLayout()
                        self.z_slider = QTimeSlider("z", self.update_display, "set z position").addToLayout()
                        #QtShortCuts.current_layout.addStretch()

                with QtShortCuts.QGroupBox(None, "fiber display") as layout:
                    with QtShortCuts.QVBoxLayout() as layout:
                        QtShortCuts.current_layout.setContentsMargins(0, 0, 0, 0)
                        self.input_cropx = QtShortCuts.QRangeSlider(None, "crop x", 0, 200)
                        self.input_cropy = QtShortCuts.QRangeSlider(None, "crop y", 0, 200)
                        self.input_cropz = QtShortCuts.QRangeSlider(None, "crop z", 0, 200)
                    with QtShortCuts.QHBoxLayout() as layout:
                        QtShortCuts.current_layout.setContentsMargins(0, 0, 0, 0)
                        self.channel0_properties = ChannelProperties().addToLayout()
                        self.channel0_properties.valueChanged.connect(self.update_display)
                        self.channel1_properties = ChannelProperties().addToLayout()
                        self.channel1_properties.valueChanged.connect(self.update_display)
                        self.channel0_properties.input_show.setValue(False)
                        self.channel1_properties.input_show.setValue(False)
                        pass
                    self.channel1_properties.input_cmap.setValue("Greens")
                    self.channel1_properties.input_sato.setValue(0)
                    self.channel1_properties.input_gauss.setValue(7)
                    #self.channel1_properties.input_percentile1.setValue(10)
                    #self.channel1_properties.input_percentile2.setValue(100)
                    self.input_thresh = QtShortCuts.QInputNumber(None, "thresh", 1, float=True, min=0, max=2, step=0.1)
                    self.input_thresh.valueChanged.connect(self.update_display)
                    #self.canvas = MatplotlibWidget(self).addToLayout()

                with QtShortCuts.QHBoxLayout():
                    self.t_slider = QTimeSlider(connected=self.update_display).addToLayout()
                    self.time_format = QtShortCuts.QInputString(None, "", "%d:%H:%M")
                    self.time_check = QtShortCuts.QInputBool(None, "", True)
                    self.time_size = QtShortCuts.QInputNumber(None, "", 18, float=False)

                    self.time_format.valueChanged.connect(self.update_display)
                    self.time_check.valueChanged.connect(self.update_display)
                    self.time_size.valueChanged.connect(self.update_display)

                with QtShortCuts.QHBoxLayout():
                    self.outputText3 = QtShortCuts.QInputFilename(None, "output",
                                                                  file_type="Image Files (*.png, *.jpf, *.tiff, *.avi, *.mp4, *.gif)",
                                                                  settings_key="export/exportfilename",
                                                                  allow_edit=True, existing=False)
                    self.input_fps = QtShortCuts.QInputNumber(None, "fps", 1)
                    self.button_export = QtShortCuts.QPushButton(None, "export", self.do_export)
                #self.tab.parent().t_slider = self.t_slider
                QtShortCuts.current_layout.addStretch()

        self.parameter_map = {
            "image": {
                "width": self.input_width,
                "height": self.input_height,
            },

            "camera": {
                "elevation": self.input_elevation,
                "azimuth": self.input_azimuth,
            },

            #"theme": self.vtk_toolbar.theme,
            "show_grid": self.vtk_toolbar.show_grid,
            "use_nans": self.vtk_toolbar.use_nans,

            "arrows": self.input_arrows,

            "deformation_arrows": {
                "autoscale": self.vtk_toolbar.auto_scale,
                "scale_max": self.vtk_toolbar.scale_max,
                "colormap": self.vtk_toolbar.colormap_chooser,
                "arrow_scale": self.vtk_toolbar.arrow_scale,
            },

            "force_arrows": {
                "autoscale": self.vtk_toolbar2.auto_scale,
                "scale_max": self.vtk_toolbar2.scale_max,
                "use_center": self.vtk_toolbar2.use_center,
                "colormap": self.vtk_toolbar.colormap_chooser,
                "arrow_scale": self.vtk_toolbar.arrow_scale,
            },

            "stack": {
                "image": self.vtk_toolbar.show_image,
                "channel": self.vtk_toolbar.channel_select,
                "z_proj": self.vtk_toolbar.button_z_proj,
                "contrast_enhance": self.vtk_toolbar.contrast_enhance,
                "colormap": self.vtk_toolbar.colormap_chooser2,
                "z": self.z_slider,
            },

            "crop": {
                "x": self.input_cropx,
                "y": self.input_cropy,
                "z": self.input_cropz,
            },

            "channel0": self.channel0_properties,
            "channel1": self.channel1_properties,

            "channel_thresh": self.input_thresh,

            "time": {
                "t": self.t_slider,
                "format": self.time_format,
                "display": self.time_check,
                "fontsize": self.time_size,
            },
        }
        self.setParameterMapping(None, {})
        self.no_update = False

    def do_export(self):
        filename_base = self.outputText3.value()
        writer = None
        if self.t_slider.t_slider.maximum() > 0:
            if filename_base.endswith(".png") or filename_base.endswith(".jpg"):
                if "{t}" not in filename_base:
                    filename_base = filename_base[:-4] + "_{t}" + filename_base[-4:]
            else:
                if filename_base.endswith(".gif"):
                    writer = imageio.get_writer(filename_base, fps=self.input_fps.value())
                if filename_base.endswith(".avi"):
                    writer = imageio.get_writer(filename_base, fps=self.input_fps.value(), format='FFMPEG', mode='I', codec='h264_x264')
                if filename_base.endswith(".mp4"):
                    writer = imageio.get_writer(filename_base, fps=self.input_fps.value(), format='FFMPEG', mode='I', codec='h264_x264')
        else:
            if not (filename_base.endswith(".png") or filename_base.endswith(".jpg")):
                raise ValueError("wrong file ending for a still image")
        for t in range(self.t_slider.t_slider.maximum()+1):
            self.t_slider.setValue(t)
            self.update_display()
            filename = self.outputText3.value()
            print("save", filename)
            if self.t_slider.t_slider.maximum() > 0:
                if writer is None:
                    print("save", filename_base.format(t=t))
                    imageio.imwrite(filename_base.format(t=t), self.im)
                else:
                    writer.append_data(self.im)
            else:
                imageio.imwrite(filename_base, self.im)
        if writer is not None:
            writer.close()


    def check_evaluated(self, result: Result) -> bool:
        return self.result is not None and result.stack is not None and len(result.stack) > 0

    def setResult(self, result: Result):
        if result and result.stack and result.stack[0]:
            self.z_slider.setRange(0, result.stack[0].shape[2] - 1)
            self.z_slider.setValue(result.stack[0].shape[2] // 2)

            if result.stack[0].channels:
                value = self.vtk_toolbar.channel_select.value()
                self.vtk_toolbar.channel_select.setValues(np.arange(len(result.stack[0].channels)), result.stack[0].channels)
                self.vtk_toolbar.channel_select.setValue(value)
                self.vtk_toolbar.channel_select.setVisible(True)
            else:
                self.vtk_toolbar.channel_select.setValue(0)
                self.vtk_toolbar.channel_select.setVisible(False)

            shape = result.stack[0].shape
            self.input_cropx.setRange(0, shape[1])
            self.input_cropx.setValue((shape[1] // 2 - 100, shape[1] // 2 + 100))
            self.input_cropy.setRange(0, shape[0])
            self.input_cropy.setValue((shape[0] // 2 - 100, shape[0] // 2 + 100))
            self.input_cropz.setRange(0, shape[2])
            self.input_cropz.setValue((shape[2] // 2 - 25, shape[2] // 2 + 25))

            if result.stack[0].shape[-1] == 1:
                self.channel1_properties.input_show.setValue(False)
                self.channel1_properties.setDisabled(True)
            else:
                self.channel1_properties.setDisabled(False)
        super().setResult(result)
        self.update_display()

    def get_parameters(self):
        def get_params(parameter_map):
            params = {}
            for name, widget in parameter_map.items():
                if isinstance(widget, dict):
                    params[name] = get_params(widget)
                else:
                    params[name] = widget.value()
            return params
        params = get_params(self.parameter_map)
        return params

    no_update = True
    def set_parameters(self, params):

        def set_params(params, parameter_map):
            for name, widget in parameter_map.items():
                if isinstance(widget, dict):
                    set_params(params[name], widget)
                else:
                    widget.setValue(params[name])

        self.no_update = True
        try:
            set_params(params, self.parameter_map)
        finally:
            self.no_update = False

    def update_display(self):
        if self.no_update:
            return
        #self.set_parameters(self.get_parameters())
        #if self.current_tab_selected is False:
        #    self.current_result_plotted = False
        #    return

        if self.check_evaluated(self.result):
            cam_pos = None
            if self.plotter.camera_position is not None and CamPos.cam_pos_initialized is True:
                cam_pos = self.plotter.camera_position
            CamPos.cam_pos_initialized = True
            #self.plotter.interactor.setToolTip(str(self.result.interpolate_parameter)+f"\nNodes {self.result.solver[self.t_slider.value()].R.shape[0]}\nTets {self.result.solver[self.t_slider.value()].T.shape[0]}")
            crops = []
            for widged in [self.input_cropx, self.input_cropy, self.input_cropz]:
                crops.extend(widged.value())
            t = self.t_slider.value()
            t_start = time.time()
            stack_data = None
            if self.channel0_properties.input_show.value():
                stack_data1 = process_stack(self.result.stack[t], 0,
                                            crops=crops,
                                            **self.channel0_properties.value())
                stack_data = stack_data1
                self.channel0_properties.sigmoid.p.set_im(stack_data1["original"])

            else:
                stack_data1 = None
            if self.channel1_properties.input_show.value():
                stack_data2 = process_stack(self.result.stack[t], 1,
                                            crops=crops,
                                            **self.channel1_properties.value())
                self.channel1_properties.sigmoid.p.set_im(stack_data2["original"])
                if stack_data1 is not None:
                    stack_data = join_stacks(stack_data1, stack_data2, self.input_thresh.value())
                else:
                    stack_data = stack_data2
            else:
                stack_data2 = None
            #stack_data = stack_data1
            if 0:#self.canvas is not None and stack_data is not None:
                self.canvas.figure.axes[0].cla()
                self.canvas.figure.axes[0].plot(np.linspace(0, 1, len(stack_data["opacity"])), stack_data["opacity"])
                self.canvas.figure.axes[0].spines["top"].set_visible(False)
                self.canvas.figure.axes[0].spines["right"].set_visible(False)
                self.canvas.figure.axes[0].set_xlim(0, 1)
                self.canvas.figure.axes[0].set_ylim(0, 1)
                self.canvas.draw()

            render = self.plotter.render
            self.plotter.render = lambda *args: None
            try:
                #if self.input_arrows = QtShortCuts.QInputChoice("arrows", "piv", values=["None", "piv", "target deformations", "fitted deformations", "fitted forces"])
                try:
                    display_image = getVectorFieldImage(self)
                except FileNotFoundError:
                    display_image = None
                if len(self.result.stack) and display_image is not None:
                    stack_shape = np.array(self.result.stack[0].shape[:3]) * np.array(
                        self.result.stack[0].voxel_size)
                else:
                    stack_shape = None
                M = None
                field = None
                center = None
                name = ""
                colormap = None
                factor = None
                scale_max = self.vtk_toolbar.getScaleMax()
                if self.input_arrows.value() == "piv":
                    if self.result is None:
                        M = None
                    else:
                        M = self.result.mesh_piv[self.t_slider.value()]

                    if M is not None:
                        if M.hasNodeVar("U_measured"):
                            #showVectorField2(self, M, "U_measured")
                            field = M.getNodeVar("U_measured")
                            factor = 0.1*self.vtk_toolbar.arrow_scale.value()
                            name = "U_measured"
                            colormap = self.vtk_toolbar.colormap_chooser.value()
                elif self.input_arrows.value() == "target deformations":
                    M = self.result.solver[self.t_slider.value()]
                    #showVectorField2(self, M, "U_target")
                    if M is not None:
                        field = M.U_target
                        factor = 0.1*self.vtk_toolbar.arrow_scale.value()
                        name = "U_target"
                        colormap = self.vtk_toolbar.colormap_chooser.value()
                elif self.input_arrows.value() == "fitted deformations":
                    M = self.result.solver[self.t_slider.value()]
                    if M is not None:
                        field = M.U
                        factor = 0.1*self.vtk_toolbar.arrow_scale.value()
                        name = "U"
                        colormap = self.vtk_toolbar.colormap_chooser.value()
                elif self.input_arrows.value() == "fitted forces":
                    M = self.result.solver[self.t_slider.value()]
                    if M is not None:
                        center = None
                        if self.vtk_toolbar2.use_center.value() is True:
                            center = M.getCenter(mode="Force")
                        field = -M.f * M.reg_mask[:, None]
                        factor = 0.15 * self.vtk_toolbar2.arrow_scale.value()
                        name = "f"
                        colormap = self.vtk_toolbar2.colormap_chooser.value()
                        scale_max = self.vtk_toolbar2.getScaleMax()
                    if 0:
                        display_image = getVectorFieldImage(self)
                        if len(self.result.stack):
                            stack_shape = np.array(self.result.stack[0].shape[:3]) * np.array(
                                self.result.stack[0].voxel_size)
                        else:
                            stack_shape = None
                        showVectorField(self.plotter, M, -M.f * M.reg_mask[:, None], "f", center=center,
                                        factor=0.15 * self.vtk_toolbar2.arrow_scale.value(),
                                        colormap=self.vtk_toolbar2.colormap_chooser.value(),
                                        colormap2=self.vtk_toolbar2.colormap_chooser2.value(),
                                        scalebar_max=self.vtk_toolbar2.getScaleMax(),
                                        show_nan=self.vtk_toolbar2.use_nans.value(),
                                        display_image=display_image, show_grid=self.vtk_toolbar2.show_grid.value(),
                                        stack_shape=stack_shape)
                showVectorField(self.plotter, M, field, name, center=center,
                                factor=factor,
                                colormap=colormap,
                                colormap2=self.vtk_toolbar.colormap_chooser2.value(),
                                scalebar_max=scale_max,
                                show_nan=self.vtk_toolbar.use_nans.value(),
                                display_image=display_image, show_grid=self.vtk_toolbar.show_grid.value(),
                                stack_shape=stack_shape)

                if stack_data is not None:
                    vol = self.plotter.add_volume(stack_data["data"], resolution=np.array(self.result.stack[0].voxel_size),
                                              cmap=stack_data["cmap"], opacity=stack_data["opacity"],
                                         blending="composite", name="fiber", render=False)  # 1.0*x
                    self.plotter.remove_scalar_bar("values")
                else:
                    self.plotter.remove_actor("fiber")
                print("plot time", f"{time.time()-t_start:.3f}s")
                self.plotter.reset_camera()
                self.plotter.camera.azimuth = self.input_azimuth.value()
                self.plotter.camera.elevation = self.input_elevation.value()
                if self.result is not None and self.result.time_delta is not None and self.time_check.value():
                    self.plotter.add_text(formatTimedelta(datetime.timedelta(seconds=float(self.t_slider.value()*self.result.time_delta)), self.time_format.value()), name="time_text", font_size=self.time_size.value())
                else:
                    self.plotter.remove_actor("time_text")
                #self.plotter.camera.zoom(self.input_zoom.value())
            finally:
                self.plotter.render = render
                #self.plotter.render()
                im = self.plotter.show(screenshot="tmp.png", return_img=True, auto_close=False, window_size=(self.input_width.value(), self.input_height.value()))

                im = plt.imread("tmp.png")
                im = (im * 255).astype(np.uint8)

                self.pixmap1.setPixmap(QtGui.QPixmap(array2qimage(im)))
                self.view1.setExtend(im.shape[1], im.shape[0])
                self.view1.fitInView()
                self.im = im

            if cam_pos is not None:
                self.plotter.camera_position = cam_pos
        else:
            pass
            #self.plotter.interactor.setToolTip("")


app = None
exporter = None
def render_image(params, result):
    global app, exporter
    if app is None:
        app = QtWidgets.QApplication([])
    if exporter is None:
        exporter = ExportViewer(None, None)
    exporter.set_parameters(params)
    exporter.setResult(result)
    exporter.set_parameters(params)
    exporter.update_display()
    return exporter.im


