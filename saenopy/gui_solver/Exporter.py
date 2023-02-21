import sys
import os

import pyvista
import qtawesome as qta
from qtpy import QtCore, QtWidgets, QtGui
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PIL import Image

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

class Writer:
    writer = None
    def __init__(self, filename, fps=None, qui_parent=None):
        self.filename_base = Path(filename)
        self.fps = fps
        self.gui_parent = qui_parent
        self.t = 0

    def __enter__(self):
        if str(self.filename_base) == "" or str(self.filename_base) == ".":
            if self.gui_parent is not None:
                QtWidgets.QMessageBox.critical(self.gui_parent, "Exporter", "Provide a valid filename.")
            raise FileExistsError
        if self.fps is None:
            if self.filename_base.suffix not in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif"]:
                if self.gui_parent is not None:
                    QtWidgets.QMessageBox.critical(self.gui_parent, "Exporter",
                                                   f"File extension '{self.filename_base.suffix}' is not supported for still images.")
                raise ValueError(f"File extension '{self.filename_base.suffix}' is not supported for still images.")
        else:
            if self.filename_base.suffix in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                if "{t}" not in str(self.filename_base):
                    self.filename_base = Path(str(self.filename_base.with_suffix("")) + "_{t}" + self.filename_base.suffix)
            else:
                if self.filename_base.suffix == ".gif":
                    self.writer = imageio.get_writer(self.filename_base, fps=self.fps, quantizer=2)
                elif self.filename_base.suffix == ".avi":
                    self.writer = imageio.get_writer(self.filename_base, fps=self.fps, format='FFMPEG',
                                                mode='I', codec='h264_x264')
                elif self.filename_base.suffix == ".mp4":
                    self.writer = imageio.get_writer(self.filename_base, fps=self.fps, format='FFMPEG',
                                                mode='I', codec='h264_x264')
                else:
                    if self.gui_parent is not None:
                        QtWidgets.QMessageBox.critical(self.gui_parent, "Exporter",
                                                   f"File extension '{self.filename_base.suffix}' is not supported.")
                    raise ValueError("invalid suffix")
        self.filename_base = str(self.filename_base)
        return self

    def write(self, im):
        if self.writer is None:
            print("save", self.filename_base.format(t=self.t))
            # JPEG does not support alpha channel.
            if self.filename_base.endswith(".jpg"):
                im = im[:, :, :3]
            imageio.imwrite(self.filename_base.format(t=self.t), im)
        else:
            print("save", self.filename_base, self.t)
            self.writer.append_data(im)
        self.t += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer is not None:
            self.writer.close()

def rotate(pos, angle):
    x, y = pos
    angle = np.deg2rad(angle)
    s, c = np.sin(angle), np.cos(angle)
    return x * c + y * s, -x * s + y * c


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
            self.pixmap1.sceneEvent = self.sceneEventFilter
            #self.view1.origin.installSceneEventFilter(self)

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

                with QtShortCuts.QHBoxLayout() as layout:
                    QtShortCuts.QPushButton(None, "save parameters", self.save_parameters)
                    QtShortCuts.QPushButton(None, "load parameters", self.load_parameters)
                    QtShortCuts.QPushButton(None, "copy to clipboard parameters", self.copy_parameters)

                with QtShortCuts.QHBoxLayout() as layout:
                    self.input_use2D = QtShortCuts.QInputBool(None, "", False, icon=["3D", "2D"], group=True)
                    self.input_use2D.valueChanged.connect(self.hide2D)
                    self.input_use2D.valueChanged.connect(self.update_display)
                    QtShortCuts.current_layout.addStretch()

                with QtShortCuts.QGroupBox(None, "image dimensions") as layout:
                    with QtShortCuts.QHBoxLayout() as layout:
                        self.input_width = QtShortCuts.QInputNumber(None, "width", 1024, float=False)
                        self.input_height = QtShortCuts.QInputNumber(None, "height", 768, float=False)


                        self.input_scale = QtShortCuts.QInputNumber(None, "scale", 1, min=0.1, max=10,
                                                                    use_slider=True, log_slider=True)
                        self.input_scale.valueChanged.connect(self.update_display)
                        self.input_antialiase = QtShortCuts.QInputBool(None, "antialiasing", True)
                        self.input_antialiase.valueChanged.connect(self.update_display)

                        self.input_logosize = QtShortCuts.QInputNumber(None, "logo size", 200, float=False, step=10)

                        self.input_width.valueChanged.connect(self.update_display)
                        self.input_height.valueChanged.connect(self.update_display)
                        self.input_logosize.valueChanged.connect(self.update_display)

                with QtShortCuts.QGroupBox(None, "camera") as (self.box_camera, _):
                    with QtShortCuts.QHBoxLayout() as layout:
                        self.input_elevation = QtShortCuts.QInputNumber(None, "elevation", 35, min=-90, max=90, use_slider=True, step=10)
                        self.input_azimuth = QtShortCuts.QInputNumber(None, "azimuth", 45, min=-180, max=180, use_slider=True, step=10)
                        self.input_distance = QtShortCuts.QInputNumber(None, "distance", 0, min=0, float=False, step=100)

                        self.input_elevation.valueChanged.connect(self.render_view)
                        self.input_azimuth.valueChanged.connect(self.render_view)
                        self.input_distance.valueChanged.connect(self.render_view)

                        def reset():
                            self.plotter.camera_position = "yz"
                            distance = self.plotter.camera.position[0]
                            self.input_distance.setValue(distance)
                            self.input_elevation.setValue(35)
                            self.input_azimuth.setValue(45)
                            self.input_roll.setValue(0)
                            self.render_view()

                        QtShortCuts.QPushButton(None, "", connect=reset, icon=qta.icon("fa5s.home"), tooltip="reset view")
                    with QtShortCuts.QHBoxLayout() as layout:
                        self.input_offset_x = QtShortCuts.QInputNumber(None, "offset x", 0, float=False, step=10)
                        self.input_offset_y = QtShortCuts.QInputNumber(None, "offset y", 0, float=False, step=10)
                        self.input_roll = QtShortCuts.QInputNumber(None, "roll", 0, float=False, step=10)
                        self.input_offset_x.valueChanged.connect(self.render_view)
                        self.input_offset_y.valueChanged.connect(self.render_view)
                        self.input_roll.valueChanged.connect(self.render_view)
                with QtShortCuts.QGroupBox(None, "general") as layout:
                    with QtShortCuts.QHBoxLayout() as layout:
                        self.vtk_toolbar.theme.addToLayout()
                        self.vtk_toolbar.show_grid.addToLayout()
                        self.vtk_toolbar.use_nans.addToLayout()
                        self.input_reference_stack = QtShortCuts.QInputBool(None, "show reference stack", False)
                        self.input_reference_stack.valueChanged.connect(self.update_display)
                        QtShortCuts.current_layout.addStretch()

                with QtShortCuts.QHBoxLayout() as layout:
                    self.input_arrows = QtShortCuts.QInputChoice(None, "arrows", "piv", values=["None", "piv", "target deformations", "fitted deformations", "fitted forces"])
                    self.input_arrows.valueChanged.connect(self.update_display)
                    self.input_arrows.valueChanged.connect(self.hide_arrow)
                    self.input_average_range = QtShortCuts.QInputNumber(None, "averaging z thickness", min=0, max=0,
                                                                        step=10)
                    self.input_average_range.valueChanged.connect(self.update_display)
                    QtShortCuts.current_layout.addStretch()

                with QtShortCuts.QHBoxLayout() as layout:
                    with QtShortCuts.QGroupBox(None, "deformation arrows") as (self.box_deformation_arrows, _):
                        with QtShortCuts.QHBoxLayout() as layout:
                            self.vtk_toolbar.auto_scale.addToLayout()
                            self.vtk_toolbar.scale_max.addToLayout()
                            self.input_arrow_opacity = QtShortCuts.QInputNumber(None, "opacity", 1, min=0, max=1, float=True, step=0.1)
                            self.input_arrow_opacity.valueChanged.connect(self.update_display)
                            self.input_arrow_skip = QtShortCuts.QInputNumber(None, "skip", 1, min=1, max=10, float=False)
                            self.input_arrow_skip.valueChanged.connect(self.update_display)
                        self.vtk_toolbar.colormap_chooser.addToLayout()

                        self.vtk_toolbar.arrow_scale.addToLayout()

                        #QtShortCuts.current_layout.addStretch()

                    with QtShortCuts.QGroupBox(None, "force arrows") as (self.box_force_arrows, _):
                        with QtShortCuts.QHBoxLayout() as layout:
                            self.vtk_toolbar2.auto_scale.addToLayout()
                            self.vtk_toolbar2.scale_max.addToLayout()
                            self.vtk_toolbar2.use_center.addToLayout()
                            self.input_arrow_opacity2 = QtShortCuts.QInputNumber(None, "opacity", 1, min=0, max=1,
                                                                                float=True, step=0.1)
                            self.input_arrow_opacity2.valueChanged.connect(self.update_display)
                            self.input_arrow_skip2 = QtShortCuts.QInputNumber(None, "skip", 1, min=1, max=10,
                                                                             float=False)
                            self.input_arrow_skip2.valueChanged.connect(self.update_display)
                        self.vtk_toolbar2.colormap_chooser.addToLayout()
                        self.vtk_toolbar2.arrow_scale.addToLayout()
                        #QtShortCuts.current_layout.addStretch()

                with QtShortCuts.QGroupBox(None, "stack image") as layout:
                    with QtShortCuts.QHBoxLayout() as layout:
                        self.vtk_toolbar.show_image.addToLayout()
                        self.vtk_toolbar.show_image.valueChanged.connect(self.hide_stack_image)
                        self.input_use2D.valueChanged.connect(self.hide_stack_image)
                        self.vtk_toolbar.channel_select.addToLayout()
                        self.vtk_toolbar.button_z_proj.addToLayout()
                        self.vtk_toolbar.contrast_enhance.addToLayout()
                        self.vtk_toolbar.colormap_chooser2.addToLayout()
                        self.z_slider = QTimeSlider("z", self.update_display, "set z position").addToLayout()
                        #QtShortCuts.current_layout.addStretch()


                with QtShortCuts.QGroupBox(None, "scale bar") as (self.box_scalebar, _):
                    with QtShortCuts.QHBoxLayout() as layout:
                        self.input_scalebar_um = QtShortCuts.QInputNumber(None, "length", 0, min=0, max=10000)
                        self.input_scalebar_um.valueChanged.connect(self.update_display)
                        self.input_scalebar_width = QtShortCuts.QInputNumber(None, "width", 5, min=0, max=100)
                        self.input_scalebar_width.valueChanged.connect(self.update_display)
                        self.input_scalebar_xpos = QtShortCuts.QInputNumber(None, "xpos", 15, min=0, max=100)
                        self.input_scalebar_xpos.valueChanged.connect(self.update_display)
                        self.input_scalebar_ypos = QtShortCuts.QInputNumber(None, "ypos", 10, min=0, max=100)
                        self.input_scalebar_ypos.valueChanged.connect(self.update_display)
                        self.input_scalebar_fontsize = QtShortCuts.QInputNumber(None, "fontsize", 18, min=0, max=100)
                        self.input_scalebar_fontsize.valueChanged.connect(self.update_display)
                        self.input_scalebar_fontsize.addToLayout()

                with QtShortCuts.QGroupBox(None, "2D arrrows") as (self.box_2darrows, _):
                    with QtShortCuts.QHBoxLayout() as layout:
                        self.input_2darrow_width = QtShortCuts.QInputNumber(None, "width", 2, min=0, max=100)
                        self.input_2darrow_width.valueChanged.connect(self.update_display)
                        self.input_2darrow_headlength = QtShortCuts.QInputNumber(None, "head length", 5, min=0, max=100)
                        self.input_2darrow_headlength.valueChanged.connect(self.update_display)
                        self.input_2darrow_headwidth = QtShortCuts.QInputNumber(None, "head width", 5, min=0, max=100)
                        self.input_2darrow_headwidth.valueChanged.connect(self.update_display)

                with QtShortCuts.QGroupBox(None, "fiber display") as (self.box_fiberdisplay, _):
                    with QtShortCuts.QVBoxLayout() as layout:
                        QtShortCuts.current_layout.setContentsMargins(0, 0, 0, 0)
                        self.input_cropx = QtShortCuts.QRangeSlider(None, "crop x", 0, 200)
                        self.input_cropy = QtShortCuts.QRangeSlider(None, "crop y", 0, 200)
                        self.input_cropz = QtShortCuts.QRangeSlider(None, "crop z", 0, 200)
                        self.input_cropx.editingFinished.connect(self.update_display)
                        self.input_cropy.editingFinished.connect(self.update_display)
                        self.input_cropz.editingFinished.connect(self.update_display)
                    with QtShortCuts.QHBoxLayout() as layout:
                        QtShortCuts.current_layout.setContentsMargins(0, 0, 0, 0)
                        self.channel0_properties = ChannelProperties().addToLayout()
                        self.channel0_properties.valueChanged.connect(self.update_display)
                        self.channel1_properties = ChannelProperties().addToLayout()
                        self.channel1_properties.valueChanged.connect(self.update_display)
                        self.channel0_properties.input_show.setValue(False)
                        self.channel0_properties.checkDisabled()
                        self.channel1_properties.input_show.setValue(False)
                        self.channel1_properties.checkDisabled()

                        self.channel0_properties.input_show.valueChanged.connect(self.hide_fiber)
                        self.channel1_properties.input_show.valueChanged.connect(self.hide_fiber)
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
                with QtShortCuts.QHBoxLayout():
                    self.time_check = QtShortCuts.QInputBool(None, "show timestamp", True)
                    self.time_format = QtShortCuts.QInputString(None, "", "%d:%H:%M")
                    self.time_start = QtShortCuts.QInputNumber(None, "start (s)", 0)
                    self.time_size = QtShortCuts.QInputNumber(None, "", 18, float=False)

                    self.time_format.valueChanged.connect(self.update_display)
                    self.time_start.valueChanged.connect(self.update_display)
                    self.time_check.valueChanged.connect(self.update_display)
                    self.time_check.valueChanged.connect(self.hide_timestamp)
                    self.time_size.valueChanged.connect(self.update_display)

                with QtShortCuts.QHBoxLayout():
                    self.outputText3 = QtShortCuts.QInputFilename(None, "output",
                                                                  file_type="Image Files (*.png, *.jpf, *.tif, *.avi, *.mp4, *.gif)",
                                                                  settings_key="export/exportfilename",
                                                                  allow_edit=True, existing=False)
                    self.button_export = QtShortCuts.QPushButton(None, "export single image", self.do_export)
                with QtShortCuts.QHBoxLayout():
                    with QtShortCuts.QVBoxLayout():
                        self.button_export_time = QtShortCuts.QPushButton(None, "export time", self.do_export_time)
                        with QtShortCuts.QHBoxLayout():
                            self.time_fps = QtShortCuts.QInputNumber(None, "fps", 1)
                            self.time_steps = QtShortCuts.QInputNumber(None, "steps", 1, float=False)
                    with QtShortCuts.QVBoxLayout():
                        self.button_export_reference = QtShortCuts.QPushButton(None, "export state/reference", self.do_export_reference)
                        with QtShortCuts.QHBoxLayout():
                            self.reference_fps = QtShortCuts.QInputNumber(None, "fps", 1)
                    with QtShortCuts.QVBoxLayout():
                        self.button_export_rotate = QtShortCuts.QPushButton(None, "export rotate", self.do_export_rotate)
                        with QtShortCuts.QHBoxLayout():
                            self.rotate_fps = QtShortCuts.QInputNumber(None, "fps", 10)
                            self.rotate_steps = QtShortCuts.QInputNumber(None, "steps", 5, float=False)
                    with QtShortCuts.QVBoxLayout():
                        self.button_export_zscan = QtShortCuts.QPushButton(None, "export z-scan", self.do_export_zscan)
                        with QtShortCuts.QHBoxLayout():
                            self.zscan_fps = QtShortCuts.QInputNumber(None, "fps", 30)
                            self.zscan_steps = QtShortCuts.QInputNumber(None, "steps", 1, float=False)
                #self.tab.parent().t_slider = self.t_slider
                self.render_progress = QtWidgets.QProgressBar().addToLayout()
                self.render_progress.setRange(0, 100)
                QtShortCuts.current_layout.addStretch()

        self.parameter_map = {
            "image": {
                "width": self.input_width,
                "height": self.input_height,
                "logo_size": self.input_logosize,
            },

            "camera": {
                "elevation": self.input_elevation,
                "azimuth": self.input_azimuth,
                "distance": self.input_distance,
                "offset_x": self.input_offset_x,
                "offset_y": self.input_offset_y,
                "roll": self.input_roll,
            },

            "theme": self.vtk_toolbar.theme,
            "show_grid": self.vtk_toolbar.show_grid,
            "use_nans": self.vtk_toolbar.use_nans,

            "arrows": self.input_arrows,

            "deformation_arrows": {
                "autoscale": self.vtk_toolbar.auto_scale,
                "scale_max": self.vtk_toolbar.scale_max,
                "colormap": self.vtk_toolbar.colormap_chooser,
                "arrow_scale": self.vtk_toolbar.arrow_scale,
                "arrow_opacity": self.input_arrow_opacity,
                "skip": self.input_arrow_skip,
            },

            "force_arrows": {
                "autoscale": self.vtk_toolbar2.auto_scale,
                "scale_max": self.vtk_toolbar2.scale_max,
                "use_center": self.vtk_toolbar2.use_center,
                "colormap": self.vtk_toolbar.colormap_chooser,
                "arrow_scale": self.vtk_toolbar.arrow_scale,
                "arrow_opacity": self.input_arrow_opacity2,
                "skip": self.input_arrow_skip2,
            },

            "stack": {
                "image": self.vtk_toolbar.show_image,
                "channel": self.vtk_toolbar.channel_select,
                "z_proj": self.vtk_toolbar.button_z_proj,
                "contrast_enhance": self.vtk_toolbar.contrast_enhance_values,
                "colormap": self.vtk_toolbar.colormap_chooser2,
                "z": self.z_slider,
                "use_reference_stack": self.input_reference_stack,
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
                "start": self.time_start,
                "display": self.time_check,
                "fontsize": self.time_size,
            },
        }
        self.setParameterMapping(None, {})
        self.no_update = False
        self.hide2D()
        self.hide_fiber()
        self.hide_arrow()
        self.hide_stack_image()

    def hide2D(self):
        is2D = self.input_use2D.value()
        is3D = not self.input_use2D.value()
        self.input_width.setVisible(is3D)
        self.input_height.setVisible(is3D)

        self.input_scale.setVisible(is2D)
        self.input_antialiase.setVisible(is2D)

        self.box_camera.setVisible(is3D)

        self.vtk_toolbar.theme.setVisible(is3D)
        self.vtk_toolbar.show_grid.setVisible(is3D)
        self.vtk_toolbar.use_nans.setVisible(is3D)

        self.input_average_range.setVisible(is2D)
        #self.input_arrow_skip.setVisible(is2D)

        self.vtk_toolbar.show_image.setVisible(is3D)

        self.box_scalebar.setVisible(is2D)
        self.box_scalebar.setVisible(is2D)
        self.box_2darrows.setVisible(is2D)

        self.box_fiberdisplay.setVisible(is3D)

        self.button_export_rotate.setEnabled(is3D)
        self.rotate_fps.setEnabled(is3D)
        self.rotate_steps.setEnabled(is3D)

    def hide_timestamp(self):
        isTimeAvailable = self.result is not None and self.result.time_delta is not None
        self.time_check.setEnabled(isTimeAvailable)
        self.button_export_time.setEnabled(isTimeAvailable)
        self.time_fps.setEnabled(isTimeAvailable)
        self.time_steps.setEnabled(isTimeAvailable)
        isTime = self.time_check.value() and self.result is not None and self.result.time_delta is not None
        self.time_format.setEnabled(isTime)
        self.time_start.setEnabled(isTime)
        self.time_size.setEnabled(isTime)

    def hide_fiber(self):
        isActive = self.channel0_properties.input_show.value() or self.channel1_properties.input_show.value()
        self.input_cropx.setEnabled(isActive)
        self.input_cropy.setEnabled(isActive)
        self.input_cropz.setEnabled(isActive)
        isActiveBoth = self.channel0_properties.input_show.value() and self.channel1_properties.input_show.value()
        self.input_thresh.setEnabled(isActiveBoth)

    def hide_arrow(self):
        isDeformation = self.input_arrows.value() in ["piv", "target deformations", "fitted deformations"]
        isForce = self.input_arrows.value() in ["fitted forces"]

        self.box_deformation_arrows.setEnabled(isDeformation)
        self.box_force_arrows.setEnabled(isForce)

    def hide_stack_image(self):
        isActive = self.vtk_toolbar.show_image.value() != 0 or self.input_use2D.value()
        self.vtk_toolbar.channel_select.setEnabled(isActive)
        self.vtk_toolbar.button_z_proj.setEnabled(isActive)
        self.vtk_toolbar.contrast_enhance.setEnabled(isActive)
        self.vtk_toolbar.colormap_chooser2.setEnabled(isActive)
        self.z_slider.setEnabled(isActive)

    def save_parameters(self):
        new_path = QtWidgets.QFileDialog.getSaveFileName(None, "Save Parameters", os.getcwd(), "JSON File (*.json)")
        if new_path:
            if isinstance(new_path, tuple):
                new_path = new_path[0]
            else:
                new_path = str(new_path)
            if not new_path.endswith(".json"):
                new_path += ".json"
            import json
            with open(new_path, "w") as fp:
                json.dump(self.get_parameters(), fp, indent=2)

    def load_parameters(self):
        new_path = QtWidgets.QFileDialog.getOpenFileName(None, "Load Parameters", os.getcwd(), "JSON File (*.json)")
        if new_path:
            if isinstance(new_path, tuple):
                new_path = new_path[0]
            else:
                new_path = str(new_path)
            import json
            with open(new_path, "r") as fp:
                self.set_parameters(json.load(fp))
            self.update_display()

    def copy_parameters(self):
        text = repr(self.get_parameters())
        cb = QtGui.QGuiApplication.clipboard()
        cb.setText(text, mode=cb.Clipboard)

    def progress_iterator(self, iter):
        self.render_progress.setRange(0, len(iter))
        self.render_progress.setValue(0)
        for i, value in enumerate(iter):
            yield value
            self.render_progress.setValue(i+1)

    def do_export(self):
        with Writer(self.outputText3.value(), None, self) as writer:
            self.update_display()
            writer.write(self.im)

    def do_export_time(self):
        with Writer(self.outputText3.value(), self.time_fps.value(), self) as writer:
            for t in self.progress_iterator(range(0, self.t_slider.t_slider.maximum() + 1, self.time_steps.value())):
                self.t_slider.setValue(t)
                self.update_display()
                writer.write(self.im)

    def do_export_reference(self):
        with Writer(self.outputText3.value(), self.reference_fps.value(), self) as writer:
            for ref in self.progress_iterator([False, True]):
                self.input_reference_stack.setValue(ref)
                self.update_display()
                writer.write(self.im)

    def do_export_rotate(self):
        with Writer(self.outputText3.value(), self.rotate_fps.value(), self) as writer:
            azimuth = 45
            for t in self.progress_iterator(range(azimuth, azimuth+360, self.rotate_steps.value())):
                while t > 180:
                    t -= 360
                self.input_azimuth.setValue(t)
                self.render_view()
                writer.write(self.im)

    def do_export_zscan(self):
        with Writer(self.outputText3.value(), self.zscan_fps.value(), self) as writer:
            for t in self.progress_iterator(range(0, self.z_slider.t_slider.maximum() + 1, self.zscan_steps.value())):
                self.z_slider.setValue(t)
                self.update_display()
                writer.write(self.im)

    def check_evaluated(self, result: Result) -> bool:
        return self.result is not None and result.stack is not None and len(result.stack) > 0

    def setResult(self, result: Result, no_update_display=False):
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

            self.input_average_range.setRange(0, shape[2]*result.stack[0].voxel_size[2])
            self.input_average_range.setValue(10)

        super().setResult(result)
        self.hide_timestamp()
        if not no_update_display:
            self.update_display()

    def get_parameters(self):
        def get_params(parameter_map):
            params = {}
            for name, widget in parameter_map.items():
                if isinstance(widget, dict):
                    params[name] = get_params(widget)
                else:
                    if isinstance(widget, QtShortCuts.QInputChoice) and widget.value_names:
                        params[name] = widget.valueName()
                    else:
                        params[name] = widget.value()
            return params
        params = get_params(self.parameter_map)
        return params

    no_update = True
    def set_parameters(self, params):

        def set_params(params, parameter_map):
            for name, widget in parameter_map.items():
                if name not in params:
                    continue
                if isinstance(widget, dict):
                    set_params(params[name], widget)
                else:
                    widget.setValue(params[name])

        self.no_update = True
        try:
            set_params(params, self.parameter_map)

            if "theme" in params:
                self.plotter.theme = self.vtk_toolbar.theme.value()
                self.plotter.set_background(self.plotter._theme.background)
        finally:
            self.no_update = False

    def get_time_text(self):
        return formatTimedelta(datetime.timedelta(seconds=float(self.t_slider.value() * self.result.time_delta) + self.time_start.value()),
                        self.time_format.value())

    def get_current_arrow_data(self):
        M = None
        field = None
        center = None
        name = ""
        colormap = None
        factor = None
        scale_max = self.vtk_toolbar.getScaleMax()
        stack_min_max = None
        skip = self.input_arrow_skip.value()
        if self.input_arrows.value() == "piv":
            if self.result is None:
                M = None
            else:
                M = self.result.mesh_piv[self.t_slider.value()]

            if M is not None:
                if M.hasNodeVar("U_measured"):
                    # showVectorField2(self, M, "U_measured")
                    field = M.getNodeVar("U_measured")
                    factor = 0.1 * self.vtk_toolbar.arrow_scale.value()
                    name = "U_measured"
                    colormap = self.vtk_toolbar.colormap_chooser.value()
                    stack_min_max = [M.R.min(axis=0) * 1e6, M.R.max(axis=0) * 1e6]
        elif self.input_arrows.value() == "target deformations":
            M = self.result.solver[self.t_slider.value()]
            # showVectorField2(self, M, "U_target")
            if M is not None:
                field = M.U_target
                factor = 0.1 * self.vtk_toolbar.arrow_scale.value()
                name = "U_target"
                colormap = self.vtk_toolbar.colormap_chooser.value()
                stack_min_max = [M.R.min(axis=0) * 1e6, M.R.max(axis=0) * 1e6]
        elif self.input_arrows.value() == "fitted deformations":
            M = self.result.solver[self.t_slider.value()]
            if M is not None:
                field = M.U
                factor = 0.1 * self.vtk_toolbar.arrow_scale.value()
                name = "U"
                colormap = self.vtk_toolbar.colormap_chooser.value()
                stack_min_max = [M.R.min(axis=0) * 1e6, M.R.max(axis=0) * 1e6]
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
                stack_min_max = [M.R.min(axis=0) * 1e6, M.R.max(axis=0) * 1e6]
                skip = self.input_arrow_skip2.value()
        else:
            # get min/max of stack
            M = self.result.solver[self.t_slider.value()]
            if M is not None:
                stack_min_max = [M.R.min(axis=0) * 1e6, M.R.max(axis=0) * 1e6]
            else:
                M = self.result.mesh_piv[self.t_slider.value()]
                if M is not None:
                    stack_min_max = [M.R.min(axis=0) * 1e6, M.R.max(axis=0) * 1e6]
                else:
                    stack_min_max = None
        return M, field, center, name, colormap, factor, scale_max, stack_min_max, skip

    def update_display(self):
        if self.no_update:
            return
        print(self.get_parameters())
        #self.set_parameters(self.get_parameters())
        #if self.current_tab_selected is False:
        #    self.current_result_plotted = False
        #    return

        if self.check_evaluated(self.result):
            if self.input_use2D.value():
                stack = self.result.stack[self.t_slider.value()]
                if self.input_reference_stack.value():
                    stack = self.result.stack_reference
                display_image = getVectorFieldImage(self, use_fixed_contrast_if_available=True, use_2D=True)
                if display_image is None:
                    return
                im_scale = self.input_scale.value()
                aa_scale = self.input_antialiase.value() + 1

                im = np.squeeze(display_image[0])

                colormap2 = self.vtk_toolbar.colormap_chooser2.value()
                if len(im.shape) == 2 and colormap2 is not None and colormap2 != "gray":
                    import matplotlib.pyplot as plt
                    cmap = plt.get_cmap(colormap2)
                    # print(img_adjusted.shape, img_adjusted.dtype, img_adjusted.min(), img_adjusted.mean(), img_adjusted.max())
                    im = (cmap(im) * 255).astype(np.uint8)[:, :, :3]
                    # print(img_adjusted.shape, img_adjusted.dtype, img_adjusted.min(), img_adjusted.mean(), img_adjusted.max())

                pil_image = Image.fromarray(im).convert("RGB")
                print("scale", im_scale)
                pil_image = pil_image.resize([int(pil_image.width*im_scale*aa_scale), int(pil_image.height*im_scale*aa_scale)])
                #im = np.asarray(im)
                def getBarParameters(pixtomu, scale=1):
                    mu = 200 * pixtomu / scale
                    values = [1, 5, 10, 25, 50, 75, 100, 150, 200, 250, 500, 1000, 1500, 2000, 2500, 5000, 10000]
                    old_v = mu
                    for v in values:
                        if mu < v:
                            mu = old_v
                            break
                        old_v = v
                    pixel = mu / pixtomu * scale
                    return pixel, mu
                if self.input_scalebar_um.value() == 0:
                    pixel, mu = getBarParameters(display_image[1][0])
                else:
                    mu = self.input_scalebar_um.value()
                    pixel = mu/display_image[1][0]

                def project_data(R, field, skip=1):
                    length = np.linalg.norm(field, axis=1)  
                    angle = np.arctan2(field[:, 1], field[:, 0])
                    data = pd.DataFrame(np.hstack((R, length[:, None], angle[:, None])),
                                        columns=["x", "y", "length", "angle"])
                    data = data.sort_values(by="length", ascending=False)
                    d2 = data.groupby(["x", "y"]).first()
                    # optional slice
                    if skip > 1:
                        d2 = d2.loc[(slice(None, None, skip), slice(None, None, skip)), :]
                    return np.array([i for i in d2.index]), d2[["length", "angle"]]

                M, field, center, name, colormap, factor, scale_max, stack_min_max, skip = self.get_current_arrow_data()
                if field is not None:
                    # rescale and offset
                    scale = 1e6 / display_image[1][0]
                    offset = np.array(display_image[0].shape[0:2]) / 2
  
                    R = M.R.copy()
                    field = field.copy()
                    R = R[:, :2][:, ::-1] * scale + offset
                    field = field[:, :2][:, ::-1] * scale * self.vtk_toolbar.arrow_scale.value()#factor

                    if scale_max is None:
                        max_length = np.nanmax(np.linalg.norm(field, axis=1))
                    else:
                        max_length = scale_max

                    z_center = (self.z_slider.value() - stack.shape[2] / 2) * display_image[1][2] * 1e-6
                    z_min = z_center - self.input_average_range.value()*1e-6
                    z_max = z_center + self.input_average_range.value()*1e-6

                    index = (z_min < M.R[:, 2]) & (M.R[:, 2] < z_max)

                    R = R[index]
                    field = field[index]
                    R, field = project_data(R, field, skip=self.input_arrow_skip.value())
                    pil_image = add_quiver(pil_image, R, field.length, field.angle, max_length=max_length, cmap=colormap, alpha=self.input_arrow_opacity.value() if self.input_arrows.value() != "fitted forces" else self.input_arrow_opacity2.value(), scale=im_scale*aa_scale,
                                           width=self.input_2darrow_width.value(), headlength=self.input_2darrow_headlength.value(), headheight=self.input_2darrow_headwidth.value())
                if aa_scale == 2:
                    pil_image = pil_image.resize([pil_image.width//2, pil_image.height//2])
                    aa_scale = 1
                pil_image = add_scalebar(pil_image, scale=1, image_scale=im_scale*aa_scale,
                                         width=self.input_scalebar_width.value()*aa_scale, xpos=self.input_scalebar_xpos.value()*aa_scale, ypos=self.input_scalebar_ypos.value()*aa_scale, fontsize=self.input_scalebar_fontsize.value()*aa_scale, pixel_width=pixel, size_in_um=mu, color="w", unit="µm")

                if self.result is not None and self.result.time_delta is not None and self.time_check.value():
                    pil_image = add_text(pil_image, self.get_time_text(), position=(10, 10))

                if self.input_logosize.value() >= 10:
                    if self.vtk_toolbar.theme.valueName() == "dark":
                        im_logo = Image.open(Path(__file__).parent / "../img/Logo_black.png")
                    else:
                        im_logo = Image.open(Path(__file__).parent / "../img/Logo.png")
                    scale = self.input_logosize.value() / im_logo.width  # im.width/400*0.2
                    im_logo = im_logo.resize([int(400 * scale * aa_scale), int(200 * scale * aa_scale)])
                    padding = int(im_logo.width * 0.1)
                    print("pil_image", pil_image.mode)
                    pil_image = pil_image.convert("RGBA")
                    pil_image.alpha_composite(im_logo, dest=(pil_image.width - im_logo.width - padding, padding))

                im = np.asarray(pil_image)
                self.pixmap1.setPixmap(QtGui.QPixmap(array2qimage(im)))
                self.view1.setExtend(im.shape[1], im.shape[0])
                self.view1.fitInView()
                self.im = im
                return
            cam_pos = None
            if self.plotter.camera_position is not None and CamPos.cam_pos_initialized is True:
                cam_pos = self.plotter.camera_position
            CamPos.cam_pos_initialized = True
            #self.plotter.interactor.setToolTip(str(self.result.interpolate_parameter)+f"\nNodes {self.result.solver[self.t_slider.value()].R.shape[0]}\nTets {self.result.solver[self.t_slider.value()].T.shape[0]}")
            crops = []
            for widged in [self.input_cropy, self.input_cropx, self.input_cropz]:
                crops.extend(widged.value())
            t = self.t_slider.value()
            t_start = time.time()
            stack_data = None
            stack = self.result.stack[t]
            if self.input_reference_stack.value():
                stack = self.result.stack_reference
            if self.channel0_properties.input_show.value() and crops[0] != crops[1] and crops[2] != crops[3] and crops[4] != crops[5]:
                stack_data1 = process_stack(stack, 0,
                                            crops=crops,
                                            **self.channel0_properties.value())
                stack_data = stack_data1
                self.channel0_properties.sigmoid.p.set_im(stack_data1["original"])

            else:
                stack_data1 = None
            if self.channel1_properties.input_show.value() and crops[0] != crops[1] and crops[2] != crops[3] and crops[4] != crops[5]:
                stack_data2 = process_stack(stack, 1,
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
                display_image = getVectorFieldImage(self, use_fixed_contrast_if_available=True)
                if len(self.result.stack):
                    stack_shape = np.array(self.result.stack[0].shape[:3]) * np.array(
                        self.result.stack[0].voxel_size)
                else:
                    stack_shape = None

                M, field, center, name, colormap, factor, scale_max, stack_min_max, skip = self.get_current_arrow_data()
                showVectorField(self.plotter, M, field, name, center=center,
                                skip=skip,
                                factor=factor,
                                colormap=colormap,
                                colormap2=self.vtk_toolbar.colormap_chooser2.value(),
                                scalebar_max=scale_max,
                                show_nan=self.vtk_toolbar.use_nans.value(),
                                display_image=display_image, show_grid=self.vtk_toolbar.show_grid.value(),
                                stack_shape=stack_shape, stack_min_max=stack_min_max,
                                arrow_opacity=self.input_arrow_opacity.value() if self.input_arrows.value() != "fitted forces" else self.input_arrow_opacity2.value())

                if stack_data is not None:
                    dataset = stack_data["data"]
                    mesh = pyvista.UniformGrid(dimensions=dataset.shape, spacing=stack_data["resolution"],
                                               origin=stack_data["center"])
                    mesh['values'] = dataset.ravel(order='F')
                    mesh.active_scalars_name = 'values'

                    vol = self.plotter.add_volume(mesh,# resolution=np.array(self.result.stack[0].voxel_size),
                                              cmap=stack_data["cmap"], opacity=stack_data["opacity"],
                                         blending="composite", name="fiber", render=False)  # 1.0*x
                    self.plotter.remove_scalar_bar("values")
                else:
                    self.plotter.remove_actor("fiber")
                print("plot time", f"{time.time()-t_start:.3f}s")

                if self.result is not None and self.result.time_delta is not None and self.time_check.value():
                    self.plotter.add_text(self.get_time_text(), name="time_text", font_size=self.time_size.value(),
                                          position=(20, self.input_height.value()-20-self.time_size.value()*2))
                else:
                    self.plotter.remove_actor("time_text")
                #self.plotter.camera.zoom(self.input_zoom.value())
            finally:
                self.plotter.render = render

                self.render_view(double_render=True)

            if cam_pos is not None:
                self.plotter.camera_position = cam_pos
        else:
            pass
            #self.plotter.interactor.setToolTip("")

    def render_view(self, double_render=False):
        render = self.plotter.render
        self.plotter.render = lambda *args: None
        try:
            # self.plotter.reset_camera()
            self.plotter.camera_position = "yz"
            # distance = self.plotter.camera.position[0]
            # self.input_distance.setValue(distance)
            if self.input_distance.value() == 0:
                distance = self.plotter.camera.position[0]
                self.input_distance.setValue(distance)

            # self.plotter.camera_position = "yz"
            # distance = self.plotter.camera.position[0]
            # self.plotter.camera.position = (self.input_distance.value(), 0, 10)


            dx = self.input_offset_x.value()
            dz = self.input_offset_y.value()
            dx, dz = rotate((dx, dz), self.input_roll.value())
            dx, dy = rotate((dx, 0), self.input_azimuth.value())
            self.plotter.camera.position = (self.input_distance.value() - dy, -dx, -dz)
            self.plotter.camera.focal_point = (0 - dy, -dx, -dz)
            # print(self.plotter.camera_position)
            self.plotter.camera.azimuth = self.input_azimuth.value()
            self.plotter.camera.elevation = self.input_elevation.value()
            self.plotter.camera.roll += self.input_roll.value()
            # im = self.plotter.show(screenshot="test.png", return_img=True, auto_close=False,
            #                       window_size=(self.input_width.value(), self.input_height.value()))
            # self.plotter.camera.elevation = 30
        finally:
            self.plotter.render = render

        import appdirs
        target_folder = Path(appdirs.user_data_dir("saenopy", "rgerum"))
        target_folder.mkdir(parents=True, exist_ok=True)
        tmp_file = str(target_folder / "tmp.png")

        if double_render is True:
            self.plotter.show(auto_close=False, window_size=(self.input_width.value(), self.input_height.value()))
        # render again to prevent potential shifts
        im = self.plotter.show(screenshot=tmp_file, return_img=True, auto_close=False,
                               window_size=(self.input_width.value(), self.input_height.value()))
        im = Image.open(tmp_file).convert("RGBA")
        if self.input_logosize.value() >= 10:
            if self.vtk_toolbar.theme.valueName() == "dark":
                im_logo = Image.open(Path(__file__).parent / "../img/Logo_black.png")
            else:
                im_logo = Image.open(Path(__file__).parent / "../img/Logo.png")
            scale = self.input_logosize.value() / im_logo.width  # im.width/400*0.2
            im_logo = im_logo.resize([int(400 * scale), int(200 * scale)])
            padding = int(im_logo.width * 0.1)
            im.alpha_composite(im_logo, dest=(im.width - im_logo.width - padding, padding))

        im = np.asarray(im)
        self.pixmap1.setPixmap(QtGui.QPixmap(array2qimage(im)))
        self.view1.setExtend(im.shape[1], im.shape[0])
        self.view1.fitInView()
        self.im = im

    drag_pos = None
    drag_pos2 = None
    drag_pos3 = None
    drag_pos4 = None
    drag_pos3_start_roll = None
    drag_pos3_start_roll1 = None
    drag_pos4_start_scale = None
    def sceneEventFilter(self, event):
        if self.input_use2D.value():
            return False
        if event.type() == QtCore.QEvent.GraphicsSceneMousePress and event.button() & QtCore.Qt.LeftButton and event.modifiers() & QtCore.Qt.ControlModifier:
            self.drag_pos3 = event.pos()
            self.drag_pos3_start_roll = self.input_roll.value()
            dx = event.pos().x() - self.input_width.value() / 2
            dy = event.pos().y() - self.input_height.value() / 2
            self.drag_pos3_start_roll1 = np.rad2deg(-np.arctan2(dy, dx))
            return True
        elif event.type() == QtCore.QEvent.GraphicsSceneMousePress and event.button() & QtCore.Qt.LeftButton:
            self.drag_pos = event.pos()
            self.drag_azimuth = self.input_azimuth.value()
            self.drag_elevation = self.input_elevation.value()
            self.drag_roll = self.input_roll.value()
            return True
        elif event.type() == QtCore.QEvent.GraphicsSceneMousePress and event.button() & QtCore.Qt.MiddleButton:
            self.drag_pos2 = event.pos()
            return True
        elif event.type() == QtCore.QEvent.GraphicsSceneMousePress and event.button() & QtCore.Qt.RightButton:
            self.drag_pos4 = event.pos()
            self.drag_pos4_start_scale = self.input_distance.value()
            return True
        elif event.type() == QtCore.QEvent.GraphicsSceneMouseRelease and event.button() & QtCore.Qt.LeftButton:
            self.drag_pos = None
            self.drag_pos3 = None
        elif event.type() == QtCore.QEvent.GraphicsSceneMouseRelease and event.button() & QtCore.Qt.MiddleButton:
            self.drag_pos2 = None
        elif event.type() == QtCore.QEvent.GraphicsSceneMouseRelease and event.button() & QtCore.Qt.MiddleButton:
            self.drag_pos4 = None
        elif event.type() == QtCore.QEvent.GraphicsSceneWheel:
            if event.delta() < 0:
                scale = self.input_distance.value() * 1.1**(-event.delta()/120)
            else:
                scale = self.input_distance.value() / 1.1**(event.delta()/120)
            self.input_distance.setValue(scale)
            self.render_view()
        elif event.type() == QtCore.QEvent.GraphicsSceneMouseMove and self.drag_pos3:
            dx = event.pos().x() - self.input_width.value()/2
            dy = event.pos().y() - self.input_height.value()/2
            angle = (np.rad2deg(-np.arctan2(dy, dx))-self.drag_pos3_start_roll1) + self.drag_pos3_start_roll
            self.input_roll.setValue(angle)
            self.render_view()
        elif event.type() == QtCore.QEvent.GraphicsSceneMouseMove and self.drag_pos2:
            dx = event.pos().x() - self.drag_pos2.x()
            dy = event.pos().y() - self.drag_pos2.y()
            self.drag_pos2 = event.pos()
            self.input_offset_x.setValue(self.input_offset_x.value() + dx)
            self.input_offset_y.setValue(self.input_offset_y.value() - dy)
            self.render_view()
        elif event.type() == QtCore.QEvent.GraphicsSceneMouseMove and self.drag_pos:
            dx = event.pos().x() - self.drag_pos.x()
            dy = event.pos().y() - self.drag_pos.y()
            dx, dy = rotate([dx, dy], -self.drag_roll)
            #self.drag_pos = event.pos()
            azimuth = (self.drag_azimuth - dx * 0.1)

            elevation = self.drag_elevation + dy * 0.2
            roll = self.drag_roll
            if elevation > 90:
                elevation = 90 - (elevation - 90)
                azimuth += 180
                roll = (self.drag_roll + 180) % 360
            elif elevation < -90:
                elevation = -90 - (elevation + 90)
                azimuth -= 180
                roll = (self.drag_roll - 180) % 360

            if azimuth < -180:
                azimuth += 360
            if azimuth >= 180:
                azimuth -= 360
            self.input_azimuth.setValue(azimuth)
            self.input_roll.setValue(roll)

            self.input_elevation.setValue(elevation)
            self.render_view()
            return True
        elif event.type() == QtCore.QEvent.GraphicsSceneMouseMove and self.drag_pos4:
            dy = event.pos().y() - self.drag_pos4.y()
            scale = self.drag_pos4_start_scale * 10 ** (dy/1000)
            self.input_distance.setValue(scale)
            self.render_view()
            return True
        return False

app = None
exporter = None
def render_image(params, result):
    global app, exporter
    if app is None:
        app = QtWidgets.QApplication([])
    if exporter is None:
        exporter = ExportViewer(None, None)
    exporter.set_parameters(params)
    exporter.setResult(result, no_update_display=True)
    exporter.set_parameters(params)

    exporter.update_display()
    return exporter.im



from PIL import Image, ImageDraw, ImageFont
def add_quiver(pil_image, R, lengths, angles, max_length, cmap, alpha=1, scale=1):
    cmap = plt.get_cmap(cmap)
    image = ImageDraw.ImageDraw(pil_image, "RGBA")
    def getarrow(length, width=2, headlength=5, headheight=5):
        length *= scale
        width *= scale
        headlength *= scale
        headheight *= scale
        print("scale", scale, length)
        if length < headlength:
            headheight = headheight*length/headlength
            headlength = length
            return [(length - headlength, headheight / 2),
                    (length, 0),
                    (length - headlength, -headheight / 2)]
        return [(0, width/2), (length-headlength, width/2), (length-headlength, headheight/2), (length, 0),
                (length-headlength, -headheight/2), (length-headlength, -width/2), (0, -width/2)]

    def get_offset(arrow, pos, angle):
        arrow = np.array(arrow)
        rot = [[np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))], [-np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]]
        arrow = arrow @ rot
        r = np.array(arrow) + np.array(pos)*scale
        return [tuple(i) for i in r]

    #max_length = np.nanmax(lengths)
    for i in range(len(R)):
        angle = angles.iloc[i]
        length = lengths.iloc[i]
        color = tuple((np.asarray(cmap(length/max_length))*255).astype(np.uint8))
        color = (color[0], color[1], color[2], int(alpha*255))
        image.polygon(get_offset(getarrow(length), R[i], np.rad2deg(angle)), fill=color, outline=color)
    return pil_image
def add_text(pil_image, text, position, fontsize=18):
    image = ImageDraw.ImageDraw(pil_image)
    font_size = int(round(fontsize * 4 / 3))  # the 4/3 appears to be a factor of "converting" screel dpi to image dpi
    try:
        font = ImageFont.truetype("arial", font_size)  # ImageFont.truetype("tahoma.ttf", font_size)
    except IOError:
        font = ImageFont.truetype("times", font_size)

    length_number = image.textsize(text, font=font)
    x, y = position

    if x < 0:
        x = pil_image.width + x - length_number[0]
    if y < 0:
        y = pil_image.height + y - length_number[1]
    color = tuple((matplotlib.colors.to_rgba_array("w")[0, :3] * 255).astype("uint8"))
    if pil_image.mode != "RGB":
        color = int(np.mean(color))

    image.text((x, y), text, color, font=font)
    return pil_image

def add_scalebar(pil_image, scale, image_scale, width, xpos, ypos, fontsize, pixel_width, size_in_um, color="w", unit="µm"):
    image = ImageDraw.ImageDraw(pil_image)
    pixel_height = width
    pixel_offset_x = xpos
    pixel_offset_y = ypos
    pixel_offset2 = 3
    font_size = int(round(fontsize*scale*4/3))  # the 4/3 appears to be a factor of "converting" screel dpi to image dpi

    #pixel_width, size_in_um = self.getBarParameters(1)
    pixel_width *= image_scale
    color = tuple((matplotlib.colors.to_rgba_array(color)[0, :3]*255).astype("uint8"))
    if pil_image.mode != "RGB":
        color = int(np.mean(color))

    if pixel_offset_x > 0:
        image.rectangle([pil_image.size[0] -pixel_offset_x - pixel_width, pil_image.size[1] -pixel_offset_y - pixel_height, pil_image.size[0] -pixel_offset_x, pil_image.size[1] -pixel_offset_y], color)
    else:
        image.rectangle([-pixel_offset_x,
                         pil_image.size[1] - pixel_offset_y - pixel_height,
                         -pixel_offset_x + pixel_width,
                         pil_image.size[1] - pixel_offset_y], color)
    if True:
        # get the font
        try:
            font = ImageFont.truetype("arial", font_size)#ImageFont.truetype("tahoma.ttf", font_size)
        except IOError:
            font = ImageFont.truetype("times", font_size)
        # width and height of text elements
        text = "%d" % size_in_um
        length_number = image.textsize(text, font=font)[0]
        length_space = 0.5*image.textsize(" ", font=font)[0] # here we emulate a half-sized whitespace
        length_unit = image.textsize(unit, font=font)[0]
        height_number = image.textsize(text+unit, font=font)[1]

        total_length = length_number + length_space + length_unit

        # find the position for the text to have it centered and bottom aligned
        if pixel_offset_x > 0:
            x = pil_image.size[0] - pixel_offset_x - pixel_width * 0.5 - total_length * 0.5
        else:
            x = - pixel_offset_x + pixel_width * 0.5 - total_length * 0.5
        y = pil_image.size[1] - pixel_offset_y - pixel_offset2 - pixel_height - height_number
        # draw the text for the number and the unit
        image.text((x, y), text, color, font=font)
        image.text((x+length_number+length_space, y), unit, color, font=font)
        return pil_image




def getarrow(length, angle, scale=1, width=2, headlength=5, headheight=5, offset=None):
    length *= scale
    width *= scale
    headlength *= scale
    headheight *= scale
    print("scale", scale, length)
    headlength = headlength*np.ones(len(length))
    headheight = headheight*np.ones(len(length))
    width = width*np.ones(len(length))
    index_small = length < headlength
    if np.any(index_small):
        headheight[index_small] = headheight[index_small] * length[index_small] / headlength[index_small]
        headlength[index_small] = length[index_small]
        width[index_small] = headheight[index_small]

    # generate the arrow points
    arrow = [(0, width / 2), (length - headlength, width / 2), (length - headlength, headheight / 2), (length, 0),
            (length - headlength, -headheight / 2), (length - headlength, -width / 2), (0, -width / 2)]
    # and distribute them for each point
    arrows = np.zeros([length.shape[0], 7, 2])
    for p in range(7):
        for i in range(2):
            arrows[:, p, i] = arrow[p][i]

    # rotate the arrow
    #angle = np.deg2rad(angle)
    rot = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    arrows = np.einsum("ijk,kli->ijl", arrows, rot)

    # add the offset
    arrows += offset[:, None, :]

    return arrows


def add_quiver(pil_image, R, lengths, angles, max_length, cmap, alpha=1, scale=1, width=2, headlength=5, headheight=5):
    # get the colormap
    cmap = plt.get_cmap(cmap)
    # calculate the colors of the arrows
    colors = cmap(lengths / max_length)
    # set the transparancy
    colors[:, 3] = alpha
    # make colors uint8
    colors = (colors*255).astype(np.uint8)

    # get the arrows
    arrows = getarrow(lengths, angles, scale=scale, width=width, headlength=headlength, headheight=headheight, offset=R*scale)

    # draw the arrows
    image = ImageDraw.ImageDraw(pil_image, "RGBA")
    for a, c in zip(arrows, colors):
        image.polygon(list(a.flatten()), fill=tuple(c), outline=tuple(c))

    return pil_image
