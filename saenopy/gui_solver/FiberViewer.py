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
from saenopy import Result

from typing import List, Tuple

from .PipelineModule import PipelineModule
from .QTimeSlider import QTimeSlider
from .VTK_Toolbar import VTK_Toolbar
from .showVectorField import showVectorField, showVectorField2
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
import time


caches = {}
class Cache:
    pass
def get_cache(data):
    cache_obj = data
    if isinstance(cache_obj, np.ndarray):
        cache_obj = Cache()
        if id(data) in caches:
            cache_obj = caches[id(data)]
        caches[id(data)] = cache_obj
    return cache_obj


def cache_results():
    def wrapper(function):
        def apply(data, **params):
            cache_name = f"_saved_{function}"
            if "channel" in params:
                cache_name = f"_saved_{function}_{params['channel']}"
                print("channel", cache_name)
            else:
                print(cache_name, params)
            t = time.time()
            cache_obj = get_cache(data)
            cached = getattr(cache_obj, cache_name, None)
            cached_params = getattr(cache_obj, f"{cache_name}_params", None)
            if cached is not None and cached_params == params:
                print("cache", function, "cached", params, f"{time.time()-t:.3f}s")
                return cached
            result = function(data, **params)
            setattr(cache_obj, cache_name, result)
            setattr(cache_obj, f"{cache_name}_params", params)
            print("cache", function, "apply", params, f"{time.time()-t:.3f}s")
            return result
        return apply
    return wrapper


@cache_results()
def sato_filter(data, sigma_sato=None):
    if sigma_sato is None or sigma_sato == 0:
        return data.astype(np.float64)
    fiber = np.zeros_like(data, dtype=np.float64)
    for z in range(data.shape[-1]):
        fiber[:, :, z] = sato(data[:, :, z], sigmas=range(sigma_sato, sigma_sato + 1, 1), black_ridges=False)
    return fiber


def get_transparency(a, b, alpha):
    x1 = np.linspace(0, 1, 21)

    def sigmoid(x):
        return 1 / (1 + np.exp(-((x - b) / a)))
        #return 1 / (1 + np.exp(-(x / a) - b))

    x = sigmoid(x1)
    opacity = 1.0 * alpha * x
    return opacity


@cache_results()
def get_stack_images(stack, channel, crops):
    minx, maxx, miny, maxy, minz, maxz = crops
    # get the stack as a numpy array for the given imaging channel
    data = stack[minx:maxx, miny:maxy, :, minz:maxz, channel]
    # make sure the stack data has onlz one color channel
    if data.shape[2] == 1:
        data = data[:, :, 0]
    else:
        data = np.mean(data, axis=2)
    return data


@cache_results()
def scale_intensity(data, percentile):
    return exposure.rescale_intensity(data, in_range=tuple(np.percentile(data, list(percentile))))


@cache_results()
def zoom_z(scaled_data, factor):
    if 1:
        pos = np.linspace(0, scaled_data.shape[2] - 1,
                          int(scaled_data.shape[2] * factor))
        f2 = pos % 1
        f1 = 1 - f2
        zoomed_data2a = scaled_data[:, :, np.floor(pos).astype(int)]
        zoomed_data2b = scaled_data[:, :, np.ceil(pos).astype(int)]
        zoomed_data2 = f1 * zoomed_data2a + f2 * zoomed_data2b
        return zoomed_data2
        plt.plot(zoomed_data[50, 50])
        plt.plot(zoomed_data2[50, 50])
        plt.plot(zoomed_data2a[50, 50])
        plt.plot(zoomed_data2b[50, 50])
    zoomed_data = ndimage.zoom(scaled_data, (1, 1, factor))
    return zoomed_data


@cache_results()
def smooth(zoomed_data, sigma_gauss):
    if sigma_gauss is not None or sigma_gauss == 0:
        smoothed_data = gaussian_filter(zoomed_data, sigma=sigma_gauss, truncate=2.0)
    else:
        smoothed_data = zoomed_data
    return smoothed_data


@cache_results()
def rescale(smoothed_data):
    combined1 = smoothed_data.copy()
    max_int = np.max(combined1)
    combined1 = combined1 / max_int * 255
    combined1[combined1 < 0] = 0
    combined1 = np.rot90(combined1)
    return combined1.astype(np.uint8)


def process_stack(stack, channel, crops=None, sigma_sato=None, sigma_gauss=None, percentiles=(0.01, 99.6), alpha=(1.3, 3.86, 1),
                  cmap="gray"):
    data = get_stack_images(stack, channel=channel, crops=crops)

    filtered_data = sato_filter(data, sigma_sato=sigma_sato)

    scaled_data = scale_intensity(filtered_data, percentile=percentiles)

    zoomed_data = zoom_z(scaled_data,  factor=stack.voxel_size[2] / stack.voxel_size[1])

    smoothed_data = smooth(zoomed_data, sigma_gauss=sigma_gauss)

    combined1 = rescale(smoothed_data)

    opacity = get_transparency(*alpha)
    return {"data": combined1, "opacity": opacity, "cmap": cmap}


def join_stacks(stack_data1, stack_data2, thres_cell=1):
    cell2_zoom = stack_data2["data"]
    combined2 = 128 + stack_data2["data"] / 2  # '* 127
    # segment the cell in 3D
    thres = threshold_yen(cell2_zoom) * thres_cell
    cell_mask = (cell2_zoom > thres)
    combined = stack_data1["data"].copy() / 2
    combined[cell_mask] = combined2[cell_mask]

    # color the fused stack
    cfib = matplotlib.cm.get_cmap(stack_data1["cmap"], 128)
    ccell = matplotlib.cm.get_cmap(stack_data2["cmap"], 128)
    newcolors = np.r_[cfib(np.linspace(0, 1, 128)), ccell(np.linspace(0.5, 1, 128))]
    newcmp = ListedColormap(newcolors)
    opacity = np.r_[stack_data1["opacity"], stack_data2["opacity"]]
    return {"data": combined, "opacity": opacity, "cmap": newcmp}


class ChannelProperties(QtWidgets.QWidget):
    valueChanged = QtCore.Signal()
    def __init__(self):
        super().__init__()
        with QtShortCuts.QHBoxLayout() as layout:
            self.input_show = QtShortCuts.QInputBool(None, "show", True)
            layout.addStretch()
        with QtShortCuts.QHBoxLayout() as layout:
            self.input_sato = QtShortCuts.QInputNumber(None, "sato filter", 2, min=0, max=7, float=False)
            self.input_gauss = QtShortCuts.QInputNumber(None, "gauss filter", 0, min=0, max=20, float=False)
            self.input_percentile1 = QtShortCuts.QInputNumber(None, "percentile_min", 0.01)
            self.input_percentile2 = QtShortCuts.QInputNumber(None, "percentile_max", 99.6)
        with QtShortCuts.QHBoxLayout() as layout:
            self.input_alpha1 = QtShortCuts.QInputNumber(None, "alpha1", 0.065, min=0, max=0.3, step=0.01, decimals=3)
            self.input_alpha2 = QtShortCuts.QInputNumber(None, "alpha2", 0.2491, min=0, max=1, step=0.01)
            self.input_alpha3 = QtShortCuts.QInputNumber(None, "alpha3", 1, min=0, max=1, step=0.1)
            self.input_cmap = QtShortCuts.QDragableColor("pink").addToLayout()
        for widget in [self.input_sato, self.input_gauss, self.input_percentile1, self.input_percentile2,
                       self.input_alpha1, self.input_alpha2, self.input_alpha3,
                       self.input_cmap, self.input_show]:
            widget.valueChanged.connect(self.valueChanged)

    def value(self):
        return dict(sigma_sato=self.input_sato.value(), sigma_gauss=self.input_gauss.value(),
                    percentiles=(self.input_percentile1.value(), self.input_percentile2.value()),
                    alpha=(self.input_alpha1.value(), self.input_alpha2.value(), self.input_alpha3.value()),
                    cmap=self.input_cmap.value())


class FiberViewer(PipelineModule):

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with self.parent.tabs.createTab("Fiber View") as self.tab:
            with QtShortCuts.QVBoxLayout() as layout:
                self.label_tab = QtWidgets.QLabel("The deformations from the piv algorithm interpolated on the new mesh for regularisation.").addToLayout()

                with QtShortCuts.QHBoxLayout() as layout:
                    self.plotter = QtInteractor(self, auto_update=False)  # , theme=pv.themes.DocumentTheme())
                    self.tab.parent().plotter = self.plotter
                    self.plotter.set_background("black")
                    layout.addWidget(self.plotter.interactor)

                    #self.z_slider = QTimeSlider("z", self.z_slider_value_changed, "set z position",
                    #                            QtCore.Qt.Vertical).addToLayout()
                    #self.z_slider.t_slider.valueChanged.connect(
                    #    lambda value: parent.shared_properties.change_property("z_slider", value, self))
                    #parent.shared_properties.add_property("z_slider", self)\

                self.vtk_toolbar = VTK_Toolbar(self.plotter, self.update_display, shared_properties=self.parent.shared_properties).addToLayout()

                with QtShortCuts.QHBoxLayout() as layout:
                    self.input_cropx = QtShortCuts.QRangeSlider(None, "crop x", 0, 200)
                    self.input_cropy = QtShortCuts.QRangeSlider(None, "y", 0, 200)
                    self.input_cropz = QtShortCuts.QRangeSlider(None, "z", 0, 200)
                self.channel0_properties = ChannelProperties().addToLayout()
                self.channel0_properties.valueChanged.connect(self.update_display)
                self.channel1_properties = ChannelProperties().addToLayout()
                self.channel1_properties.valueChanged.connect(self.update_display)
                self.channel1_properties.input_cmap.setValue("Greens")
                self.channel1_properties.input_sato.setValue(0)
                self.channel1_properties.input_gauss.setValue(7)
                self.channel1_properties.input_percentile1.setValue(10)
                self.channel1_properties.input_percentile2.setValue(100)
                self.input_thresh = QtShortCuts.QInputNumber(None, "thresh", 1, float=True, min=0, max=2, step=0.1)
                self.input_thresh.valueChanged.connect(self.update_display)
                self.canvas = MatplotlibWidget(self).addToLayout()

                self.t_slider = QTimeSlider(connected=self.update_display).addToLayout()
                self.tab.parent().t_slider = self.t_slider

        self.setParameterMapping(None, {})


    def check_evaluated(self, result: Result) -> bool:
        return self.result is not None and result.stack is not None and len(result.stack) > 0

    def setResult(self, result: Result):
        super().setResult(result)
        if result and result.stack and result.stack[0]:
            shape = result.stack[0].shape
            self.input_cropx.setRange(0, shape[1])
            self.input_cropx.setValue((shape[1] // 2 - 100, shape[1] // 2 + 100))
            self.input_cropy.setRange(0, shape[0])
            self.input_cropy.setValue((shape[0] // 2 - 100, shape[0] // 2 + 100))
            self.input_cropz.setRange(0, shape[2])
            self.input_cropz.setValue((shape[2] // 2 - 25, shape[2] // 2 + 25))



    def update_display(self):
        if self.current_tab_selected is False:
            self.current_result_plotted = False
            return

        if self.check_evaluated(self.result):
            cam_pos = None
            if self.plotter.camera_position is not None and CamPos.cam_pos_initialized is True:
                cam_pos = self.plotter.camera_position
            CamPos.cam_pos_initialized = True
            #self.plotter.interactor.setToolTip(str(self.result.interpolate_parameter)+f"\nNodes {self.result.solver[self.t_slider.value()].R.shape[0]}\nTets {self.result.solver[self.t_slider.value()].T.shape[0]}")
            crops = []
            for widged in [self.input_cropx, self.input_cropy, self.input_cropz]:
                crops.extend(widged.value())
            t = time.time()
            stack_data = None
            if self.channel0_properties.input_show.value():
                stack_data1 = process_stack(self.result.stack[0], 0,
                                            crops=crops,
                                            **self.channel0_properties.value())
                stack_data = stack_data1
            else:
                stack_data1 = None
            if self.channel1_properties.input_show.value():
                stack_data2 = process_stack(self.result.stack[0], 1,
                                            crops=crops,
                                            **self.channel1_properties.value())
                if stack_data1 is not None:
                    stack_data = join_stacks(stack_data1, stack_data2, self.input_thresh.value())
                else:
                    stack_data = stack_data2
            else:
                stack_data2 = None
            #stack_data = stack_data1
            if self.canvas is not None and stack_data is not None:
                self.canvas.figure.axes[0].cla()
                self.canvas.figure.axes[0].plot(np.linspace(0, 1, len(stack_data["opacity"])), stack_data["opacity"])
                self.canvas.figure.axes[0].spines["top"].set_visible(False)
                self.canvas.figure.axes[0].spines["right"].set_visible(False)
                self.canvas.figure.axes[0].set_xlim(0, 1)
                self.canvas.figure.axes[0].set_ylim(0, 1)
                self.canvas.draw()

            vol = self.plotter.add_volume(stack_data["data"], cmap=stack_data["cmap"], opacity=stack_data["opacity"],
                                     blending="composite", name="fiber")  # 1.0*x
            print("plot time", f"{time.time()-t:.3f}s")
            # plotter.remove_scalar_bar()
            self.plotter.render()

            if cam_pos is not None:
                self.plotter.camera_position = cam_pos
        else:
            self.plotter.interactor.setToolTip("")
