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
from ..stack import Stack, getStack, format_glob
from saenopy.multigridHelper import getScaledMesh, getNodesWithOneFace
from saenopy.loadHelpers import Saveable
from saenopy import Result

from typing import List, Tuple

from .PipelineModule import PipelineModule
from .QTimeSlider import QTimeSlider
from .VTK_Toolbar import VTK_Toolbar
from .showVectorField import showVectorField, showVectorField2
from .DeformationDetector import CamPos


class MeshCreator(PipelineModule):
    mesh_size = [200, 200, 200]
    pipeline_name = "interpolate mesh"

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with QtShortCuts.QVBoxLayout(self) as layout:
            layout.setContentsMargins(0, 0, 0, 0)
            with CheckAbleGroup(self, "interpolate mesh").addToLayout() as self.group:

                with QtShortCuts.QVBoxLayout():

                    with QtShortCuts.QHBoxLayout() as layout2:
                        self.input_reference = QtShortCuts.QInputChoice(None, "reference stack", "next",
                                                                        ["next", "median"])
                        self.input_reference.setEnabled(False)
                        self.input_element_size = QtShortCuts.QInputNumber(None, "mesh elem. size", 7, unit="μm")
                        #with QtShortCuts.QHBoxLayout() as layout2:
                        self.input_inner_region = QtShortCuts.QInputNumber(None, "inner region", 100, unit="μm")
                        self.input_inner_region.setVisible(False)
                        self.input_thinning_factor = QtShortCuts.QInputNumber(None, "thinning factor", 0, step=0.1)
                        self.input_thinning_factor.setVisible(False)
                        layout2.addStretch()
                    with QtShortCuts.QHBoxLayout() as layout2:
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

                with QtShortCuts.QHBoxLayout() as layout:
                    self.plotter = QtInteractor(self, auto_update=False)  # , theme=pv.themes.DocumentTheme())
                    self.tab.parent().plotter = self.plotter
                    self.plotter.set_background("black")
                    layout.addWidget(self.plotter.interactor)

                    self.z_slider = QTimeSlider("z", self.z_slider_value_changed, "set z position",
                                                QtCore.Qt.Vertical).addToLayout()
                    self.z_slider.t_slider.valueChanged.connect(
                        lambda value: parent.shared_properties.change_property("z_slider", value, self))
                    parent.shared_properties.add_property("z_slider", self)\

                self.vtk_toolbar = VTK_Toolbar(self.plotter, self.update_display, shared_properties=self.parent.shared_properties).addToLayout()


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
            try:
                valid = self.result.mesh_piv[0] is not None
            except (AttributeError, IndexError, TypeError):
                valid = False
            if valid:
                x, y, z = (self.result.mesh_piv[0].R.max(axis=0) - self.result.mesh_piv[0].R.min(axis=0))*1e6
                self.input_mesh_size_x.setValue(x)
                self.setParameter("mesh_size_x", x)
                self.input_mesh_size_y.setValue(y)
                self.setParameter("mesh_size_y", y)
                self.input_mesh_size_z.setValue(z)
                self.setParameter("mesh_size_z", z)

    def z_slider_value_changed(self):
        self.update_display()

    def check_available(self, result: Result):
        return result is not None and result.mesh_piv is not None and len(result.mesh_piv) and result.mesh_piv[0] is not None

    def check_evaluated(self, result: Result) -> bool:
        return result is not None and result.solver is not None and len(result.solver) and result.solver[0] is not None

    def property_changed(self, name, value):
        if name == "z_slider":
            self.z_slider.setValue(value)

    def setResult(self, result: Result):
        super().setResult(result)
        if result and result.stack and result.stack[0]:
            self.z_slider.setRange(0, result.stack[0].shape[2] - 1)
            self.z_slider.setValue(self.result.stack[0].shape[2] // 2)

            if result.stack[0].channels:
                self.vtk_toolbar.channel_select.setValues(np.arange(len(result.stack[0].channels)), result.stack[0].channels)
                self.vtk_toolbar.channel_select.setVisible(True)
            else:
                self.vtk_toolbar.channel_select.setValue(0)
                self.vtk_toolbar.channel_select.setVisible(False)

            if result.stack_reference is None:
                self.input_reference.setValues(["next", "median"])
                self.input_reference.setEnabled(True)
            else:
                self.input_reference.setValues(["reference stack"])
                self.input_reference.setEnabled(False)

    def update_display(self):
        if self.current_tab_selected is False:
            self.current_result_plotted = False
            return

        if self.result is not None and len(self.result.mesh_piv) > 2:
            self.input_reference.setEnabled(True)
        else:
            self.input_reference.setEnabled(False)
        if self.check_evaluated(self.result):
            cam_pos = None
            if self.plotter.camera_position is not None and CamPos.cam_pos_initialized is True:
                cam_pos = self.plotter.camera_position
            CamPos.cam_pos_initialized = True
            self.plotter.interactor.setToolTip(str(self.result.interpolate_parameter)+f"\nNodes {self.result.solver[self.t_slider.value()].R.shape[0]}\nTets {self.result.solver[self.t_slider.value()].T.shape[0]}")
            M = self.result.solver[self.t_slider.value()]
            showVectorField2(self, M, "U_target")
            if cam_pos is not None:
                self.plotter.camera_position = cam_pos
        else:
            self.plotter.interactor.setToolTip("")

    def process(self, result: Result, params: dict):
        
        # make sure the solver list exists and has the required length
        if result.solver is None or len(result.solver) != len(result.mesh_piv):
            result.solver = [None]*len(result.mesh_piv)
        
        # correct for the reference state
        displacement_list = saenopy.substract_reference_state(result.mesh_piv, params["reference_stack"])
        
        # set the parameters
        result.interpolate_parameter = params
        # iterate over all stack pairs
        for i in range(len(result.mesh_piv)):
            # and create the interpolated solver mesh
            result.solver[i] = saenopy.interpolate_mesh(result.mesh_piv[i], displacement_list[i], params)
        # save the meshes
        result.save()
            
        return
        solvers = []
        mode = self.input_reference.value()
        
        if params["mesh_size_same"]:
            x, y, z = (result.mesh_piv[0].R.max(axis=0) - result.mesh_piv[0].R.min(axis=0))*1e6
            params["mesh_size_x"] = x
            params["mesh_size_y"] = y
            params["mesh_size_z"] = z

        U = [M.getNodeVar("U_measured") for M in result.mesh_piv]
        # correct for the median position
        if result.stack_reference is None:# len(U) > 2:
            xpos2 = np.cumsum(U, axis=0)  # mittlere position
            if mode == "first":
                xpos2 -= xpos2[0]
            elif mode == "median":
                xpos2 -= np.nanmedian(xpos2, axis=0)  # aktuelle abweichung von
            elif mode == "last":
                xpos2 -= xpos2[-1]
            elif mode == "next":
                xpos2 = U
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
        import_code = ""
        results = None
        def code(my_mesh_params):
            # define the parameters to generate the solver mesh and interpolate the piv mesh onto it
            params = my_mesh_params

            # iterate over all the results objects
            for result in results:
                # correct for the reference state
                displacement_list = saenopy.substract_reference_state(result.mesh_piv, params["reference_stack"])
                # set the parameters
                result.interpolate_parameter = params
                # iterate over all stack pairs
                for i in range(len(result.mesh_piv)):
                    # and create the interpolated solver mesh
                    result.solver[i] = saenopy.interpolate_mesh(result.mesh_piv[i], displacement_list[i], params)
                # save the meshes
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

