import sys
import os
import time
import qtawesome as qta
from qtpy import QtCore, QtWidgets, QtGui
import numpy as np
import pyvista as pv
from typing import List, Tuple
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
import saenopy.multigrid_helper
from saenopy.multigrid_helper import get_scaled_mesh, get_nodes_with_one_face
import saenopy.get_deformations
import saenopy.materials
from saenopy.stack import Stack, format_glob
from saenopy.saveable import Saveable
from saenopy import Result
from saenopy.gui.common import QtShortCuts, QExtendedGraphicsView
from saenopy.gui.common.gui_classes import Spoiler, CheckAbleGroup, QHLine, QVLine, MatplotlibWidget, NavigationToolbar, execute, kill_thread, ListWidget, QProcess, ProcessSimple
from saenopy.gui.common.stack_selector import StackSelector


from .PipelineModule import PipelineModule
from .QTimeSlider import QTimeSlider
from .VTK_Toolbar import VTK_Toolbar
from .showVectorField import showVectorField, showVectorField2
from .DeformationDetector import CamPos
from .code_export import get_code

class MeshSizeWidget(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(object)

    def __init__(self):
        super().__init__()
        with QtShortCuts.QVBoxLayout(self):
            with QtShortCuts.QHBoxLayout():
                self.input_mesh_size_same = QtShortCuts.QInputBool(None, "mesh size same as stack", True,
                                                                   #value_changed=self.valueChanged,
                                                                   tooltip="make the mesh size the same as the piv mesh")
            with QtShortCuts.QHBoxLayout():
                self.input_mesh_size_x = QtShortCuts.QInputNumber(None, "x", 200, step=1, unit="μm",
                                                                  tooltip="the custom new mesh size")
                self.input_mesh_size_y = QtShortCuts.QInputNumber(None, "y", 200, step=1, unit="μm",
                                                                  tooltip="the custom new mesh size")
                self.input_mesh_size_z = QtShortCuts.QInputNumber(None, "z", 200, step=1, unit="μm",
                                                                  tooltip="the custom new mesh size")
            self.input_mesh_size_x.valueChanged.connect(self.valueChangedCallback)
            self.input_mesh_size_y.valueChanged.connect(self.valueChangedCallback)
            self.input_mesh_size_z.valueChanged.connect(self.valueChangedCallback)
            self.input_mesh_size_same.valueChanged.connect(self.valueChangedCallback)

            self.setValue("piv")

    def valueChangedCallback(self):
        self.input_mesh_size_x.setDisabled(self.input_mesh_size_same.value())
        self.input_mesh_size_y.setDisabled(self.input_mesh_size_same.value())
        self.input_mesh_size_z.setDisabled(self.input_mesh_size_same.value())
        self.valueChanged.emit(self.value())

    def value(self):
        if self.input_mesh_size_same.value():
            return "piv"
        else:
            return (self.input_mesh_size_x.value(), self.input_mesh_size_y.value(), self.input_mesh_size_z.value())

    def setValue(self, value):
        if value == "piv":
            self.input_mesh_size_same.setValue(True)
        else:
            self.input_mesh_size_same.setValue(False)
            self.input_mesh_size_x.setValue(value[0])
            self.input_mesh_size_y.setValue(value[1])
            self.input_mesh_size_z.setValue(value[2])
        self.input_mesh_size_x.setDisabled(self.input_mesh_size_same.value())
        self.input_mesh_size_y.setDisabled(self.input_mesh_size_same.value())
        self.input_mesh_size_z.setDisabled(self.input_mesh_size_same.value())


class MeshCreator(PipelineModule):
    mesh_size = [200, 200, 200]
    pipeline_name = "interpolate mesh"

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with QtShortCuts.QVBoxLayout(self) as layout:
            layout.setContentsMargins(0, 0, 0, 0)
            with CheckAbleGroup(self, "interpolate mesh", url="https://saenopy.readthedocs.io/en/latest/interface_solver.html#create-finite-element-mesh").addToLayout() as self.group:

                with QtShortCuts.QVBoxLayout():

                    with QtShortCuts.QHBoxLayout() as layout2:
                        self.input_reference = QtShortCuts.QInputChoice(None, "reference stack", "next",
                                                                        ["next", "median"], tooltip="the reference for the deformations")
                        self.input_reference.setEnabled(False)
                        self.input_element_size = QtShortCuts.QInputNumber(None, "mesh elem. size", 7, unit="μm", tooltip="the element size of the regularisatio mesh")
                        layout2.addStretch()

                    self.input_mesh_size = MeshSizeWidget().addToLayout()

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

        self.setParameterMapping("mesh_parameters", {
            "reference_stack": self.input_reference,
            "element_size": self.input_element_size,
            "mesh_size": self.input_mesh_size,
        })

    def z_slider_value_changed(self):
        self.update_display()

    def check_available(self, result: Result):
        return result is not None and result.mesh_piv is not None and len(result.mesh_piv) and result.mesh_piv[0] is not None

    def check_evaluated(self, result: Result) -> bool:
        return result is not None and result.solvers is not None and len(result.solvers) and result.solvers[0] is not None

    def property_changed(self, name, value):
        if name == "z_slider":
            self.z_slider.setValue(value)

    def setResult(self, result: Result):
        super().setResult(result)
        if result and result.stacks and result.stacks[0]:
            self.z_slider.setRange(0, result.stacks[0].shape[2] - 1)
            self.z_slider.setValue(self.result.stacks[0].shape[2] // 2)

            if result.stacks[0].channels:
                self.vtk_toolbar.channel_select.setValues(np.arange(len(result.stacks[0].channels)), result.stacks[0].channels)
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
            M = self.result.solvers[self.t_slider.value()]
            mesh = M.mesh
            self.plotter.interactor.setToolTip(str(self.result.mesh_parameters) + f"\nNodes {mesh.nodes.shape[0]}\nTets {mesh.tetrahedra.shape[0]}")
            showVectorField2(self, mesh, "displacements_target")
            if cam_pos is not None:
                self.plotter.camera_position = cam_pos
        else:
            self.plotter.interactor.setToolTip("")

    def process(self, result: Result, mesh_parameters: dict):
        # demo run
        if os.environ.get("DEMO") == "true":  # pragma: no cover
            result.solvers = result.solver_demo
            return
        
        # make sure the solver list exists and has the required length
        if result.solvers is None or len(result.solvers) != len(result.mesh_piv):
            result.solvers = [None] * len(result.mesh_piv)
        
        # correct for the reference state
        displacement_list = saenopy.subtract_reference_state(result.mesh_piv, mesh_parameters["reference_stack"])
        
        # set the parameters
        result.mesh_parameters = mesh_parameters
        # iterate over all stack pairs
        for i in range(len(result.mesh_piv)):
            # and create the interpolated solver mesh
            result.solvers[i] = saenopy.interpolate_mesh(result.mesh_piv[i], displacement_list[i], mesh_parameters)
        # save the meshes
        result.save()

    def get_code(self) -> Tuple[str, str]:
        import_code = ""
        results = None
        def code(my_mesh_params):  # pragma: no cover
            # define the parameters to generate the solver mesh and interpolate the piv mesh onto it
            mesh_parameters = my_mesh_params

            # iterate over all the results objects
            for result in results:
                # correct for the reference state
                displacement_list = saenopy.subtract_reference_state(result.mesh_piv, mesh_parameters["reference_stack"])
                # set the parameters
                result.mesh_parameters = mesh_parameters
                # iterate over all stack pairs
                for i in range(len(result.mesh_piv)):
                    # and create the interpolated solver mesh
                    result.solvers[i] = saenopy.interpolate_mesh(result.mesh_piv[i], displacement_list[i], mesh_parameters)
                # save the meshes
                result.save()
        data = {
            "my_mesh_params": self.result.mesh_parameters_tmp,
        }

        code = get_code(code, data)
        return import_code, code
