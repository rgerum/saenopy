import os
from qtpy import QtCore, QtWidgets, QtGui
import qtawesome as qta
from typing import Tuple
from pathlib import Path

import saenopy
import saenopy.multigrid_helper
import saenopy.get_deformations
import saenopy.materials
from saenopy import Result
from saenopy.gui.common import QtShortCuts
from saenopy.gui.common.gui_classes import CheckAbleGroup


from saenopy.gui.common.PipelineModule import PipelineModule, StateEnum
from saenopy.gui.common.code_export import get_code, export_as_string


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

class CancelSignal:
    cancel = False

class MeshCreator(PipelineModule):
    mesh_size = [200, 200, 200]
    pipeline_name = "interpolate mesh"

    pipeline_allow_cancel = True
    pipeline_button_name = "interpolate mesh"

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

                    with QtShortCuts.QHBoxLayout():
                        self.input_button = QtWidgets.QPushButton("interpolate mesh").addToLayout()
                        self.input_button.clicked.connect(self.start_process)
                        self.input_button_text = QtWidgets.QLabel().addToLayout()
                        self.input_button_text.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

                        self.input_button_reset = QtShortCuts.QPushButton(None, "", self.reset, icon=qta.icon("fa5s.trash-alt"),
                                                                          tooltip="reset")
                        self.input_button_reset.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        self.setParameterMapping("mesh_parameters", {
            "reference_stack": self.input_reference,
            "element_size": self.input_element_size,
            "mesh_size": self.input_mesh_size,
        })

    def cancel_process(self):
        self.set_result_state(self.result, StateEnum.cancelling)
        self.parent.result_changed.emit(self.result)

        self.cancel_p.cancel = True

    def reset(self):
        if self.result is not None:
            if self.parent.has_scheduled_tasks():
                raise ValueError("Tasks are still scheduled")
            self.result.reset_solver()
            self.set_result_state(self.result, StateEnum.idle)
            self.parent.result_changed.emit(self.result)

    def check_available(self, result: Result):
        valid = result is not None and result.mesh_piv is not None and len(result.mesh_piv) and result.mesh_piv[0] is not None
        if not valid:
            return False
        for piv in result.mesh_piv:
            if piv is None:
                return False
        return True

    def check_status(self, result: Result) -> Tuple[str, int, int]:
        if result is None or result.solvers is None:
            return "not-available", 0, 0
        max_count = len(result.solvers)
        count = 0
        for solver in result.solvers:
            if solver is None:
                break
            count += 1
        if count < max_count:
            return "progress", count, max_count
        return "finished", max_count, max_count

    def setResult(self, result: Result):
        super().setResult(result)
        if result and result.stacks and result.stacks[0] and result.mesh_parameters:
            if result.stack_reference is None:
                self.input_reference.setValues(["next", "median", "cumul."])
                self.input_reference.setValue(result.mesh_parameters["reference_stack"])
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
        self.cancel_p = CancelSignal()
        # iterate over all stack pairs
        for i in range(len(result.mesh_piv)):
            self.parent.signal_process_status_update.emit(f"{i}/{len(result.mesh_piv)} creating mesh", f"{Path(result.output).name}")
            # and create the interpolated solver mesh
            result.solvers[i] = saenopy.interpolate_mesh(result.mesh_piv[i], displacement_list[i], mesh_parameters)
            if self.cancel_p.cancel is True:
                return "Terminated"
            self.parent.result_changed.emit(result)
        self.parent.signal_process_status_update.emit(f"{i + 1}/{len(result.mesh_piv)} creating mesh",
                                                      f"{Path(result.output).name}")

        # save the meshes
        #print("save")
        #result.save()
        #print("save done")

    def get_code(self) -> Tuple[str, str]:
        import_code = ""
        results = None
        @export_as_string
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
