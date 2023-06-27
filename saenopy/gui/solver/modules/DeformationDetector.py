import os
import time
from qtpy import QtWidgets, QtCore
import numpy as np
from pyvistaqt import QtInteractor
import inspect
import tqdm
from typing import Tuple

import saenopy
import saenopy.multigrid_helper
import saenopy.materials
from saenopy import Result
from saenopy.gui.common import QtShortCuts
from saenopy.gui.common.gui_classes import CheckAbleGroup, QProcess, ProcessSimple
import saenopy.get_deformations

from .PipelineModule import PipelineModule
from .QTimeSlider import QTimeSlider
from .VTK_Toolbar import VTK_Toolbar
from .showVectorField import showVectorField, showVectorField2
from .code_export import get_code


class CamPos:
    cam_pos_initialized = False


class DeformationDetector(PipelineModule):
    pipeline_name = "find deformations"
    use_thread = False

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with QtShortCuts.QVBoxLayout(self) as layout:
            layout.setContentsMargins(0, 0, 0, 0)
            with CheckAbleGroup(self, "find deformations (piv)", url="https://saenopy.readthedocs.io/en/latest/interface_solver.html#detect-deformations").addToLayout() as self.group:
                with QtShortCuts.QVBoxLayout() as layout:
                    with QtShortCuts.QHBoxLayout():
                        self.input_element_size = QtShortCuts.QInputNumber(None, "piv elem. size", 15.0, step=1,
                                                                          value_changed=self.valueChanged, unit="μm",
                                                                          tooltip="the grid size for deformation detection")

                        self.input_win = QtShortCuts.QInputNumber(None, "window size", 30,
                                                                  value_changed=self.valueChanged, unit="μm",
                                                                  tooltip="the size of the volume to look for a match")
                    with QtShortCuts.QHBoxLayout():
                        self.input_signoise = QtShortCuts.QInputNumber(None, "signal to noise", 1.3, step=0.1,
                                                                       tooltip="the signal to noise ratio threshold value, values below are ignore")
                        self.input_driftcorrection = QtShortCuts.QInputBool(None, "drift correction", True,
                                                                            tooltip="remove the mean displacement to correct for a global drift")
                    self.label = QtWidgets.QLabel().addToLayout()
                    self.input_button = QtShortCuts.QPushButton(None, "detect deformations", self.start_process)

        with self.parent.tabs.createTab("PIV Deformations") as self.tab:
            with QtShortCuts.QVBoxLayout() as layout:
                self.label_tab = QtWidgets.QLabel(
                    "The deformations from the piv algorithm at every window where the crosscorrelation was evaluated.").addToLayout()

                with QtShortCuts.QHBoxLayout() as layout:
                    self.plotter = QtInteractor(self, auto_update=False)  # , theme=pv.themes.DocumentTheme())
                    self.tab.parent().plotter = self.plotter
                    self.plotter.set_background("black")
                    layout.addWidget(self.plotter.interactor)

                    self.z_slider = QTimeSlider("z", self.z_slider_value_changed, "set z position",
                                                QtCore.Qt.Vertical).addToLayout()
                    self.z_slider.t_slider.valueChanged.connect(
                        lambda value: parent.shared_properties.change_property("z_slider", value, self))
                    parent.shared_properties.add_property("z_slider", self)

                self.vtk_toolbar = VTK_Toolbar(self.plotter, self.update_display, "deformation", z_slider=self.z_slider, shared_properties=self.parent.shared_properties).addToLayout()

                self.t_slider = QTimeSlider(connected=self.update_display).addToLayout()
                self.tab.parent().t_slider = self.t_slider

        self.setParameterMapping("piv_parameters", {
            "window_size": self.input_win,
            "element_size": self.input_element_size,
            "signal_to_noise": self.input_signoise,
            "drift_correction": self.input_driftcorrection,
        })

    def z_slider_value_changed(self):
        self.update_display()

    def check_available(self, result: Result) -> bool:
        return result is not None and result.stacks is not None and len(result.stacks)

    def check_evaluated(self, result: Result) -> bool:
        try:
            return self.result.mesh_piv[0] is not None
        except (AttributeError, IndexError):
            return False

    def property_changed(self, name, value):
        if name == "z_slider":
            self.z_slider.setValue(value)

    def setResult(self, result: Result):
        super().setResult(result)
        if result and result.stacks and result.stacks[0]:
            self.z_slider.setRange(0, result.stacks[0].shape[2] - 1)
            self.z_slider.setValue(result.stacks[0].shape[2] // 2)

            if result.stacks[0].channels:
                value = self.vtk_toolbar.channel_select.value()
                self.vtk_toolbar.channel_select.setValues(np.arange(len(result.stacks[0].channels)), result.stacks[0].channels)
                self.vtk_toolbar.channel_select.setValue(value)
                self.vtk_toolbar.channel_select.setVisible(True)
            else:
                self.vtk_toolbar.channel_select.setValue(0)
                self.vtk_toolbar.channel_select.setVisible(False)

    def update_display(self, *, plotter=None):
        if self.current_tab_selected is False:
            self.current_result_plotted = False
            return

        if plotter is None:
            plotter = self.plotter
        cam_pos = None
        if plotter.camera_position is not None and CamPos.cam_pos_initialized is True:
            cam_pos = self.plotter.camera_position
        CamPos.cam_pos_initialized = True
        plotter.interactor.setToolTip("")
        if self.result is None:
            mesh = None
        else:
            mesh = self.result.mesh_piv[self.t_slider.value()]

        if mesh is None:
            plotter.show()
            return

        plotter.interactor.setToolTip(
            str(self.result.piv_parameters) + f"\nNodes {mesh.nodes.shape[0]}\nTets {mesh.tetrahedra.shape[0]}")

        if mesh.displacements_measured is not None:
            showVectorField2(self, mesh, "displacements_measured")

        if cam_pos is not None:
            plotter.camera_position = cam_pos

    def valueChanged(self):
        if self.check_available(self.result):
            voxel_size1 = self.result.stacks[0].voxel_size
            stack_deformed = self.result.stacks[0]
            overlap = 1 - (self.input_element_size.value() / self.input_win.value())
            stack_size = np.array(stack_deformed.shape)[:3] * voxel_size1 - self.input_win.value()
            self.label.setText(
                f"""Overlap between neighbouring windows\n(size={self.input_win.value()}µm or {(self.input_win.value() / np.array(voxel_size1)).astype(int)} px) is choosen \n to {int(overlap * 100)}% for an element_size of {self.input_element_size.value():.1f}μm elements.\nTotal region is {stack_size}.""")
        else:
            self.label.setText("")

    def process(self, result: Result, piv_parameters: dict):
        # demo run
        if os.environ.get("DEMO") == "true":
            self.parent.progressbar.setRange(0, 10)
            for i in range(11):
                time.sleep(0.1)
                self.parent.progressbar.setValue(i)
            result.mesh_piv = result.mesh_piv_demo
            return

        if not isinstance(result.mesh_piv, list):
            result.mesh_piv = [None] * (len(result.stacks) - 1)

        count = len(result.stacks)
        if result.stack_reference is None:
            count -= 1

        for i in range(count):
            p = ProcessSimple(getDeformation, (i, result, piv_parameters), {}, self.processing_progress, use_thread=self.use_thread)
            p.start()
            return_value = p.join()
            if isinstance(return_value, Exception):
                raise return_value
            else:
                result.mesh_piv[i] = return_value

        result.solvers = None

    def get_code(self) -> Tuple[str, str]:
        import_code = ""

        results = []
        def code(my_piv_params):  # pragma: no cover
            # define the parameters for the piv deformation detection
            piv_parameters = my_piv_params

            # iterate over all the results objects
            for result in results:
                # set the parameters
                result.piv_parameters = piv_parameters
                # get count
                count = len(result.stacks)
                if result.stack_reference is None:
                    count -= 1
                # iterate over all stack pairs
                for i in range(count):
                    # get two consecutive stacks
                    if result.stack_reference is None:
                        stack1, stack2 = result.stacks[i], result.stacks[i + 1]
                    # or reference stack and one from the list
                    else:
                        stack1, stack2 = result.stack_reference, result.stacks[i]
                    # and calculate the displacement between them
                    result.mesh_piv[i] = saenopy.get_displacements_from_stacks(stack1, stack2,
                                                                               piv_parameters["window_size"],
                                                                               piv_parameters["element_size"],
                                                                               piv_parameters["signal_to_noise"],
                                                                               piv_parameters["drift_correction"])
                # save the displacements
                result.save()

        data = {
            "my_piv_params": self.result.piv_parameters_tmp
        }

        code = get_code(code, data)

        return import_code, code



def getDeformation(progress, i, result, params):
    t = tqdm.tqdm
    n = tqdm.tqdm.__new__
    old_update = tqdm.tqdm.update
    old_init = tqdm.tqdm.__init__
    def wrap_update(update):
        def do_update(self, n):
            update(self, n)
            progress.put((self.n, self.total))
        return do_update
    tqdm.tqdm.update = wrap_update(tqdm.tqdm.update)
    def wrap_init(init):
        def do_init(self, *args, **kwargs):
            init(self, *args, **kwargs)
            progress.put((0, self.total))
        return do_init
    tqdm.tqdm.__init__ = wrap_init(tqdm.tqdm.__init__)
    #tqdm.tqdm.__new__ = lambda cls, iter: progress.put(iter)

    try:
        if result.stack_reference is None:
            stack1, stack2 = result.stacks[i], result.stacks[i + 1]
        else:
            stack1, stack2 = result.stack_reference, result.stacks[i]
        mesh_piv = saenopy.get_displacements_from_stacks(stack1, stack2,
                                                         params["window_size"],
                                                         params["element_size"],
                                                         params["signal_to_noise"],
                                                         params["drift_correction"])
    except Exception as err:
        return err
    finally:
        tqdm.tqdm.update = old_update
        tqdm.tqdm.__init__ = old_init
    return mesh_piv
