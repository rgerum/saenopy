import os
import time
from qtpy import QtWidgets, QtCore
import qtawesome as qta
import numpy as np
import tqdm
from typing import Tuple
from pathlib import Path

import saenopy
import saenopy.multigrid_helper
import saenopy.materials
from saenopy import Result
from saenopy.gui.common import QtShortCuts
from saenopy.gui.common.gui_classes import CheckAbleGroup, ProcessSimple
import saenopy.get_deformations

from saenopy.gui.common.PipelineModule import PipelineModule, StateEnum
from saenopy.gui.common.code_export import get_code, export_as_string


class CamPos:
    cam_pos_initialized = False


class DeformationDetector(PipelineModule):
    pipeline_name = "find deformations"
    use_thread = False
    signal_process_status_update = QtCore.Signal(int, int)

    current_p = None
    cancel_p = False

    pipeline_allow_cancel = True
    pipeline_button_name = "detect deformations"

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
                    with QtShortCuts.QHBoxLayout():
                        self.input_button = QtShortCuts.QPushButton(None, "detect deformations", self.start_process)
                        self.input_button_text = QtWidgets.QLabel().addToLayout()
                        self.input_button_text.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

                        self.input_button_reset = QtShortCuts.QPushButton(None, "", self.reset, icon=qta.icon("fa5s.trash-alt"), tooltip="reset")
                        self.input_button_reset.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        self.setParameterMapping("piv_parameters", {
            "window_size": self.input_win,
            "element_size": self.input_element_size,
            "signal_to_noise": self.input_signoise,
            "drift_correction": self.input_driftcorrection,
        })

        self.signal_process_status_update.connect(self.update_button)

    def cancel_process(self):
        self.set_result_state(self.result, StateEnum.cancelling)
        self.parent.result_changed.emit(self.result)

        self.cancel_p = True
        self.current_p.process.join(1)
        self.current_p.terminate()

    def reset(self):
        if self.result is not None:
            if self.parent.has_scheduled_tasks():
                raise ValueError("Tasks are still scheduled")
            self.result.reset_piv()
            self.parent.result_changed.emit(self.result)

    def update_button(self, i, i2):
        #self.input_button.setText(f"pause detection ({i}/{i2} done)")
        return

    def check_available(self, result: Result) -> bool:
        return result is not None and result.stacks is not None and len(result.stacks)

    def check_status(self, result: Result) -> Tuple[str, int, int]:
        available = result is not None and result.stacks is not None and len(result.stacks)
        if not available:
            return "not-available", 0, 0
        max_count = len(result.mesh_piv)
        count = 0
        for piv in result.mesh_piv:
            if piv is None:
                break
            count += 1
        if count < max_count:
            return "progress", count, max_count
        return "finished", max_count, max_count

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
            result.reset_piv()

        count = len(result.stacks)
        if result.stack_reference is None:
            count -= 1
        self.signal_process_status_update.emit(0, count)

        result.piv_parameters = piv_parameters

        for i in range(count):
            if result.mesh_piv[i] is not None:
                continue
            self.parent.signal_process_status_update.emit(f"{i}/{count} finding deformations", f"{Path(result.output).name}")
            p = ProcessSimple(getDeformation, (i, result, piv_parameters), {}, self.processing_progress, use_thread=self.use_thread)
            p.start()
            self.current_p = p
            return_value = p.join()
            if return_value == "Terminated":
                return "Terminated"
            if isinstance(return_value, Exception):
                raise return_value
            else:
                result.mesh_piv[i] = return_value
                result.save()
                self.parent.result_changed.emit(result)
        self.parent.signal_process_status_update.emit(f"{count}/{count} finding deformations",
                                                      f"{Path(result.output).name}")

        result.solvers = None

    def get_code(self) -> Tuple[str, str]:
        import_code = ""

        results = []

        @export_as_string
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
