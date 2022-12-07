from qtpy import QtWidgets
import numpy as np
from pyvistaqt import QtInteractor
import inspect

import saenopy
import saenopy.multigridHelper
from saenopy.gui import QtShortCuts
from saenopy.gui.gui_classes import CheckAbleGroup, QProcess, ProcessSimple
import saenopy.getDeformations
import saenopy.multigridHelper
import saenopy.materials
from saenopy.solver import Result

from typing import Tuple

from .PipelineModule import PipelineModule
from .QTimeSlider import QTimeSlider
from .VTK_Toolbar import VTK_Toolbar
from .showVectorField import showVectorField


class CamPos:
    cam_pos_initialized = False


class DeformationDetector(PipelineModule):
    pipeline_name = "find deformations"

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with QtShortCuts.QVBoxLayout(self) as layout:
            layout.setContentsMargins(0, 0, 0, 0)
            with CheckAbleGroup(self, "find deformations (piv)").addToLayout() as self.group:
                with QtShortCuts.QVBoxLayout() as layout:
                    with QtShortCuts.QHBoxLayout():
                        self.input_elementsize = QtShortCuts.QInputNumber(None, "piv element size", 15.0, step=1,
                                                                          value_changed=self.valueChanged,
                                                                          tooltip="the grid size for deformation detection")

                        self.input_win = QtShortCuts.QInputNumber(None, "window size", 30,
                                                                  value_changed=self.valueChanged, unit="μm",
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
                self.label_tab = QtWidgets.QLabel(
                    "The deformations from the piv algorithm at every window where the crosscorrelation was evaluated.").addToLayout()

                self.plotter = QtInteractor(self, auto_update=False)  # , theme=pv.themes.DocumentTheme())
                self.tab.parent().plotter = self.plotter
                self.plotter.set_background("black")
                layout.addWidget(self.plotter.interactor)

                self.vtk_toolbar = VTK_Toolbar(self.plotter, self.update_display, "deformation").addToLayout()

                self.t_slider = QTimeSlider(connected=self.update_display).addToLayout()
                self.tab.parent().t_slider = self.t_slider

        self.setParameterMapping("piv_parameter", {
            "win_um": self.input_win,
            "elementsize": self.input_elementsize,
            # "fac_overlap": self.input_overlap,
            "signoise_filter": self.input_signoise,
            "drift_correction": self.input_driftcorrection,
        })

    def check_available(self, result: Result) -> bool:
        return result is not None and result.stack is not None and len(result.stack)

    def check_evaluated(self, result: Result) -> bool:
        return self.result is not None and self.result.mesh_piv is not None and self.result.mesh_piv[0] is not None

    def update_display(self, *, plotter=None):
        if plotter is None:
            plotter = self.plotter
        cam_pos = None
        if plotter.camera_position is not None and CamPos.cam_pos_initialized is True:
            cam_pos = self.plotter.camera_position
        CamPos.cam_pos_initialized = True
        plotter.interactor.setToolTip(
            str(self.result.piv_parameter) + f"\nNodes {self.result.mesh_piv[0].R.shape[0]}\nTets {self.result.mesh_piv[0].T.shape[0]}")
        M = self.result.mesh_piv[self.t_slider.value()]

        if M is None:
            plotter.show()
            return

        if M.hasNodeVar("U_measured"):
            showVectorField(plotter, M, M.getNodeVar("U_measured"), "U_measured",
                            scalebar_max=self.vtk_toolbar.getScaleMax(), show_nan=self.vtk_toolbar.use_nans.value())

        if cam_pos is not None:
            plotter.camera_position = cam_pos

    def valueChanged(self):
        if self.check_available(self.result):
            voxel_size1 = self.result.stack[0].voxel_size
            stack_deformed = self.result.stack[0]
            overlap = 1 - (self.input_elementsize.value() / self.input_win.value())
            stack_size = np.array(stack_deformed.shape) * voxel_size1 - self.input_win.value()
            # self.label.setText(f"Deformation grid with {unit_size:.1f}μm elements.\nTotal region is {stack_size}.")
            self.label.setText(
                f"""Overlap between neighbouring windows\n(size={self.input_win.value()}µm or {(self.input_win.value() / np.array(voxel_size1)).astype(int)} px) is choosen \n to {int(overlap * 100)}% for an elementsize of {self.input_elementsize.value():.1f}μm elements.\nTotal region is {stack_size}.""")
        else:
            self.label.setText("")

    def process(self, result: Result, params: dict):

        if not isinstance(result.mesh_piv, list):
            result.mesh_piv = [None] * (len(result.stack) - 1)

        for i in range(len(result.stack) - 1):
            p = ProcessSimple(getDeformation, (i, result, params), {})
            p.start()
            result.mesh_piv[i] = p.join()

        result.solver = None

    def get_code(self) -> Tuple[str, str]:
        import_code = ""

        results = None
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
                    result.mesh_piv[i] = saenopy.get_displacements_from_stacks(result.stack[i], result.stack[i + 1],
                                                                               params["win_um"],
                                                                               params["elementsize"],
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
    import tqdm
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
        mesh_piv = saenopy.get_displacements_from_stacks(result.stack[i], result.stack[i + 1],
                                                         params["win_um"],
                                                         params["elementsize"],
                                                         params["signoise_filter"],
                                                         params["drift_correction"])
    finally:
        tqdm.tqdm.update = old_update
        tqdm.tqdm.__init__ = old_init
    return mesh_piv