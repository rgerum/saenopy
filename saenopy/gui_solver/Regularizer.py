from qtpy import QtCore, QtWidgets
import numpy as np
from pyvistaqt import QtInteractor

import inspect

import saenopy
import saenopy.multigridHelper
from saenopy.gui import QtShortCuts
from saenopy.gui.gui_classes import CheckAbleGroup, MatplotlibWidget, NavigationToolbar
import saenopy.getDeformations
import saenopy.multigridHelper
import saenopy.materials
from saenopy import Result

from typing import Tuple

from .PipelineModule import PipelineModule
from .QTimeSlider import QTimeSlider
from .VTK_Toolbar import VTK_Toolbar
from .showVectorField import showVectorField, getVectorFieldImage
from .DeformationDetector import CamPos


class Regularizer(PipelineModule):
    pipeline_name = "fit forces"
    iteration_finished = QtCore.Signal(object, object, int, int)

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with QtShortCuts.QVBoxLayout(self) as layout:
            layout.setContentsMargins(0, 0, 0, 0)
            with CheckAbleGroup(self, "fit forces (regularize)").addToLayout() as self.group:

                with QtShortCuts.QVBoxLayout() as main_layout:
                    with QtShortCuts.QGroupBox(None, "Material Parameters") as self.material_parameters:
                        with QtShortCuts.QHBoxLayout() as layout2:
                            self.input_k = QtShortCuts.QInputString(None, "k", "1645", type=float)
                            self.input_d0 = QtShortCuts.QInputString(None, "d_0", "0.0008", type=float)
                            self.input_lamda_s = QtShortCuts.QInputString(None, "λ_s", "0.0075", type=float)
                            self.input_ds = QtShortCuts.QInputString(None, "d_s", "0.033", type=float)

                    with QtShortCuts.QGroupBox(None, "Regularisation Parameters") as self.material_parameters:
                        with QtShortCuts.QHBoxLayout(None) as layout:
                            self.input_alpha = QtShortCuts.QInputString(None, "alpha", "1e10", type="exp")
                            self.input_stepper = QtShortCuts.QInputString(None, "stepper", "0.33", type=float)
                        with QtShortCuts.QHBoxLayout(None) as layout:
                            self.input_imax = QtShortCuts.QInputNumber(None, "i_max", 100, float=False)
                            self.input_conv_crit = QtShortCuts.QInputString(None, "rel_conv_crit", 0.01, type=float)

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

                self.vtk_toolbar = VTK_Toolbar(self.plotter, self.update_display, center=True, shared_properties=self.parent.shared_properties).addToLayout()

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
            "rel_conv_crit": self.input_conv_crit,
        })

        self.iteration_finished.connect(self.iteration_callback)
        self.iteration_finished.emit(None, np.ones([10, 3]), 0, None)

    def z_slider_value_changed(self):
        self.update_display()

    def check_available(self, result: Result):
        try:
            return self.result.solver[0] is not None
        except (AttributeError, IndexError):
            return False

    def check_evaluated(self, result: Result) -> bool:
        try:
            if self.result is not None and self.result.solver is not None:
                relrec = getattr(self.result.solver[self.t_slider.value()], "relrec", None)
                if relrec is not None:
                    return True
            return getattr(self.result.solver[0], "regularisation_results", None) is not None
        except (AttributeError, IndexError):
            return False

    def iteration_callback(self, result, relrec, i=0, imax=None):
        if imax is not None:
            self.parent.progressbar.setRange(0, imax)
            self.parent.progressbar.setValue(i)
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
                try:
                    self.canvas.figure.tight_layout()
                except np.linalg.LinAlgError:
                    pass
                self.canvas.draw()

    def process(self, result: Result, params: dict):
        for i in range(len(result.solver)):
            M = result.solver[i]

            def callback(M, relrec, i, imax):
                self.iteration_finished.emit(result, relrec, i, imax)

            M.setMaterialModel(saenopy.materials.SemiAffineFiberMaterial(
                               params["k"],
                               params["d0"] if params["d0"] != "None" else None,
                               params["lambda_s"] if params["lambda_s"] != "None" else None,
                               params["ds"] if params["ds"] != "None" else None,
                               ))

            M.solve_regularized(stepper=params["stepper"], i_max=params["i_max"],
                                alpha=params["alpha"], rel_conv_crit=params["rel_conv_crit"],
                                callback=callback, verbose=True)

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

    def update_display(self):
        if self.current_tab_selected is False:
            self.current_result_plotted = False
            return

        if self.check_evaluated(self.result):
            cam_pos = None
            if self.plotter.camera_position is not None and CamPos.cam_pos_initialized is True:
                cam_pos = self.plotter.camera_position
            CamPos.cam_pos_initialized = True
            self.plotter.interactor.setToolTip(str(self.result.solve_parameter)+f"\nNodes {self.result.solver[self.t_slider.value()].R.shape[0]}\nTets {self.result.solver[self.t_slider.value()].T.shape[0]}")
            M = self.result.solver[self.t_slider.value()]
            center = None
            if self.vtk_toolbar.use_center.value() is True:
                center = M.getCenter(mode="Force")
            display_image = getVectorFieldImage(self)
            showVectorField(self.plotter, M, -M.f * M.reg_mask[:, None], "f", center=center, factor=0.15, scalebar_max=self.vtk_toolbar.getScaleMax(), show_nan=self.vtk_toolbar.use_nans.value(), display_image=display_image, show_grid=self.vtk_toolbar.show_grid.value(), stack_shape=self.result.stack[0].shape[:3]*self.result.stack[0].voxel_size)
            if cam_pos is not None:
                self.plotter.camera_position = cam_pos
            relrec = getattr(self.result.solver[self.t_slider.value()], "relrec", None)
            if relrec is None:
                relrec = self.result.solver[self.t_slider.value()].regularisation_results
            self.iteration_callback(self.result, relrec)
        else:
            self.plotter.interactor.setToolTip("")

    def get_code(self) -> Tuple[str, str]:
        import_code = "import saenopy\n"
        results = None
        def code(my_reg_params):
            # define the parameters to generate the solver mesh and interpolate the piv mesh onto it
            params = my_reg_params

            # iterate over all the results objects
            for result in results:
                result.solve_parameter = params
                for M in result.solver:
                    # set the material model
                    M.setMaterialModel(saenopy.materials.SemiAffineFiberMaterial(
                        params["k"],
                        params["d0"],
                        params["lambda_s"],
                        params["ds"],
                    ))
                    # find the regularized force solution
                    M.solve_regularized(stepper=params["stepper"], i_max=params["i_max"], alpha=params["alpha"], rel_conv_crit=params["rel_conv_crit"], verbose=True)
                # save the forces
                result.save()

        params = self.result.solve_parameter_tmp
        if params["d0"] == "None":
            params["d0"] = None
        if params["lambda_s"] == "None":
            params["lambda_s"] = None
        if params["ds"] == "None":
            params["ds"] = None
        data = {
            "my_reg_params": params,
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

