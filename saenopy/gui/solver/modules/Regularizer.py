import os
import time
from qtpy import QtCore, QtWidgets
import numpy as np
from typing import Tuple
from pathlib import Path

import saenopy
import saenopy.multigrid_helper
from saenopy import Result
import saenopy.get_deformations
import saenopy.materials
from saenopy.gui.common import QtShortCuts
from saenopy.gui.common.gui_classes import CheckAbleGroup, MatplotlibWidget

from saenopy.gui.common.PipelineModule import PipelineModule
from saenopy.gui.common.code_export import get_code, export_as_string

import matplotlib.ticker as ticker

class OmitLast30PercentLocator(ticker.AutoLocator):
    def __call__(self):
        ticks = super(OmitLast30PercentLocator, self).__call__()
        # Safely fetch the axis limits without modifying them
        lim = self.axis.get_view_interval()
        cutoff = lim[0] + (lim[1] - lim[0]) * 0.7
        return [t for t in ticks if t < cutoff]


class Regularizer(PipelineModule):
    pipeline_name = "fit forces"
    iteration_finished = QtCore.Signal(object, object, int, int)

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with QtShortCuts.QVBoxLayout(self) as layout:
            layout.setContentsMargins(0, 0, 0, 0)
            with CheckAbleGroup(self, "fit forces (regularize)", url="https://saenopy.readthedocs.io/en/latest/interface_solver.html#fit-deformations-and-calculate-forces").addToLayout() as self.group:

                with QtShortCuts.QVBoxLayout() as main_layout:
                    with QtShortCuts.QGroupBox(None, "Material Parameters") as self.material_parameters:
                        with QtShortCuts.QHBoxLayout() as layout2:
                            self.input_k = QtShortCuts.QInputString(None, "k", "1645", type=float, tooltip="the stiffness of the material's fibers")
                            self.input_d_0 = QtShortCuts.QInputString(None, "d_0", "0.0008", type=float, tooltip="the bluckling strength of the material's fibers")
                            self.input_lamda_s = QtShortCuts.QInputString(None, "λ_s", "0.0075", type=float, tooltip="the length at which strain stiffening of the material's fibers starts")
                            self.input_d_s = QtShortCuts.QInputString(None, "d_s", "0.033", type=float, tooltip="the strain stiffening strength of the material's fibers")

                    with QtShortCuts.QGroupBox(None, "Regularisation Parameters") as self.material_parameters:
                        self.input_previous_t_as_start = QtShortCuts.QInputBool(None, "use previous time steps deformation field", True,
                                                                tooltip="wether to use the previous time steps deformation field as a starting value for the next regularisation")
                        with QtShortCuts.QHBoxLayout(None) as layout:
                            self.input_alpha = QtShortCuts.QInputString(None, "alpha", "1e10", type="exp", tooltip="the strength of the regularisation (higher values mean weaker forces)")
                            self.input_step_size = QtShortCuts.QInputString(None, "step size", "0.33", type=float, tooltip="the step with of the iteration algorithm")
                        with QtShortCuts.QHBoxLayout(None) as layout:
                            self.input_imax = QtShortCuts.QInputNumber(None, "max iterations", 100, float=False, tooltip="the maximum number of iterations after which to abort the iteration algorithm")
                            self.input_conv_crit = QtShortCuts.QInputString(None, "rel. conv. crit.", 0.01, type=float, tooltip="the convergence criterion of the iteration algorithm")

                    self.input_button = QtShortCuts.QPushButton(None, "calculate forces", self.start_process, tooltip="run the force calculation")

                    self.canvas = MatplotlibWidget(self)
                    self.parent.results_pane.addWidget(QtWidgets.QLabel("convergence of force fit"))
                    self.parent.results_pane.addWidget(self.canvas, 1)
                    #NavigationToolbar(self.canvas, self).addToLayout()

        self.setParameterMapping("material_parameters", {
            "k": self.input_k,
            "d_0": self.input_d_0,
            "lambda_s": self.input_lamda_s,
            "d_s": self.input_d_s,
        })
        self.setParameterMapping("solve_parameters", {
            "alpha": self.input_alpha,
            "step_size": self.input_step_size,
            "max_iterations": self.input_imax,
            "rel_conv_crit": self.input_conv_crit,
            "prev_t_as_start": self.input_previous_t_as_start,
        })

        self.initialize_plot()
        self.iteration_finished.connect(self.iteration_callback)
        self.iteration_finished.emit(None, np.ones([10, 3]), 0, None)

    def check_available(self, result: Result):
        try:
            return self.result.solvers[0] is not None
        except (AttributeError, IndexError, TypeError):
            return False

    def initialize_plot(self):
        self.canvas.figure.axes[0].cla()
        self.canvas_text = self.canvas.figure.axes[0].text(0.5, 0.5, "no fit yet", ha="center",
                                        transform=self.canvas.figure.axes[0].transAxes)
        self.canvas_plot = self.canvas.figure.axes[0].semilogy([[0,1]], label="total loss")[0]
        self.canvas.figure.axes[0].spines["top"].set_visible(False)
        self.canvas.figure.axes[0].spines["right"].set_visible(False)

        self.canvas.figure.axes[0].text(0, 1, "error  ", ha="right", transform=self.canvas.figure.axes[0].transAxes)
        self.canvas.figure.axes[0].text(1, 0, "\n\niteration", ha="right", va="center",
                                        transform=self.canvas.figure.axes[0].transAxes)
        self.canvas.figure.axes[0].xaxis.set_major_locator(OmitLast30PercentLocator())  # Set default automatic locator
        try:
            self.canvas.figure.tight_layout(pad=0)
        except np.linalg.LinAlgError:
            pass
        QtCore.QTimer.singleShot(0, self.canvas.draw)

    def iteration_callback(self, result, relrec, i=0, imax=None):
        if imax is not None:
            self.parent.progressbar.setRange(0, imax)
            self.parent.progressbar.setValue(i)
        if result is self.result:
            #for i in range(self.parent.tabs.count()):
            #    if self.parent.tabs.widget(i) == self.tab.parent():
            #        self.parent.tabs.setTabEnabled(i, self.check_evaluated(result))
            if self.canvas is not None:
                relrec = np.array(relrec).reshape(-1, 3)
                self.canvas_plot.set_xdata(np.arange(len(relrec[:, 0])))
                self.canvas_plot.set_ydata(relrec[:, 0])
                self.canvas_plot.set_visible(True)
                self.canvas_text.set_visible(False)
                self.canvas.figure.axes[0].set_xlim(0, len(relrec[:, 0])+0.1)

                self.canvas.figure.axes[0].relim()  # Recompute limits based on data
                self.canvas.figure.axes[0].autoscale_view()  # Apply updated limits
                try:
                    self.canvas.figure.tight_layout(pad=0)
                except np.linalg.LinAlgError:
                    pass
                QtCore.QTimer.singleShot(0, self.canvas.draw_idle)  # Use Qt timer to prevent recursive repaints

    def plot_empty(self):
        self.canvas_plot.set_visible(False)
        self.canvas_text.set_visible(True)
        QtCore.QTimer.singleShot(0, self.canvas.draw_idle)

    def process(self, result: Result, material_parameters: dict, solve_parameters: dict):
        # demo run
        if os.environ.get("DEMO") == "true":
            imax = 100
            self.parent.progressbar.setRange(0, imax)
            for i in range(len(result.solver_relrec_demo)):
                time.sleep(0.2)
                self.iteration_finished.emit(result, result.solver_relrec_demo[:i], i, imax)
            result.solvers[0].regularisation_results = result.solver_relrec_demo
            return

        for i in range(len(result.solvers)):
            self.parent.signal_process_status_update.emit(f"{i + 1}/{len(result.solvers)} fitting forces", f"{Path(result.output).name}")

            print(f"Current Timstep: {i}")
            M = result.solvers[i]

            if i > 0 and solve_parameters["prev_t_as_start"]:
                M.mesh.displacements[:] = result.solvers[i-1].mesh.displacements.copy()

            def callback(M, relrec, i, imax):
                self.iteration_finished.emit(result, relrec, i, imax)

            M.set_material_model(saenopy.materials.SemiAffineFiberMaterial(
                               material_parameters["k"],
                               material_parameters["d_0"] if material_parameters["d_0"] != "None" else None,
                               material_parameters["lambda_s"] if material_parameters["lambda_s"] != "None" else None,
                               material_parameters["d_s"] if material_parameters["d_s"] != "None" else None,
                               ))

            M.solve_regularized(step_size=solve_parameters["step_size"], max_iterations=solve_parameters["max_iterations"],
                                alpha=solve_parameters["alpha"], rel_conv_crit=solve_parameters["rel_conv_crit"],
                                callback=callback, verbose=True)

            # clear the cache of the solver
            result.clear_cache(i)
            result.save()

    def setResult(self, result: Result):
        super().setResult(result)
        self.update_plot()

    def update_plot(self):
        if self.check_evaluated(self.result):
            relrec = getattr(self.result.solvers[self.t_slider.value()], "relrec", None)
            if relrec is None:
                relrec = self.result.solvers[self.t_slider.value()].regularisation_results
            self.iteration_callback(self.result, relrec)
        else:
            self.plot_empty()

    def get_code(self) -> Tuple[str, str]:
        import_code = "import saenopy\n"
        results: Result = None

        @export_as_string
        def code(my_reg_params1, my_reg_params2):  # pragma: no cover
            # define the parameters to generate the solver mesh and interpolate the piv mesh onto it
            material_parameters = my_reg_params1
            solve_parameters = my_reg_params2

            # iterate over all the results objects
            for result in results:
                result.material_parameters = material_parameters
                result.solve_parameters = solve_parameters
                for index, M in enumerate(result.solvers):
                    # optionally copy the displacement field from the previous time step as a starting value
                    if index > 0 and solve_parameters["prev_t_as_start"]:
                        M.mesh.displacements[:] = result.solvers[index - 1].mesh.displacements.copy()

                    # set the material model
                    M.set_material_model(saenopy.materials.SemiAffineFiberMaterial(
                        material_parameters["k"],
                        material_parameters["d_0"],
                        material_parameters["lambda_s"],
                        material_parameters["d_s"],
                    ))
                    # find the regularized force solution
                    M.solve_regularized(alpha=solve_parameters["alpha"], step_size=solve_parameters["step_size"],
                                        max_iterations=solve_parameters["max_iterations"], rel_conv_crit=solve_parameters["rel_conv_crit"],
                                        verbose=True)
                    # save the forces
                    result.save()
                    # clear the cache of the solver
                    result.clear_cache(index)

        # params with convert text Nones to real Nones
        data = {
            "my_reg_params1": {k: None if v == "None" else v for k, v in self.result.material_parameters_tmp.items()},
            "my_reg_params2": {k: None if v == "None" else v for k, v in self.result.solve_parameters_tmp.items()},
        }

        code = get_code(code, data)
        return import_code, code
