import os
import time
from qtpy import QtCore, QtWidgets
import qtawesome as qta
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

from saenopy.gui.common.PipelineModule import PipelineModule, StateEnum
from saenopy.gui.common.code_export import get_code, export_as_string

import matplotlib.ticker as ticker

class OmitLast30PercentLocator(ticker.AutoLocator):
    def __call__(self):
        ticks = super(OmitLast30PercentLocator, self).__call__()
        # Safely fetch the axis limits without modifying them
        lim = self.axis.get_view_interval()
        cutoff = lim[0] + (lim[1] - lim[0]) * 0.7
        return [t for t in ticks if t < cutoff]


class CancelSignal:
    cancel = False


from saenopy.reconstruction import segment_with_params
from saenopy.surface_regularization import DEFAULT_SEG_CHANNEL
from .live_fit import solver_snapshot


def segmentation_preview_mesh(body, voxel_size):
    """Measured voxel interface for display only, without artificial edge caps."""
    from skimage.measure import marching_cubes
    body = np.asarray(body, dtype=bool)
    if body.ndim != 3 or not body.any() or body.all():
        raise ValueError("no cell/gel interface: segmentation is empty or fills the entire image; adjust threshold")
    spacing = np.asarray(voxel_size, dtype=float)[[1, 0, 2]]
    vertices, faces, _, _ = marching_cubes(body.astype(np.float32), .5, spacing=tuple(spacing))
    vertices -= (np.asarray(body.shape) - 1) / 2 * spacing
    touches_edge = any(np.any(np.take(body, [0, -1], axis=axis)) for axis in range(3))
    return vertices, faces, touches_edge


class Regularizer(PipelineModule):
    pipeline_name = "fit forces"
    iteration_finished = QtCore.Signal(object, object, int, int)
    live_field_ready = QtCore.Signal(object, int, object)

    pipeline_allow_cancel = True
    pipeline_button_name = "calculate forces"

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
                            self.input_alpha = QtShortCuts.QInputString(
                                None, "alpha", "1e10", type="exp",
                                tooltip="Regularization penalty relative to the displacement fit. "
                                        "Mesh normalization is enabled. This is the single alpha used "
                                        "for both Classic and Surface and is referenced to a 14 um mesh.")
                            self.input_step_size = QtShortCuts.QInputString(None, "step size", "0.2", type=float, tooltip="Fraction of each displacement update to apply. Default 0.2.")
                        with QtShortCuts.QHBoxLayout(None) as layout:
                            self.input_imax = QtShortCuts.QInputNumber(None, "max iterations", 300, float=False, tooltip="the maximum number of iterations after which to abort the iteration algorithm")
                            self.input_conv_crit = QtShortCuts.QInputString(None, "rel. conv. crit.", 0.01, type=float, tooltip="the convergence criterion of the iteration algorithm")

                    with QtShortCuts.QGroupBox(None, "Surface Regularisation") as (self.surface_parameters, surface_layout):
                        surface_layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
                        surface_layout.setContentsMargins(9, 6, 9, 8)
                        surface_layout.setSpacing(4)
                        self.input_surface = QtShortCuts.QInputBool(
                            None, "surface regularization", False,
                            tooltip="Regularize traction on the segmented cell surface and suppress forces "
                                    "in the surrounding gel. Surface always uses geometry normalization "
                                    "and the single 'alpha' above.")
                        # Populated from the loaded stack's channels in setResult().
                        self.input_seg_channel = QtShortCuts.QInputChoice(
                            None, "cell channel", DEFAULT_SEG_CHANNEL, values=[0], value_names=["0"],
                            tooltip="Channel showing the cell / cell stain (used for the segmentation).")
                        self.input_seg_channel.combobox.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
                        self.input_thr_method = QtShortCuts.QInputChoice(
                            None, "threshold method", "li", values=["li", "otsu", "yen"],
                            tooltip="Method for the automatic segmentation threshold. Li stops on "
                                    "non-convergence (64 iterations or 30 s, checked between iterations); "
                                    "no new mask is created on failure.")
                        self.input_thr_factor = QtShortCuts.QInputString(
                            None, "threshold factor", "0.6", type=float,
                            tooltip="Multiplied onto the automatic threshold (which is in raw intensity units). "
                                    "Lower = include dimmer parts of the cell, higher = only the bright core. "
                                    "Use 'preview segmentation' to see the effect.")
                        self.input_thr_factor.line_edit.setMinimumWidth(90)
                        self.input_dilate_layers = QtShortCuts.QInputNumber(
                            None, "surface dilation", 1, min=0, float=False,
                            tooltip="Number of mesh-node shells added to the detected surface.")
                        self.input_dilate_layers.spin_box.setMinimumWidth(90)
                        self.input_button_preview = QtShortCuts.QPushButton(
                            None, "preview segmentation", self.preview_segmentation,
                            tooltip="Segment the cell with the current channel/threshold for the current "
                                    "time step and show the surface nodes in the Forces view - without "
                                    "running the force reconstruction. Use it to tune the parameters.")

                        # QtShortCuts initially inserts controls into the current layout.
                        # Pair them in a form which wraps at narrow widths / larger fonts,
                        # instead of squeezing labels or reducing input heights.
                        surface_form = QtWidgets.QFormLayout()
                        surface_form.setContentsMargins(0, 0, 0, 0)
                        surface_form.setHorizontalSpacing(12)
                        surface_form.setVerticalSpacing(4)
                        surface_form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
                        surface_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
                        for left, right in (
                                (self.input_surface, self.input_button_preview),
                                (self.input_seg_channel, self.input_dilate_layers),
                                (self.input_thr_method, self.input_thr_factor)):
                            for control in (left, right):
                                if hasattr(control, "label"):
                                    control.label.setSizePolicy(
                                        QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
                                    control.layout().setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
                            surface_layout.removeWidget(left)
                            surface_layout.removeWidget(right)
                            surface_form.addRow(left, right)
                        surface_layout.addLayout(surface_form)
                        self.input_button_preview_text = QtWidgets.QLabel().addToLayout()
                        self.input_button_preview_text.setWordWrap(True)
                        self.input_button_preview_text.hide()

                    with QtShortCuts.QHBoxLayout():
                        self.input_button = QtShortCuts.QPushButton(None, "calculate forces", self.start_process,
                                                                    tooltip="run the force calculation")
                        self.input_button_text = QtWidgets.QLabel().addToLayout()
                        self.input_button_text.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

                        self.input_button_reset = QtShortCuts.QPushButton(None, "", self.reset, icon=qta.icon("fa5s.trash-alt"),
                                                                          tooltip="reset")
                        self.input_button_reset.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)


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
            # --- surface-restricted regularization (missing keys in older result
            # files fall back to these widgets' defaults -> backward compatible) ---
            "surface": self.input_surface,
            "seg_channel": self.input_seg_channel,
            "seg_threshold_method": self.input_thr_method,
            "seg_threshold_factor": self.input_thr_factor,
            "seg_dilate_layers": self.input_dilate_layers,
        })

        self.parent.tasks_changed.connect(self._update_preview_controls)
        self._update_preview_controls()
        self.initialize_plot()
        self.iteration_finished.connect(self.iteration_callback)
        self.live_field_ready.connect(self.show_live_field)
        self.iteration_finished.emit(None, np.ones([10, 3]), 0, None)

    def _preview_blocked(self):
        return (self.parent.has_scheduled_tasks()
                or self.get_result_state(self.result) in
                (StateEnum.scheduled, StateEnum.running, StateEnum.cancelling))

    def _update_preview_controls(self):
        self.input_button_preview.setEnabled(
            self.check_available(self.result) and not self._preview_blocked())

    def state_changed(self, result):
        if result is not None and self.get_result_state(result) not in (
                StateEnum.scheduled, StateEnum.running, StateEnum.cancelling):
            result._live_fit_active = False
            result._live_fit_solvers = {}
        super().state_changed(result)
        self._update_preview_controls()
        for viewer in (self.parent.tab4, self.parent.tab5):
            viewer.resultChanged(result)
        if result is self.result and self.get_result_state(result) == StateEnum.cancelling:
            for mapping in self.parameter_mappings:
                mapping.setDisabled(True)

    def start_process(self, x=None, result=None):
        target = result if result is not None else self.result
        if target is not None:
            for mapping in self.parameter_mappings:
                mapping.ensure_tmp_params_initialized(target)
            # GUI fits always normalize; raw Classic remains a Python option.
            target.solve_parameters_tmp["physical_normalization"] = True
            if self.get_result_state(target) not in (
                    StateEnum.scheduled, StateEnum.running, StateEnum.cancelling):
                target._live_fit_solvers = {
                    i: solver_snapshot(s) for i, s in enumerate(target.solvers or []) if s is not None}
                target._live_fit_active = True
                # Enable both viewers before the worker starts; select Forces.
                for viewer in (self.parent.tab4, self.parent.tab5):
                    viewer.resultChanged(target)
                if target is self.result:
                    self.parent.tabs.setCurrentWidget(self.parent.tab5.tab.parent())
        return super().start_process(x, result)

    def show_live_field(self, result, frame, snapshot):
        # This slot runs on the GUI thread. Finished/failed jobs discard any
        # late queued snapshot and return to the final saved solver fields.
        if not getattr(result, "_live_fit_active", False):
            return
        result._live_fit_solvers[frame] = snapshot
        for viewer in (self.parent.tab4, self.parent.tab5):
            viewer.resultChanged(result)

    def cancel_process(self):
        self.set_result_state(self.result, StateEnum.cancelling)
        self.parent.result_changed.emit(self.result)

        self.cancel_p.cancel = True

    def reset(self):
        if self.result is not None:
            if self.parent.has_scheduled_tasks():
                raise ValueError("Tasks are still scheduled")
            self.result.reset_regularisation_results()
            self.set_result_state(self.result, StateEnum.idle)
            self.parent.result_changed.emit(self.result)

    def preview_segmentation(self):
        """Segment the cell with the CURRENT channel/threshold for the time step that
        is currently shown and display the resulting surface nodes in the Forces view
        -- WITHOUT running the force reconstruction. Lets the user tune the
        segmentation parameters and see the effect immediately."""
        if self.result is None or getattr(self.result, "solvers", None) is None:
            return
        self.input_button_preview_text.show()
        if self._preview_blocked():
            self.input_button_preview_text.setText("preview unavailable while tasks are pending")
            return
        from saenopy import surface_regularization as sr
        try:
            i = self.parent.t_slider.value()
            M = self.result.solvers[i]
            if M is None or M.mesh is None or M.mesh.nodes is None:
                self.input_button_preview_text.setText("no solver mesh yet")
                return
            params = {
                "seg_channel": self.input_seg_channel.value(),
                "seg_threshold_method": self.input_thr_method.value(),
                "seg_threshold_factor": self.input_thr_factor.value(),
                "seg_dilate_layers": self.input_dilate_layers.value(),
            }
            self.input_button_preview_text.setText("segmenting ...")
            # Repaint only: processing arbitrary events here could start a fit
            # while this preview is still modifying its solver's surface mask.
            self.input_button_preview_text.repaint()
            stack, image, body, shell, used = segment_with_params(self.result, i, params)
            vertices, faces, touches_edge = segmentation_preview_mesh(body, stack.voxel_size)
            element_size = float(self.result.mesh_parameters["element_size"])
            mask = sr.surface_node_mask(
                M.mesh.nodes, shell, element_size,
                dilate_layers=int(params["seg_dilate_layers"]))
            mask &= M.mesh.regularisation_mask & M.mesh.movable
            if not mask.any():
                raise ValueError("no active surface nodes; check segmentation and mesh size")
            sr.set_surface_regularization(M, mask)
            # Transient display data, not a solver constraint or saved file field.
            M.mesh._segmentation_preview = (vertices, faces)
            status = "touches image edge; not capped" if touches_edge else "inside image bounds"
            self.input_button_preview_text.setText(
                f"threshold {used:.1f} → {int(mask.sum())} surface nodes "
                f"({mask.mean() * 100:.1f}%)\n{status}")
            viewer = self.parent.tab5
            viewer.vtk_toolbar.use_surface.setValue(True)
            self.parent.result_changed.emit(self.result)
            self.parent.tabs.setCurrentWidget(viewer.tab.parent())
            viewer.update_display()
        except Exception as err:  # keep the GUI alive on a bad channel/threshold
            self.input_button_preview_text.setText(
                f"segmentation failed: {err}\nPrevious preview, if any, is unchanged.")

    def check_available(self, result: Result):
        if result is None or result.solvers is None:
            return False
        for solver in result.solvers:
            if solver is None:
                return False
        return True

    def check_status(self, result: Result) -> Tuple[str, int, int]:
        if result is None or result.solvers is None:
            return "not-available", 0, 0
        max_count = len(result.solvers)
        count = 0
        for solver in result.solvers:
            relrec = getattr(solver, "regularisation_results", None)
            params = getattr(solver, "regularisation_parameters", None) or {}
            if relrec is None or params.get("cancelled", False):
                break
            count += 1
        if count < max_count:
            return "progress", count, max_count
        return "finished", max_count, max_count

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
        self.cancel_p = CancelSignal()
        # demo run
        if os.environ.get("DEMO") == "true":
            imax = 100
            self.parent.progressbar.setRange(0, imax)
            for i in range(len(result.solver_relrec_demo)):
                time.sleep(0.2)
                self.iteration_finished.emit(result, result.solver_relrec_demo[:i], i, imax)
            result.solvers[0].regularisation_results = result.solver_relrec_demo
            return

        i = 0
        for i in range(len(result.solvers)):
            if self.cancel_p.cancel:
                return "Terminated"
            solver = result.solvers[i]
            cancelled = (solver.regularisation_parameters or {}).get("cancelled", False)
            if solver.regularisation_results is not None and not cancelled:
                continue
            # Fit recomputes segmentation; do not show a stale preview surface.
            result.solvers[i].mesh._segmentation_preview = None
            self.parent.signal_process_status_update.emit(f"{i}/{len(result.solvers)} fitting forces", f"{Path(result.output).name}")

            from saenopy.reconstruction import fit_result

            last_display = 0.0
            def callback(M, relrec, iteration, imax, frame=i):
                nonlocal last_display
                self.iteration_finished.emit(result, np.asarray(relrec).copy(), iteration, imax)
                now = time.monotonic()
                if now - last_display >= 1.0:
                    self.live_field_ready.emit(result, frame, solver_snapshot(M))
                    last_display = now

            solver = fit_result(result, i, parameters=solve_parameters,
                       material_parameters=material_parameters, callback=callback,
                       cancel_signal=self.cancel_p, verbose=True, resume=cancelled)
            # Always publish the final field, even after early convergence or
            # cancellation within the one-second display throttle.
            self.live_field_ready.emit(result, i, solver_snapshot(solver))

            # clear the cache of the solver
            result.clear_cache(i)
            result.save()
            self.parent.result_changed.emit(result)

            if self.cancel_p.cancel is True:
                return "Terminated"

        self.parent.signal_process_status_update.emit(f"{i+1}/{len(result.solvers)} fitting forces",
                                                      f"{Path(result.output).name}")

    def setResult(self, result: Result):
        # Populate choices before ParameterMapping applies the saved channel.
        # Otherwise loading channel 1 into the initial [0] selector silently
        # leaves the preview on channel 0 while the fit still uses channel 1.
        try:
            channels = result.stacks[0].channels if (result and result.stacks) else None
            if channels:
                self.input_seg_channel.setValues(list(np.arange(len(channels))),
                                                 [str(c) for c in channels])
        except (AttributeError, IndexError, TypeError):
            pass
        super().setResult(result)
        if result is not None:
            # Only edit parameters for the NEXT fit, not the stored result.
            result.solve_parameters_tmp["physical_normalization"] = True
        self._update_preview_controls()
        self.update_plot()

    def update_plot(self):
        if self.result is None or self.result.solvers is None or len(self.result.solvers) == 0:
            return
        relrec = getattr(self.result.solvers[self.parent.t_slider.value()], "relrec", None)
        if relrec is None:
            relrec = getattr(self.result.solvers[self.parent.t_slider.value()], "regularisation_results", None)
        if relrec is not None:
            self.iteration_callback(self.result, relrec)
        else:
            self.plot_empty()

    def get_code(self) -> Tuple[str, str]:
        import_code = "import saenopy\n"
        results: Result = None

        @export_as_string
        def code(my_reg_params1, my_reg_params2):  # pragma: no cover
            from saenopy.reconstruction import fit_result
            material_parameters = my_reg_params1
            solve_parameters = my_reg_params2
            for result in results:
                for index in range(len(result.solvers)):
                    fit_result(result, index, parameters=solve_parameters,
                               material_parameters=material_parameters, verbose=True)
                    result.clear_cache(index)
                    result.save()

        # params with convert text Nones to real Nones
        export_solve_parameters = {
            key: value for key, value in self.result.solve_parameters_tmp.items()
            if key not in {"surface_area_method"}
        }
        export_solve_parameters["physical_normalization"] = True
        from saenopy.solver import DEFAULT_CG_MAXITER_FACTOR
        export_solve_parameters.setdefault("cg_maxiter_factor", DEFAULT_CG_MAXITER_FACTOR)
        export_solve_parameters.setdefault("solver_precision", 1e-18)
        data = {
            "my_reg_params1": {k: None if v == "None" else v for k, v in self.result.material_parameters_tmp.items()},
            "my_reg_params2": {k: None if v == "None" else v for k, v in export_solve_parameters.items()},
        }

        code = get_code(code, data)
        return import_code, code
