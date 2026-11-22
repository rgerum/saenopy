"""Regression coverage for review findings in the reconstruction GUI."""
import copy
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np
import pytest

from test_release import example_result

pytestmark = pytest.mark.usefixtures("isolated_qt_settings")


def series_result(count=3):
    result = example_result()
    result.solvers.extend(copy.deepcopy(result.solvers[0]) for _ in range(count - 1))
    result.output = "unused.saenopy"
    result.save = lambda: None
    return result


def process_owner():
    signal = SimpleNamespace(emit=lambda *args: None)
    return SimpleNamespace(parent=SimpleNamespace(signal_process_status_update=signal,
                                                  result_changed=signal),
                           iteration_finished=signal, live_field_ready=signal)


def completed_parameters():
    return dict(step_size=.2, solver_precision=1e-18, max_iterations=2,
                rel_conv_crit=.01, alpha=1e8, method="huber", cancelled=False)


def test_segmentation_retry_preserves_completed_frames_with_same_settings(monkeypatch):
    import saenopy.reconstruction as reconstruction
    from saenopy.gui.solver.modules.Regularizer import Regularizer
    from saenopy.surface_regularization import SegmentationError

    result, gui = series_result(), process_owner()
    params = dict(result.solve_parameters, surface=True)
    prepared, fitted = [], []
    fail = True

    def prepare(result, index, parameters):
        prepared.append(index)
        if fail and index == 1:
            raise SegmentationError("Li threshold timed out")
        return {"frame": index}

    def fit(result, index, prepared_surface, **kwargs):
        assert prepared_surface == {"frame": index}
        fitted.append(index)
        solver = result.solvers[index]
        solver.mesh.forces[:] = index + 1
        solver.regularisation_results = np.ones((2, 3))
        solver.regularisation_parameters = completed_parameters()
        return solver

    monkeypatch.setattr(reconstruction, "prepare_surface", prepare)
    monkeypatch.setattr(reconstruction, "fit_result", fit)
    with pytest.raises(SegmentationError, match="timed out"):
        Regularizer.process(gui, result, result.material_parameters, params)
    first = result.solvers[0]
    assert result._segmentation_error == "Li threshold timed out"
    assert fitted == [0]
    assert result.solvers[1].regularisation_results is None
    fail = False
    Regularizer.process(gui, result, result.material_parameters, params)
    assert prepared == [0, 1, 1, 2]
    assert fitted == [0, 1, 2]
    assert result.solvers[0] is first
    np.testing.assert_array_equal(first.mesh.forces, np.ones((4, 3)))
    assert result._segmentation_error is None


@pytest.mark.parametrize("change_after_reload", [False, True])
def test_changed_retry_keeps_old_pending_frames_incomplete_after_cancel(
        monkeypatch, tmp_path, gui_window, change_after_reload):
    from saenopy import Result
    import saenopy.reconstruction as reconstruction
    from saenopy.gui.solver.modules.Regularizer import Regularizer

    result, gui = series_result(), process_owner()
    regularizer = gui_window.sub_module_regularize
    regularizer.setResult(result)
    for mapping in regularizer.parameter_mappings:
        setattr(result, mapping.params_name, dict(getattr(result, mapping.params_name + "_tmp")))
    result.output = str(tmp_path / "retry.saenopy")
    result.save = lambda: Result.save(result)
    for solver in result.solvers:
        solver.regularisation_results = np.ones((2, 3))
        solver.regularisation_parameters = completed_parameters()
        solver.mesh.movable[0] = False
        solver.mesh.displacements[0] = (1., 2., 3.)
    original = [solver for solver in result.solvers]
    result._retry_all_frames = True
    fitted = []

    def fit(result, index, cancel_signal, **kwargs):
        fitted.append(index)
        solver = result.solvers[index]
        assert not solver.mesh.movable[0]
        np.testing.assert_array_equal(solver.mesh.displacements[0], (1., 2., 3.))
        solver.regularisation_results = np.ones((2, 3))
        solver.regularisation_parameters = completed_parameters()
        if len(fitted) == 1:
            cancel_signal.cancel = True
        return solver

    monkeypatch.setattr(reconstruction, "fit_result", fit)
    assert Regularizer.process(gui, result, result.material_parameters, result.solve_parameters) == "Terminated"
    assert fitted == [0]
    for index in (1, 2):
        assert result.solvers[index] is original[index]
        assert result.solvers[index].regularisation_results is None
        assert result.solvers[index].regularisation_parameters is None
    assert Regularizer.check_status(None, result) == ("progress", 1, 3)
    loaded = Result.load(result.output)
    assert Regularizer.check_status(None, loaded) == ("progress", 1, 3)
    for solver in loaded.solvers[1:]:
        assert solver.regularisation_results is None
        assert not solver.mesh.movable[0]
        np.testing.assert_array_equal(solver.mesh.displacements[0], (1., 2., 3.))
    assert not hasattr(loaded, "_fit_attempt_parameters")
    regularizer.setResult(loaded)
    loaded.material_parameters_tmp["d_0"] = "None"
    assert not regularizer._retry_settings_changed(loaded)
    if change_after_reload:
        regularizer.input_k.setValue(1234., send_signal=True)
    assert regularizer.input_button.text() == ("retry all frames" if change_after_reload else "continue")
    queued = []
    monkeypatch.setattr(gui_window, "addTask", lambda *args: queued.append(args))
    regularizer.start_process()
    assert loaded._retry_all_frames is change_after_reload
    assert loaded.solvers[0].regularisation_results is not None
    # Exercise the real base worker, which stores the new settings before it
    # calls Regularizer.process; the all-frame decision must survive that.
    queued[0][0](loaded, queued[0][2])
    assert fitted == ([0, 0, 1, 2] if change_after_reload else [0, 1, 2])
    assert loaded.solve_parameters_state == "finished"


@pytest.fixture
def gui_window(monkeypatch):
    from qtpy import QtCore, QtGui, QtWidgets
    from pyvistaqt import QtInteractor
    import qtawesome
    from saenopy.gui.solver.modules.BatchEvaluate import BatchEvaluate

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(QtInteractor, "render", lambda self: None)
    monkeypatch.setattr(qtawesome, "icon", lambda *args, **kwargs: QtGui.QIcon())
    window = BatchEvaluate()
    yield window
    for _ in range(3):
        app.processEvents()
    for plotter in window.findChildren(QtInteractor):
        plotter.close()
    window.close()
    window.deleteLater()
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    app.processEvents()


@pytest.mark.parametrize("changed", [False, True])
def test_failed_segmentation_retry_button_and_queue_preserve_fields(gui_window, monkeypatch, changed):
    from saenopy.gui.common.PipelineModule import StateEnum
    window = gui_window
    regularizer = window.sub_module_regularize
    result = series_result(2)
    result.solvers[0].regularisation_results = np.ones((2, 3))
    regularizer.setResult(result)
    result._segmentation_error = "Li threshold timed out"
    result._fit_attempt_parameters = {
        mapping.params_name: dict(getattr(result, mapping.params_name + "_tmp"))
        for mapping in regularizer.parameter_mappings}
    regularizer.set_result_state(result, StateEnum.failed)
    regularizer.state_changed(result)
    assert regularizer.input_button.isEnabled()
    assert regularizer.input_button.text() == "continue"
    if changed:
        regularizer.input_thr_method.combobox.setCurrentIndex(
            regularizer.input_thr_method.values.index("otsu"))
    assert regularizer.input_button.text() == ("retry all frames" if changed else "continue")
    assert "Li threshold timed out" in regularizer.input_button_preview_text.text()
    queued = []
    monkeypatch.setattr(window, "addTask", lambda *args: queued.append(args))
    before = result.solvers[0].regularisation_results
    regularizer.start_process()
    assert len(queued) == 1
    assert result._retry_all_frames is changed
    assert result.solvers[0].regularisation_results is before
    assert "Li threshold timed out" in regularizer.input_button_preview_text.text()


@pytest.mark.parametrize("widget_name,value", [
    ("input_step_size", .35), ("input_surface", True), ("input_k", 1234.),
])
def test_partial_fit_defaults_and_edits_survive_dataset_switch(gui_window, widget_name, value):
    regularizer = gui_window.sub_module_regularize
    other = example_result()
    other.solve_parameters.update(step_size=.4, prev_t_as_start=True,
                                  seg_threshold_method="otsu")
    regularizer.setResult(other)
    result = series_result(2)
    saved = dict(result.solve_parameters)
    for solver in result.solvers:
        solver.regularisation_results = np.ones((2, 3))
        solver.regularisation_parameters = {}
    result.solvers[-1].regularisation_parameters["cancelled"] = True
    regularizer.setResult(result)
    assert result.solve_parameters_tmp["step_size"] == .2
    assert result.solve_parameters_tmp["prev_t_as_start"] is False
    assert result.solve_parameters_tmp["seg_threshold_method"] == "li"
    assert regularizer.input_button.text() == "continue"
    widget = getattr(regularizer, widget_name)
    original = widget.value()
    widget.setValue(value, send_signal=True)
    assert regularizer.input_button.text() == "retry all frames"
    regularizer.setResult(other)
    regularizer.setResult(result)
    assert regularizer.input_button.text() == "retry all frames"
    widget.setValue(original, send_signal=True)
    assert regularizer.input_button.text() == "continue"
    assert result.solve_parameters == saved


@pytest.mark.parametrize("changed", [False, True])
def test_unselected_partial_batch_result_uses_same_retry_defaults(gui_window, monkeypatch, changed):
    regularizer = gui_window.sub_module_regularize
    regularizer.setResult(example_result())
    result = series_result(2)
    result.solvers[0].regularisation_results = np.ones((2, 3))
    result.solve_parameters_tmp = dict(result.solve_parameters)
    if changed:
        result.solve_parameters_tmp["alpha"] *= 2
    queued = []
    monkeypatch.setattr(gui_window, "addTask", lambda *args: queued.append(args))
    regularizer.start_process(result=result)
    assert len(queued) == 1
    assert result._retry_all_frames is changed
    assert result.solve_parameters_tmp["prev_t_as_start"] is False


@pytest.mark.parametrize("state", ["", "finished"])
@pytest.mark.parametrize("widget_name,value", [("input_alpha", 2e8), ("input_k", 1234.)])
def test_completed_series_edits_recalculate_every_frame(gui_window, monkeypatch, state, widget_name, value):
    import saenopy.reconstruction as reconstruction

    regularizer = gui_window.sub_module_regularize
    result = series_result(2)
    for solver in result.solvers:
        solver.regularisation_results = np.ones((2, 3))
        solver.regularisation_parameters = completed_parameters()
    regularizer.set_result_state(result, state)
    regularizer.setResult(result)
    assert not regularizer.input_button.isEnabled()
    widget = getattr(regularizer, widget_name)
    old_value = widget.value()
    widget.setValue(value, send_signal=True)
    assert regularizer.input_button.text() == "recalculate all frames"
    assert regularizer.input_button.isEnabled()
    widget.setValue(old_value, send_signal=True)
    assert not regularizer.input_button.isEnabled()
    widget.setValue(value, send_signal=True)
    queued, fitted = [], []
    monkeypatch.setattr(gui_window, "addTask", lambda *args: queued.append(args))

    def fit(result, index, parameters, material_parameters, **kwargs):
        fitted.append((index, parameters["alpha"], material_parameters["k"]))
        solver = result.solvers[index]
        assert solver.regularisation_results is None
        solver.regularisation_results = np.ones((2, 3))
        solver.regularisation_parameters = completed_parameters()
        return solver

    monkeypatch.setattr(reconstruction, "fit_result", fit)
    original_results = [solver.regularisation_results for solver in result.solvers]
    regularizer.start_process()
    assert result._retry_all_frames
    for solver, original in zip(result.solvers, original_results):
        assert solver.regularisation_results is original
    queued[0][0](result, queued[0][2])
    expected_alpha = value if widget_name == "input_alpha" else 1e8
    expected_k = value if widget_name == "input_k" else 1000.
    assert fitted == [(0, expected_alpha, expected_k), (1, expected_alpha, expected_k)]
    assert result._fit_attempt_parameters["solve_parameters"]["alpha"] == expected_alpha
    assert result._fit_attempt_parameters["material_parameters"]["k"] == expected_k
    assert result.solve_parameters["alpha"] == expected_alpha
    assert result.material_parameters["k"] == expected_k
    assert not regularizer.input_button.isEnabled()


@pytest.mark.parametrize("refresh_mesh", [False, True])
def test_complete_batch_noop_preserves_metadata_but_fits_replaced_meshes(gui_window, monkeypatch, refresh_mesh):
    import saenopy.reconstruction as reconstruction

    regularizer = gui_window.sub_module_regularize
    regularizer.setResult(example_result())
    result = series_result(2)
    for solver in result.solvers:
        solver.regularisation_results = np.ones((2, 3))
        solver.regularisation_parameters = completed_parameters()
    original_solve, original_material = result.solve_parameters, result.material_parameters
    result._fit_attempt_parameters = dict(solve_parameters=dict(original_solve),
                                          material_parameters=dict(original_material))
    original_attempt = result._fit_attempt_parameters
    queued, fitted, saved = [], [], []
    result.save = lambda: saved.append(True)
    monkeypatch.setattr(gui_window, "addTask", lambda *args: queued.append(args))

    def fit(result, index, **kwargs):
        fitted.append(index)
        solver = result.solvers[index]
        solver.regularisation_results = np.ones((2, 3))
        solver.regularisation_parameters = completed_parameters()
        return solver

    monkeypatch.setattr(reconstruction, "fit_result", fit)
    regularizer.start_process(result=result)
    assert not result._retry_all_frames
    if refresh_mesh:
        # An earlier queued MeshCreator step replaces the completed solvers.
        result.solvers = series_result(2).solvers
    queued[0][0](result, queued[0][2])
    assert result.solve_parameters_state == "finished"
    assert fitted == ([0, 1] if refresh_mesh else [])
    if not refresh_mesh:
        assert saved == [True]
        assert result.solve_parameters is original_solve
        assert result.material_parameters is original_material
        assert result._fit_attempt_parameters is original_attempt


@pytest.mark.parametrize("fail_retry", [False, True])
def test_final_frame_save_retry_persists_without_refitting(gui_window, monkeypatch, tmp_path, fail_retry):
    from qtpy import QtWidgets
    from saenopy import Result
    import saenopy.reconstruction as reconstruction

    regularizer = gui_window.sub_module_regularize
    result = series_result(2)
    result.output = str(tmp_path / "save-retry.saenopy")
    regularizer.setResult(result)
    queued, fitted, errors, finished = [], [], [], []
    monkeypatch.setattr(gui_window, "addTask", lambda *args: queued.append(args))
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args: None)
    regularizer.processing_error.connect(errors.append)
    regularizer.processing_finished.connect(lambda: finished.append(True))
    remaining_failures = 2 if fail_retry else 1

    def save():
        nonlocal remaining_failures
        if len(fitted) == 2 and remaining_failures:
            remaining_failures -= 1
            raise OSError("temporary save failure")
        Result.save(result)

    def fit(result, index, **kwargs):
        fitted.append(index)
        solver = result.solvers[index]
        solver.mesh.forces[:] = index + 1
        solver.regularisation_results = np.ones((2, 3))
        solver.regularisation_parameters = completed_parameters()
        return solver

    def run():
        regularizer.start_process(result=result)
        task = queued.pop(0)
        task[0](result, task[2])

    result.save = save
    monkeypatch.setattr(reconstruction, "fit_result", fit)
    run()
    assert fitted == [0, 1]
    assert result.solve_parameters_state == "failed"
    assert errors == ["temporary save failure"]
    assert finished == []
    assert regularizer.check_status(result) == ("finished", 2, 2)
    assert regularizer.check_status(Result.load(result.output)) == ("progress", 1, 2)
    original_solve, original_material = result.solve_parameters, result.material_parameters
    original_attempt = result._fit_attempt_parameters

    run()
    if fail_retry:
        assert result.solve_parameters_state == "failed"
        assert errors == ["temporary save failure"] * 2
        assert finished == []
        assert regularizer.check_status(Result.load(result.output)) == ("progress", 1, 2)
        run()
    assert result.solve_parameters_state == "finished"
    assert finished == [True]
    assert fitted == [0, 1]
    assert result.solve_parameters is original_solve
    assert result.material_parameters is original_material
    assert result._fit_attempt_parameters is original_attempt
    loaded = Result.load(result.output)
    assert regularizer.check_status(loaded) == ("finished", 2, 2)
    np.testing.assert_array_equal(loaded.solvers[-1].mesh.forces, np.full((4, 3), 2.))


def test_process_without_pending_frames_does_not_record_unapplied_settings():
    from saenopy.gui.solver.modules.Regularizer import Regularizer

    result = series_result(2)
    for solver in result.solvers:
        solver.regularisation_results = np.ones((2, 3))
    previous = dict(solve_parameters=dict(result.solve_parameters))
    result._fit_attempt_parameters = previous
    Regularizer.process(process_owner(), result, result.material_parameters,
                        dict(result.solve_parameters, alpha=2e8))
    assert result._fit_attempt_parameters is previous


def test_actual_channel_edit_after_dataset_switch_updates_fit_settings(gui_window):
    from saenopy import Result
    regularizer = gui_window.sub_module_regularize
    first, second = Result(), Result()
    for result in (first, second):
        result.stacks = [SimpleNamespace(channels=["matrix", "cell"])]
        result.solve_parameters = {"seg_channel": 0}
    regularizer.setResult(first)
    choice = regularizer.input_seg_channel
    choice.combobox.setCurrentIndex(1)
    assert first.solve_parameters_tmp["seg_channel"] == 1
    regularizer.setResult(second)
    assert choice.value() == 0
    choice.combobox.setCurrentIndex(1)
    assert choice.value() == second.solve_parameters_tmp["seg_channel"] == 1
    assert type(second.solve_parameters_tmp["seg_channel"]) is int
    # Import actions also emit an explicit programmatic selection. Returning to
    # the last manually selected channel must still update the next fit.
    choice.setValues([0, 1, "local:0"], ["matrix", "cell", "local cell"])
    choice.setValue("local:0", send_signal=True)
    choice.combobox.setCurrentIndex(1)
    assert second.solve_parameters_tmp["seg_channel"] == 1


def test_preview_mask_is_transient_used_for_display_and_not_saved(gui_window, monkeypatch, tmp_path):
    import importlib
    from unittest.mock import patch
    from saenopy import Result, surface_regularization as sr
    regularizer_module = importlib.import_module("saenopy.gui.solver.modules.Regularizer")
    renderer = importlib.import_module("saenopy.gui.solver.modules.showVectorField")
    regularizer = gui_window.sub_module_regularize
    result = example_result()
    solver = result.solvers[0]
    fitted_mask = np.array([True, False, False, False])
    preview_mask = np.array([False, True, False, False])
    solver.mesh.cell_boundary_mask = fitted_mask.copy()
    solver.mesh.forces[:] = 17
    solver.regularisation_results = np.ones((2, 3))
    solver.regularisation_parameters = completed_parameters()
    regularizer.setResult(result)
    body = np.zeros((8, 8, 8), bool)
    body[2:6, 2:6, 2:6] = True
    monkeypatch.setattr(regularizer_module, "segment_with_params", lambda *args: (
        SimpleNamespace(voxel_size=(1., 1., 1.)), body, body, np.zeros((1, 3)), 42.))
    monkeypatch.setattr(sr, "surface_node_mask", lambda *args, **kwargs: preview_mask.copy())
    monkeypatch.setattr(gui_window.tab5, "update_display", lambda: None)
    regularizer.preview_segmentation()
    assert "segmentation failed" not in regularizer.input_button_preview_text.text()
    np.testing.assert_array_equal(solver.mesh.cell_boundary_mask, fitted_mask)
    np.testing.assert_array_equal(solver.mesh._segmentation_preview[2], preview_mask)
    assert gui_window.tab5.checkTabEnabled(result)

    class DisplayMesh:
        def __init__(self, points, *args):
            self.points = np.asarray(points)

        def glyph(self, **kwargs):
            return self

    actors = {}
    plotter = SimpleNamespace(remove_actor=lambda *args: None,
        add_mesh=lambda mesh, **kwargs: actors.update({kwargs["name"]: mesh}))
    with patch.object(renderer.pv, "PolyData", DisplayMesh):
        renderer._draw_surface_nodes(plotter, solver.mesh, True)
    np.testing.assert_array_equal(actors["surface_nodes"].points, solver.mesh.nodes[preview_mask] * 1e6)
    result.save(tmp_path / "preview.saenopy")
    loaded = Result.load(tmp_path / "preview.saenopy")
    np.testing.assert_array_equal(loaded.solvers[0].mesh.cell_boundary_mask, fitted_mask)
    np.testing.assert_array_equal(loaded.solvers[0].mesh.forces, np.full((4, 3), 17.))
    assert not hasattr(loaded.solvers[0].mesh, "_segmentation_preview")


def test_missing_live_frame_waits_without_reading_worker_solver(gui_window, monkeypatch):
    from saenopy.gui.solver.modules.live_fit import displayed_solver, solver_snapshot
    result = series_result(2)
    result.solvers[0].regularisation_results = np.ones((2, 3))
    result.solve_parameters_state = "running"
    result._live_fit_active = True
    result._live_fit_solvers = {0: solver_snapshot(result.solvers[0])}
    for viewer in (gui_window.tab4, gui_window.tab5):
        viewer.result = result
        viewer.current_tab_selected = True
        viewer.t_slider.setRange(0, 1)
        cleared = []
        monkeypatch.setattr(viewer.plotter, "clear", lambda: cleared.append(True))
        viewer.t_slider.setValue(1)
        assert displayed_solver(result, 1) is None
        assert not viewer.checkTabEnabled(result)
        assert "waiting for this frame" in viewer.label_tab.text()
        assert cleared
    snapshot = solver_snapshot(result.solvers[1])
    result._live_fit_solvers[1] = snapshot
    for viewer in (gui_window.tab4, gui_window.tab5):
        assert viewer.checkTabEnabled(result)
    assert displayed_solver(result, 1) is snapshot


@pytest.mark.parametrize("terminal_state", ["finished", ""])
def test_unselected_finished_or_cancelled_result_releases_snapshots(gui_window, terminal_state):
    from saenopy import Result
    regularizer = gui_window.sub_module_regularize
    regularizer.setResult(Result())
    other = Result()
    other.solve_parameters_state = "running"
    other._live_fit_active = True
    other._live_fit_solvers = {0: object()}
    gui_window.result_changed.emit(other)
    assert other._live_fit_solvers
    other.solve_parameters_state = terminal_state
    gui_window.result_changed.emit(other)
    assert not other._live_fit_active
    assert other._live_fit_solvers == {}


def test_force_time_slider_refreshes_stop_reason(gui_window):
    regularizer = gui_window.sub_module_regularize
    result = series_result(2)
    result.solvers[0].regularisation_parameters = {"stop_reason": "max_iterations"}
    result.solvers[1].regularisation_parameters = {"stop_reason": "objective_plateau"}
    regularizer.setResult(result)
    assert "iteration budget" in regularizer.fit_status_text.text()
    gui_window.t_slider.setRange(0, 1)
    gui_window.t_slider.setValue(1)
    assert "Objective plateau" in regularizer.fit_status_text.text()
    gui_window.t_slider.setValue(0)
    assert "iteration budget" in regularizer.fit_status_text.text()
