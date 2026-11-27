"""Viewer status follows the selected result, including hidden tabs and failures."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from types import SimpleNamespace
import pytest
from test_review_gui_regressions import gui_window

pytestmark = pytest.mark.usefixtures("isolated_qt_settings")


def test_display_snapshot_accepts_partial_metadata_without_mutating_worker():
    from test_release import example_result
    from saenopy.gui.solver.modules.live_fit import solver_snapshot

    worker = example_result().solvers[0]
    worker.regularisation_parameters = {"cancelled": True}
    snapshot = solver_snapshot(worker)
    assert snapshot.regularisation_parameters == {"cancelled": True}
    snapshot.regularisation_parameters["cancelled"] = False
    assert worker.regularisation_parameters == {"cancelled": True}


def test_display_snapshot_releases_worker_caches_and_preserves_preview():
    import gc
    import weakref
    import numpy as np
    from test_release import example_result
    from saenopy.reconstruction import fit_result
    from saenopy.gui.solver.modules.live_fit import solver_snapshot

    result = example_result()
    fit_result(result)
    worker = result.solvers[0]
    worker.mesh.cell_boundary_mask = np.array([True, False, False, False])
    preview = (worker.mesh.nodes.copy() * 1e6, np.array([[0, 1, 2]]),
               worker.mesh.cell_boundary_mask.copy())
    worker.mesh._segmentation_preview = preview
    snapshot = solver_snapshot(worker)
    for name in ('K_glo', 'A', 'KAK', '_compute_regularization_a_and_b',
                 '_update_local_regularization_weigth', '_record_regularization_status'):
        assert name not in snapshot.__dict__
    assert snapshot.mesh.Phi is None
    np.testing.assert_array_equal(snapshot.mesh.nodes, worker.mesh.nodes)
    np.testing.assert_array_equal(snapshot.mesh.tetrahedra, worker.mesh.tetrahedra)
    for name in ('forces', 'forces_border', 'displacements', 'displacements_target',
                 'displacements_target_mask', 'regularisation_mask', 'cell_boundary_mask', 'movable'):
        np.testing.assert_array_equal(getattr(snapshot.mesh, name), getattr(worker.mesh, name))
        assert not np.shares_memory(getattr(snapshot.mesh, name), getattr(worker.mesh, name))
    for frozen, original in zip(snapshot.mesh._segmentation_preview, preview):
        np.testing.assert_array_equal(frozen, original)
        assert not np.shares_memory(frozen, original)

    worker_reference = weakref.ref(worker)
    result.clear_cache(0)
    del worker
    gc.collect()
    assert worker_reference() is None  # Keeping the snapshot must not retain its worker.


def test_zero_displacement_field_removes_previous_nan_markers(gui_window):
    import numpy as np
    from test_release import example_result
    from saenopy.gui.solver.modules.showVectorField import showVectorField

    result = example_result()
    solver = result.solvers[0]
    solver.regularisation_results = np.ones((2, 3))
    viewer = gui_window.tab4
    viewer.result = result
    viewer.current_tab_selected = True
    field = np.ones_like(solver.mesh.nodes) * 1e-6
    field[0] = np.nan
    showVectorField(viewer.plotter, solver.mesh, field, 'displacements', show_grid=False)
    assert 'nans' in viewer.plotter.actors
    viewer.vtk_toolbar.use_nans.setValue(False)
    solver.mesh.displacements[:] = 0
    viewer.update_display()
    assert 'arrows' not in viewer.plotter.actors
    assert 'nans' not in viewer.plotter.actors


def test_partial_result_keeps_tabs_reachable_from_unfitted_frame(gui_window, monkeypatch):
    import numpy as np
    from test_review_gui_regressions import series_result

    result = series_result(2)
    result.solvers[0].regularisation_results = np.ones((2, 3))
    for viewer in (gui_window.tab4, gui_window.tab5):
        viewer.result = result
        viewer.current_tab_selected = True
        viewer.t_slider.setRange(0, 1)
        viewer.t_slider.setValue(1)
        cleared = []
        monkeypatch.setattr(viewer.plotter, 'clear', lambda: cleared.append(True))
        viewer.resultChanged(result)
        tab_index = gui_window.tabs.indexOf(viewer.tab.parent())
        assert gui_window.tabs.isTabEnabled(tab_index)
        assert not viewer.frame_available(result, 1)
        assert cleared
        viewer.t_slider.setValue(0)
        assert viewer.frame_available(result, 0)


@pytest.mark.parametrize("forces", [None, 0.])
def test_zero_force_preview_keeps_selected_image(gui_window, monkeypatch, forces):
    import importlib
    import numpy as np
    from test_release import example_result

    module = importlib.import_module("saenopy.gui.solver.modules.TabForces")
    result = example_result()
    solver = result.solvers[0]
    solver.mesh.forces = None if forces is None else np.zeros_like(solver.mesh.nodes)
    solver.mesh.cell_boundary_mask = np.array([True, False, False, False])
    viewer = gui_window.tab5
    viewer.result = result
    viewer.current_tab_selected = True
    monkeypatch.setattr(module, "getVectorFieldImage", lambda self: [
        np.full((8, 6, 1), .5), (1., 1., 1.), 0])
    viewer.vtk_toolbar.colormap_chooser2.setValue("viridis")

    viewer.update_display()
    assert "image_mesh" in viewer.plotter.actors
    texture = viewer.plotter.actors["image_mesh"].GetTexture()
    assert texture.GetInput().GetNumberOfScalarComponents() == 3
    assert "arrows" not in viewer.plotter.actors
    # The image toggle must still be able to remove the background.
    monkeypatch.setattr(module, "getVectorFieldImage", lambda self: None)
    viewer.update_display()
    assert "image_mesh" not in viewer.plotter.actors


@pytest.mark.parametrize("state,active,status", [
    ("", False, None), ("finished", False, None),
    ("scheduled", True, "queued..."), ("running", True, "in progress..."),
    ("cancelling", True, "cancelling..."), ("failed", False, "fit failed"),
    ("finished", True, "in progress..."), ("failed", True, "in progress..."),
])
def test_fit_status_text(state, active, status):
    from test_release import example_result
    from saenopy.gui.solver.modules.live_fit import fit_status_label, solver_snapshot
    solver = example_result().solvers[0]
    result = SimpleNamespace(solve_parameters_state=state, _live_fit_active=active,
                             solvers=[solver], _live_fit_solvers={0: solver_snapshot(solver)})
    expected = "Field." if status is None else f"Field. <b>({status})</b>"
    assert fit_status_label(result, "Field.") == expected
    assert fit_status_label(None, "Field.") == "Field."


def test_status_follows_gui_fit_lifecycle(tmp_path, monkeypatch):
    from qtpy import QtCore, QtWidgets
    from pyvistaqt import QtInteractor
    from saenopy import Result
    from saenopy.gui.common.PipelineModule import StateEnum
    from saenopy.gui.solver.modules.BatchEvaluate import BatchEvaluate

    monkeypatch.setattr(QtInteractor, "render", lambda self: None)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    class TestWindow(BatchEvaluate):
        settings_key = "saenopy_live_status_test"

    window = TestWindow()
    viewers = (window.tab4, window.tab5)
    regularizer = window.sub_module_regularize
    result, other = Result(), Result()
    try:
        regularizer.setResult(result)
        for viewer in viewers:
            viewer.current_tab_selected = False
            viewer.setResult(result)
            assert viewer.label_tab.text() == viewer.field_description

        result._live_fit_active = True
        result._live_fit_solvers = {}
        for state, text in [(StateEnum.scheduled, "queued..."),
                            (StateEnum.running, "in progress..."),
                            (StateEnum.cancelling, "cancelling...")]:
            regularizer.set_result_state(result, state)
            regularizer.processing_state_changed.emit(result)
            for viewer in viewers:
                assert text in viewer.label_tab.text()
                viewer.setResult(other)
                assert viewer.label_tab.text() == viewer.field_description
                viewer.resultChanged(result)
                assert viewer.label_tab.text() == viewer.field_description
                viewer.setResult(result)
                assert text in viewer.label_tab.text()

        # Cancellation returns to idle; a normal finish clears the notice;
        # failure reports failure instead of leaving an eternal progress label.
        for state, text in [(StateEnum.idle, None), (StateEnum.finished, None),
                            (StateEnum.failed, "fit failed")]:
            result._live_fit_active = True
            regularizer.set_result_state(result, state)
            if state in (StateEnum.idle, StateEnum.finished):
                window.result_changed.emit(result)
            else:
                regularizer.processing_state_changed.emit(result)
            assert not result._live_fit_active
            for viewer in viewers:
                if text:
                    assert text in viewer.label_tab.text()
                else:
                    assert viewer.label_tab.text() == viewer.field_description

        # A saved cancelled last frame stays resumable, and each viewer labels
        # the selected partial frame even while its tab is hidden.
        import copy
        import numpy as np
        from test_release import example_result
        partial = example_result()
        partial.solvers.append(copy.deepcopy(partial.solvers[0]))
        for solver in partial.solvers:
            solver.regularisation_results = np.ones((2, 3))
            solver.regularisation_parameters = {}
        partial.solvers[1].regularisation_parameters["cancelled"] = True
        regularizer.setResult(partial)
        assert regularizer.input_button.isEnabled()
        assert regularizer.input_button.text() == "continue"
        for viewer in viewers:
            viewer.setResult(partial)
            viewer.t_slider.setRange(0, 1)
            viewer.t_slider.setValue(1)
            assert "fit cancelled; partial result" in viewer.label_tab.text()
            viewer.t_slider.setValue(0)
            assert viewer.label_tab.text() == viewer.field_description
    finally:
        # Drain queued matplotlib draws before deleting their Qt canvases.
        for _ in range(3):
            app.processEvents()
        for plotter in window.findChildren(QtInteractor):
            plotter.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        app.processEvents()
