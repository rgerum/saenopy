"""Queued previews must belong to the mesh currently used for reconstruction."""
import numpy as np
import pytest

from test_release import example_result
from test_review_gui_regressions import completed_parameters, gui_window, series_result

pytestmark = pytest.mark.usefixtures('isolated_qt_settings')


@pytest.mark.parametrize('replacement', ['solver', 'empty'])
def test_live_snapshot_rejected_as_soon_as_source_is_replaced(replacement):
    from saenopy.gui.solver.modules.live_fit import displayed_solver, live_fit_available, solver_snapshot

    result = example_result()
    original = result.solvers[0]
    snapshot = solver_snapshot(original)
    result._live_fit_active = True
    result._live_fit_solvers = {0: snapshot}
    assert displayed_solver(result, 0) is snapshot
    result.solvers[0] = example_result().solvers[0] if replacement == 'solver' else None
    # Equal coordinates on a newly interpolated solver still mean a new fit.
    assert displayed_solver(result, 0) is None
    assert not live_fit_available(result, 0)


@pytest.mark.parametrize('tab', ['tab4', 'tab5'])
def test_mesh_replacement_between_availability_and_render_clears_view(gui_window, monkeypatch, tab):
    from saenopy.gui.solver.modules.live_fit import solver_snapshot

    result = example_result()
    result._live_fit_active = True
    result._live_fit_solvers = {0: solver_snapshot(result.solvers[0])}
    viewer = getattr(gui_window, tab)
    viewer.result = result
    viewer.current_tab_selected = True
    cleared = []

    def available(*args):
        # Simulate the worker replacing a mesh after the availability check.
        result.solvers[0] = example_result().solvers[0]
        return True

    monkeypatch.setattr(viewer, 'frame_available', available)
    monkeypatch.setattr(viewer.plotter, 'clear', lambda: cleared.append(True))
    viewer.update_display()
    assert cleared


def test_queued_mesh_replacement_waits_until_current_fit_snapshot(gui_window, monkeypatch):
    import saenopy
    import saenopy.reconstruction as reconstruction
    from saenopy.get_deformations import PivMesh
    from saenopy.gui.solver.modules.live_fit import displayed_solver, live_fit_available

    result = series_result(2)
    for solver in result.solvers:
        solver.regularisation_results = np.ones((2, 3))
        solver.regularisation_parameters = completed_parameters()
    result.mesh_piv = [PivMesh(nodes=solver.mesh.nodes, tetrahedra=solver.mesh.tetrahedra,
                              displacements_measured=solver.mesh.displacements_target)
                       for solver in result.solvers]
    mesh_creator = gui_window.sub_module_mesh
    regularizer = gui_window.sub_module_regularize
    mesh_creator.setResult(result)
    regularizer.setResult(result)
    queued = []
    monkeypatch.setattr(gui_window, 'addTask', lambda *args: queued.append(args))
    mesh_creator.start_process(result=result)
    regularizer.start_process(result=result)
    old_snapshots = dict(result._live_fit_solvers)
    assert len(queued) == 2
    assert all(displayed_solver(result, frame) is old_snapshots[frame] for frame in range(2))

    generated, between_meshes = [], []
    monkeypatch.setattr(saenopy, 'subtract_reference_state', lambda *args: [None, None])

    def interpolate(*args, **kwargs):
        if generated:
            # The first mesh has been replaced, before mesh creation or the fit finishes.
            between_meshes.append(displayed_solver(result, 0))
        solver = example_result().solvers[0]
        solver.mesh.nodes *= 2
        generated.append(solver)
        return solver

    monkeypatch.setattr(saenopy, 'interpolate_mesh', interpolate)
    queued[0][0](result, queued[0][2])
    assert between_meshes == [None]
    assert result.solve_parameters_state == 'scheduled'
    for frame in range(2):
        assert displayed_solver(result, frame) is None
        assert not live_fit_available(result, frame)
        for viewer in (gui_window.tab4, gui_window.tab5):
            assert not viewer.frame_available(result, frame)

    publications = []
    regularizer.live_field_ready.connect(
        lambda current, frame, snapshot: publications.append(displayed_solver(current, frame) is snapshot))

    def fit(current, index, callback, cancel_signal, **kwargs):
        assert index == 0
        assert displayed_solver(current, index) is None
        solver = current.solvers[index]
        solver.regularisation_parameters = completed_parameters()
        callback(solver, [], 0, 2)
        assert displayed_solver(current, index) is not None
        current_snapshot = displayed_solver(current, index)
        # A delayed event from the previous mesh must not replace current data.
        regularizer.show_live_field(current, index, old_snapshots[index])
        assert displayed_solver(current, index) is current_snapshot
        solver.regularisation_results = np.ones((2, 3))
        solver.regularisation_parameters['cancelled'] = True
        cancel_signal.cancel = True
        return solver

    monkeypatch.setattr(reconstruction, 'fit_result', fit)
    queued[1][0](result, queued[1][2])
    assert publications == [True, True]  # Includes the final snapshot after clear_cache.
    assert not result._live_fit_active
    assert result.solvers[0].regularisation_parameters['cancelled']
    assert result.solvers[1].regularisation_results is None
    np.testing.assert_array_equal(result.solvers[1].mesh.nodes, generated[1].mesh.nodes)
    assert gui_window.tab5.frame_available(result, 0)
    assert not gui_window.tab5.frame_available(result, 1)
