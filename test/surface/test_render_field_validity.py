"""Pending force fields and missing measurements must remain distinguishable."""
import importlib

import numpy as np
import pytest

from test_release import example_result
from test_review_gui_regressions import completed_parameters, gui_window, series_result

pytestmark = pytest.mark.usefixtures('isolated_qt_settings')


@pytest.mark.parametrize('nan_row', [False, True])
def test_zero_field_preserves_nan_markers_without_arrows(gui_window, nan_row):
    result = example_result()
    solver = result.solvers[0]
    solver.regularisation_results = np.ones((2, 3))
    solver.mesh.displacements[:] = 0
    # A partial NaN also represents an invalid vector, regardless of component.
    solver.mesh.displacements[0, 1] = np.nan
    if nan_row:
        solver.mesh.displacements[0] = np.nan
    viewer = gui_window.tab4
    viewer.result = result
    viewer.current_tab_selected = True
    viewer.vtk_toolbar.use_nans.setValue(True)
    with np.errstate(divide='raise', invalid='raise'):
        viewer.update_display()
    assert 'nans' in viewer.plotter.actors
    assert viewer.plotter.actors['nans'].GetMapper().GetInput().GetNumberOfPoints() == 1
    assert 'arrows' not in viewer.plotter.actors
    assert viewer.plotter.auto_value == 0

    viewer.vtk_toolbar.use_nans.setValue(False)
    viewer.update_display()
    assert 'nans' not in viewer.plotter.actors
    viewer.vtk_toolbar.use_nans.setValue(True)
    solver.mesh.displacements[:] = 0
    viewer.update_display()
    assert 'nans' not in viewer.plotter.actors
    assert 'arrows' not in viewer.plotter.actors


def test_refit_pending_frames_hide_retained_forces(gui_window, monkeypatch):
    import saenopy.reconstruction as reconstruction
    from saenopy import Solver

    module = importlib.import_module('saenopy.gui.solver.modules.TabForces')
    result = series_result(2)
    for solver in result.solvers:
        solver.regularisation_results = np.ones((2, 3))
        solver.regularisation_parameters = completed_parameters()
        solver.mesh.cell_boundary_mask = np.array([True, False, False, False])
        solver.mesh.forces[:] = 17.
    pending_displacements = result.solvers[1].mesh.displacements.copy()
    regularizer = gui_window.sub_module_regularize
    regularizer.setResult(result)
    regularizer.input_alpha.setValue(2e8, send_signal=True)
    queued, fields, centers = [], [], []
    monkeypatch.setattr(gui_window, 'addTask', lambda *args: queued.append(args))
    monkeypatch.setattr(module, 'getVectorFieldImage', lambda self: None)
    monkeypatch.setattr(module, 'showVectorField', lambda *args, **kwargs: fields.append(args[2]))
    monkeypatch.setattr(Solver, 'get_center', lambda *args, **kwargs: centers.append(True) or np.zeros(3))

    def fit(result, index, cancel_signal, **kwargs):
        assert index == 0
        solver = result.solvers[index]
        solver.regularisation_results = np.ones((2, 3))
        solver.regularisation_parameters = dict(completed_parameters(), cancelled=True)
        cancel_signal.cancel = True
        return solver

    monkeypatch.setattr(reconstruction, 'fit_result', fit)
    regularizer.start_process()
    viewer = gui_window.tab5
    viewer.result = result
    viewer.current_tab_selected = True
    viewer.vtk_toolbar.use_center.setValue(1)
    viewer.t_slider.setRange(0, 1)
    viewer.t_slider.setValue(1)
    fields.clear()
    centers.clear()
    viewer.update_display()
    assert fields == [None]  # Initial retry snapshot still owns the old forces.
    assert not centers
    assert result.solvers[1].regularisation_results is not None  # Not modified while queued.

    queued[0][0](result, queued[0][2])
    assert not result._live_fit_active
    fields.clear()
    centers.clear()
    viewer.update_display()
    assert fields == [None]
    assert not centers
    assert result.solvers[1].regularisation_results is None
    np.testing.assert_array_equal(result.solvers[1].mesh.forces, np.full((4, 3), 17.))
    np.testing.assert_array_equal(result.solvers[1].mesh.displacements, pending_displacements)
    viewer.t_slider.setValue(0)
    fields.clear()
    centers.clear()
    viewer.update_display()
    assert fields[-1] is not None  # The current cancelled field is still valid.
    assert centers


@pytest.mark.parametrize('state', ['completed', 'cancelled', 'live', 'failed'])
def test_forces_require_completed_or_current_live_evidence(gui_window, monkeypatch, state):
    from saenopy.gui.solver.modules.live_fit import solver_snapshot

    module = importlib.import_module('saenopy.gui.solver.modules.TabForces')
    result = example_result()
    solver = result.solvers[0]
    solver.mesh.forces[:] = 17.
    solver.mesh.cell_boundary_mask = np.array([True, False, False, False])
    solver.regularisation_parameters = {'cancelled': state == 'cancelled'}
    if state in ('completed', 'cancelled'):
        solver.regularisation_results = np.ones((2, 3))
    if state == 'live':
        result._live_fit_active = True
        result._live_fit_solvers = {0: solver_snapshot(solver)}
    viewer = gui_window.tab5
    viewer.result = result
    viewer.current_tab_selected = True
    fields = []
    monkeypatch.setattr(module, 'getVectorFieldImage', lambda self: None)
    monkeypatch.setattr(module, 'showVectorField', lambda *args, **kwargs: fields.append(args[2]))
    viewer.update_display()
    assert (fields[-1] is not None) == (state != 'failed')
