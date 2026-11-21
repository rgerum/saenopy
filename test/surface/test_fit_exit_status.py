"""Last accepted fields and old-result compatibility after control removal."""
from types import SimpleNamespace
import numpy as np
from saenopy import Solver, Result
from saenopy.reconstruction import fit_result
from test_release import example_result
from test_cg_budget import _replay_terms


def test_old_experimental_settings_do_not_stop_a_refit(monkeypatch):
    result = example_result()
    result.solve_parameters.update(stop_on_stagnation=True, max_time_seconds=1e-12)
    fit_result(result)
    assert len(result.solvers[0].regularisation_results) == 3
    assert result.solvers[0].regularisation_parameters['stop_reason'] == 'max_iterations'
    assert 'stop_on_stagnation' not in result.solve_parameters
    assert 'max_time_seconds' not in result.solve_parameters


def test_export_omits_retired_settings_without_modifying_result():
    from saenopy.gui.solver.modules.Regularizer import Regularizer
    old = dict(stop_on_stagnation=True, max_time_seconds=2.)
    result = SimpleNamespace(solve_parameters_tmp=old.copy(), material_parameters_tmp=dict(k=1000))
    imports, code = Regularizer.get_code(SimpleNamespace(result=result))
    compile(imports + code, '<export>', 'exec')
    assert 'stop_on_stagnation' not in code and 'max_time_seconds' not in code
    assert result.solve_parameters_tmp == old


def test_exit_reason_distinguishes_budget_from_plateau(monkeypatch):
    s = _replay_terms(monkeypatch, np.ones(31), np.geomspace(1., 100., 31))
    assert s.regularisation_parameters['stop_reason'] == 'max_iterations'
    s = _replay_terms(monkeypatch, np.ones(31))
    assert s.regularisation_parameters['stop_reason'] == 'objective_plateau'


def test_cancelled_inner_step_retains_initial_field_and_saved_status(monkeypatch, tmp_path):
    import saenopy.solver as module
    result = example_result()
    initial = result.solvers[0].mesh.displacements.copy()
    signal = SimpleNamespace(cancel=False)
    def interrupted(A, b, maxiter, tol, **kwargs):
        kwargs['cancel_signal'].cancel = True
        return np.ones_like(b), dict(iterations=1, maxiter=maxiter, relative_residual=1.,
            relative_tolerance=np.sqrt(tol), converged=False, reason='cancelled')
    monkeypatch.setattr(module, 'cg', interrupted)
    fit_result(result, cancel_signal=signal)
    s = result.solvers[0]
    np.testing.assert_array_equal(s.mesh.displacements, initial)
    assert len(s.regularisation_results) == 1
    assert s.regularisation_parameters['stop_reason'] == 'cancelled'
    assert s.regularisation_parameters['cancelled'] is True
    assert not s.regularisation_parameters['objective_plateau_reached']
    p = tmp_path / 'cancelled.saenopy'
    result.save(p)
    loaded = Result.load(p)
    assert loaded.solvers[0].regularisation_parameters['stop_reason'] == 'cancelled'
    assert bool(loaded.solvers[0].regularisation_parameters['cancelled'])
    np.testing.assert_array_equal(loaded.solvers[0].mesh.displacements, initial)
