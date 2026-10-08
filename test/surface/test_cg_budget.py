"""Bounded inner accuracy, cancellation and backwards-compatible metadata."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
from saenopy.conjugate_gradient import cg
from saenopy import Solver, Result
from saenopy.materials import SemiAffineFiberMaterial
from saenopy.solver import DEFAULT_CG_MAXITER_FACTOR
from saenopy.reconstruction import fit_result
from test_release import example_result


def test_residual_and_iteration_limit_are_reported():
    A = np.diag(np.geomspace(1., 100., 20))
    b = np.ones(20)
    x, info = cg(A, b, maxiter=2, tol=1e-16, return_info=True)
    assert info["iterations"] == 2 and not info["converged"]
    assert info["reason"] == "iteration_limit"
    assert info["relative_residual"] == pytest.approx(np.linalg.norm(A @ x - b) / np.linalg.norm(b))
    x, info = cg(A, b, maxiter=80, tol=1e-16, return_info=True)
    assert info["converged"] and info["iterations"] < 80
    np.testing.assert_allclose(A @ x, b, atol=1e-7)


def test_zero_rhs_and_cancellation():
    x, info = cg(np.eye(3), np.zeros(3), return_info=True)
    assert x.shape == (3,) and info["iterations"] == 0 and info["converged"]
    x, info = cg(np.eye(3), np.ones(3), return_info=True, cancel_signal=SimpleNamespace(cancel=True))
    assert not x.any() and info["reason"] == "cancelled"
    assert not info["converged"]


def test_breakdown_is_not_silently_applied():
    with pytest.raises(FloatingPointError, match="breakdown"):
        cg(np.zeros((3, 3)), np.ones(3))


def test_solver_multiplier_and_warning_only_once(monkeypatch):
    import saenopy.solver as module
    solver = example_result().solvers[0]
    solver.mesh.number_nodes = len(solver.mesh.nodes)
    solver.A = np.eye(12)
    solver.b = np.ones((4, 3))
    solver.regularisation_parameters = dict(cg_maxiter_factor=4)
    solver._cg_diagnostics, solver._cg_warned = [], False
    caps = []
    def fake(A, b, maxiter, **kwargs):
        caps.append(maxiter)
        return np.ones_like(b), dict(iterations=maxiter, maxiter=maxiter, relative_residual=.1,
            relative_tolerance=1e-8, converged=False, reason="iteration_limit")
    monkeypatch.setattr(module, "cg", fake)
    with pytest.warns(RuntimeWarning, match="Inner CG") as messages:
        solver._solve_regularization_cg()
        solver._solve_regularization_cg()
    assert len(messages) == 1
    assert caps == [200, 200]
    assert len(solver._cg_diagnostics) == 2


@pytest.mark.parametrize("converged", [True, False])
def test_objective_plateau_keeps_inner_diagnostics(monkeypatch, converged):
    result = example_result()
    def fake(self, *args):
        self.last_cg_info = dict(iterations=50, relative_residual=0. if converged else .1,
                                 converged=converged)
        self._cg_diagnostics.append(self.last_cg_info)
        return 0.
    monkeypatch.setattr(Solver, "_solve_regularization_cg", fake)
    fit_result(result, parameters=dict(max_iterations=60, rel_conv_crit=.01))
    solver = result.solvers[0]
    assert len(solver.regularisation_results) - 1 == 24
    info = solver.regularisation_parameters
    assert info['objective_plateau_reached']
    assert info['cg_converged'] == [converged] * 24
    assert info['cg_unconverged_steps'] == (0 if converged else 24)


def _replay_terms(monkeypatch, data, force=None, **parameters):
    """Exercise the actual outer loop; substitute observations, not its stop rule."""
    solver = example_result().solvers[0]
    solver.set_material_model(SemiAffineFiberMaterial(k=1000))
    force = np.ones_like(data) if force is None else force
    def record(records, alpha, filename=None):
        i = len(records)
        records.append((data[i]+alpha*force[i], data[i], force[i]))
    monkeypatch.setattr(solver, '_record_regularization_status', record)
    monkeypatch.setattr(solver, '_solve_regularization_cg', lambda *args: 0.)
    solver.solve_regularized(max_iterations=len(data)-1, **parameters)
    return solver


@pytest.mark.parametrize('changing_term', ['data', 'force'])
def test_intermediate_plateau_resets_both_terms(monkeypatch, changing_term):
    data, force = np.ones(81), np.ones(81)
    (data if changing_term == 'data' else force)[23:] = 2.
    solver = _replay_terms(monkeypatch, data, force)
    assert len(solver.relrec)-1 == 46
    assert solver.regularisation_parameters['objective_plateau_reached']


def test_flat_data_cannot_hide_changing_force_penalty(monkeypatch):
    solver = _replay_terms(monkeypatch, np.ones(81), np.geomspace(1.,100.,81))
    assert len(solver.relrec)-1 == 80
    assert not solver.regularisation_parameters['objective_plateau_reached']


@pytest.mark.parametrize('bad_value', [np.nan, np.inf, -1.])
def test_invalid_term_resets_confirmation(monkeypatch, bad_value):
    force = np.ones(81); force[23] = bad_value
    with np.errstate(invalid='ignore'):
        solver = _replay_terms(monkeypatch, np.ones(81), force)
    assert len(solver.relrec)-1 == 47


@pytest.mark.parametrize('data,force', [(0.,0.),(1.,0.),(0.,1.)])
def test_identically_zero_term_is_stable(monkeypatch, data, force):
    solver = _replay_terms(monkeypatch, np.full(61,data), np.full(61,force))
    assert len(solver.relrec)-1 == 24


def test_unregularized_fit_only_checks_data(monkeypatch):
    solver = _replay_terms(monkeypatch, np.ones(61), np.geomspace(1.,100.,61), alpha=0)
    assert len(solver.relrec)-1 == 24


@pytest.mark.parametrize('limit', [1,19,20,23])
def test_short_fit_retains_hard_limit(monkeypatch, limit):
    solver = _replay_terms(monkeypatch, np.ones(limit+1))
    assert len(solver.relrec)-1 == limit
    assert not solver.regularisation_parameters['objective_plateau_reached']


@pytest.mark.parametrize('threshold', [0.,-1.])
def test_nonpositive_threshold_disables_stop(monkeypatch, threshold):
    solver = _replay_terms(monkeypatch, np.ones(61), rel_conv_crit=threshold)
    assert len(solver.relrec)-1 == 60


def test_minimum_iterations_and_cancellation(monkeypatch):
    solver = _replay_terms(monkeypatch, np.ones(81), i_min=50)
    assert len(solver.relrec)-1 == 56
    cancel = SimpleNamespace(cancel=False)
    def callback(solver, records, *args):
        if len(records)==24: cancel.cancel=True
    solver = _replay_terms(monkeypatch, np.ones(81), cancel_signal=cancel, callback=callback)
    assert len(solver.relrec)-1 == 23
    assert solver.regularisation_parameters['cancelled']
    assert not solver.regularisation_parameters['objective_plateau_reached']


def test_new_default_and_saved_override_reach_inner_solver(monkeypatch):
    import saenopy.solver as module
    caps=[]; tolerances=[]
    def fake(A,b,maxiter,tol,**kwargs):
        caps.append(maxiter); tolerances.append(tol)
        return np.zeros_like(b), dict(iterations=0,maxiter=maxiter,relative_residual=0.,
            relative_tolerance=np.sqrt(tol),converged=True,reason='converged')
    monkeypatch.setattr(module,'cg',fake)
    fresh=example_result(); fit_result(fresh)
    assert DEFAULT_CG_MAXITER_FACTOR == 16
    assert fresh.solve_parameters['cg_maxiter_factor'] == 16
    saved=example_result(); saved.solve_parameters['cg_maxiter_factor']=4; fit_result(saved)
    assert saved.solve_parameters['cg_maxiter_factor'] == 4
    assert caps == [800,800,200,200]
    assert tolerances == [4e-18]*4


@pytest.mark.parametrize('saved_factor,expected', [(None,16),(4,4)])
def test_gui_export_default_and_saved_override(saved_factor, expected):
    from saenopy.gui.solver.modules.Regularizer import Regularizer
    parameters = {} if saved_factor is None else dict(cg_maxiter_factor=saved_factor)
    result = SimpleNamespace(solve_parameters_tmp=parameters.copy(), material_parameters_tmp=dict(k=1000))
    imports, code = Regularizer.get_code(SimpleNamespace(result=result))
    compile(imports+code, '<exported reconstruction>', 'exec')
    assert f"'cg_maxiter_factor': {expected}" in code
    assert "'solver_precision': 1e-18" in code
    assert result.solve_parameters_tmp == parameters


def test_settings_and_diagnostics_roundtrip(tmp_path):
    result = example_result()
    fit_result(result, parameters=dict(cg_maxiter_factor=2, solver_precision=1e-16))
    info = result.solvers[0].regularisation_parameters
    assert info["cg_maxiter_factor"] == 2
    assert len(info["cg_iterations"]) == 2
    assert all(info["cg_converged"])
    json.dumps(info, allow_nan=False)
    filename = tmp_path / "cg_diagnostics.saenopy"
    result.save(filename)
    loaded = Result.load(filename)
    assert loaded.solve_parameters["cg_maxiter_factor"] == 2
    assert loaded.solvers[0].regularisation_parameters["cg_iterations"] == info["cg_iterations"]
    assert loaded.solvers[0].regularisation_parameters["cg_converged"] == info["cg_converged"]
    assert loaded.solvers[0].regularisation_parameters['convergence_window'] == 20
    assert loaded.solvers[0].regularisation_parameters['convergence_patience'] == 5
    assert not loaded.solvers[0].regularisation_parameters['objective_plateau_reached']


@pytest.mark.parametrize("value", [0, -1, 1.5, np.nan, np.inf])
def test_invalid_budget(value):
    result = example_result()
    with pytest.raises(ValueError, match="cg_maxiter_factor"):
        fit_result(result, parameters=dict(cg_maxiter_factor=value))
