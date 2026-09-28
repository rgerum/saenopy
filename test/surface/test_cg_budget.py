"""Bounded inner accuracy, cancellation and backwards-compatible metadata."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
from saenopy.conjugate_gradient import cg
from saenopy import Solver, Result
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
def test_outer_plateau_cannot_hide_inaccurate_inner_steps(monkeypatch, converged):
    result = example_result()
    def fake(self, *args):
        self.last_cg_info = dict(iterations=50, relative_residual=0. if converged else .1,
                                 converged=converged)
        self._cg_diagnostics.append(self.last_cg_info)
        return 0.
    monkeypatch.setattr(Solver, "_solve_regularization_cg", fake)
    fit_result(result, parameters=dict(max_iterations=20, rel_conv_crit=.01))
    count = len(result.solvers[0].regularisation_results) - 1
    assert count < 20 if converged else count == 20


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


@pytest.mark.parametrize("value", [0, -1, 1.5, np.nan, np.inf])
def test_invalid_budget(value):
    result = example_result()
    with pytest.raises(ValueError, match="cg_maxiter_factor"):
        fit_result(result, parameters=dict(cg_maxiter_factor=value))
