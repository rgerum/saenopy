"""Native-equation equivalence and lifecycle checks for normalized matrix-free CG."""
from types import MethodType

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from saenopy import Solver
from saenopy import physical_regularization as pr
from saenopy.conjugate_gradient import cg
from saenopy.materials import SemiAffineFiberMaterial


def _solver(mode, missing_target=False):
    solver = Solver()
    solver.set_nodes(np.array([[0, 0, 0], [10e-6, 0, 0],
                               [0, 10e-6, 0], [0, 0, 10e-6]], dtype=float))
    solver.set_tetrahedra(np.array([[0, 1, 2, 3]], dtype=int))
    solver.set_target_displacements(-0.01 * solver.mesh.nodes, np.ones(4, bool))
    if missing_target:
        solver.mesh.displacements_target_mask[1] = False
        solver.mesh.displacements_target[1] = np.nan
        solver.mesh.movable[3] = False
    kwargs = (dict(surface_mask=np.array([True, True, False, False]),
                   surface_area_m2=2e-10) if mode == "surface" else {})
    info = pr.configure_solver(solver, 1e4, 14, **kwargs)
    solver.set_material_model(SemiAffineFiberMaterial(1000, None, None, None))
    return solver, info


def _explicit_system(solver, alpha):
    """The previous sparse assembly, retained here as an independent reference."""
    coefficient = np.repeat(solver.localweight * alpha, 3)
    KA = solver.K_glo.multiply(coefficient[None, :])
    target = solver.mesh.displacements_target_mask & solver.mesh.movable
    q = solver.physical_data_weights * target
    A = sparse.diags(np.repeat(q, 3), format="csr") + KA @ solver.K_glo
    b = (KA @ solver.mesh.forces.ravel()).reshape(solver.mesh.forces.shape)
    residual = solver.mesh.displacements_target - solver.mesh.displacements
    residual[~target] = 0
    b += q[:, None] * residual
    return A, b


@pytest.mark.parametrize("mode", ["classic", "surface"])
@pytest.mark.parametrize("nonsymmetric", [False, True])
def test_native_operator_and_rhs_with_missing_and_fixed_targets(mode, nonsymmetric):
    solver, _ = _solver(mode, missing_target=True)
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(12, 12))
    matrix = matrix.T @ matrix + np.eye(12)
    if nonsymmetric:
        matrix[0, 2] += 1  # Detect an unintended transpose of native K.
    solver.K_glo = sparse.csr_matrix(matrix)
    solver.localweight = np.ones(4)
    solver.mesh.forces[:] = rng.normal(size=(4, 3)) * 1e-9
    solver._update_local_regularization_weigth("huber")
    expected_A, expected_b = _explicit_system(solver, 3.0)
    forces = solver.mesh.forces.copy()
    K = solver.K_glo
    solver._compute_regularization_a_and_b(3.0)

    assert isinstance(solver.A, LinearOperator)
    assert solver.KAK is None
    assert solver.K_glo is K
    np.testing.assert_array_equal(solver.b, expected_b)
    np.testing.assert_array_equal(solver.mesh.forces, forces)
    assert np.isfinite(solver.b).all()
    for _ in range(4):
        vector = rng.normal(size=12)
        np.testing.assert_allclose(solver.A @ vector, expected_A @ vector,
                                   rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(solver.A @ vector[:, None],
                                   (expected_A @ vector)[:, None], rtol=1e-12, atol=1e-12)


def test_operator_keeps_its_state_across_inplace_changes_and_new_assembly():
    solver, _ = _solver("classic")
    solver.K_glo = sparse.diags(np.arange(1., 13.), format="csr")
    solver.localweight = np.arange(1., 5.)
    solver._compute_regularization_a_and_b(2.)
    old = solver.A
    vector = np.arange(12, dtype=float)
    expected = old @ vector
    solver.K_glo.data *= 3
    solver.localweight *= .5
    solver.physical_data_weights[:] *= 2
    solver.mesh.displacements_target_mask[0] = False
    solver.I.data[:] = 0
    solver._compute_regularization_a_and_b(5.)
    np.testing.assert_array_equal(old @ vector, expected)
    expected_new, _ = _explicit_system(solver, 5.)
    np.testing.assert_allclose(solver.A @ vector, expected_new @ vector, rtol=1e-14)
    assert not np.allclose(solver.A @ vector, expected)


def test_cg_matches_dense_reference_for_a_well_conditioned_native_system():
    solver, _ = _solver("classic")
    rng = np.random.default_rng(8)
    matrix = rng.normal(size=(12, 12)) * .03 + np.eye(12)
    solver.K_glo = sparse.csr_matrix(matrix.T @ matrix)
    solver.localweight = np.linspace(.8, 1.2, 4)
    solver.mesh.forces[:] = rng.normal(size=(4, 3)) * 1e-8
    expected_A, expected_b = _explicit_system(solver, .5)
    solver._compute_regularization_a_and_b(.5)
    solution, info = cg(solver.A, solver.b.ravel(), maxiter=100,
                        tol=1e-24, return_info=True)
    assert info["converged"]
    np.testing.assert_allclose(solution, np.linalg.solve(expected_A.toarray(), expected_b.ravel()),
                               rtol=1e-10, atol=1e-18)


@pytest.mark.parametrize("mode", ["classic", "surface"])
def test_small_fem_fit_matches_explicit_assembly_and_keeps_defaults(mode):
    matrix_free, info = _solver(mode)
    explicit, _ = _solver(mode)

    def assemble(self, alpha):
        self.A, self.b = _explicit_system(self, alpha)

    explicit._compute_regularization_a_and_b = MethodType(assemble, explicit)
    for solver in (matrix_free, explicit):
        solver.solve_regularized(alpha=info["internal_alpha"], max_iterations=5,
                                  rel_conv_crit=0.0)
        assert solver.regularisation_parameters["cg_maxiter_factor"] == 2
        assert solver.regularisation_parameters["step_size"] == .2
    np.testing.assert_allclose(matrix_free.mesh.displacements, explicit.mesh.displacements,
                               rtol=1e-7, atol=1e-15)
    np.testing.assert_allclose(matrix_free.mesh.forces, explicit.mesh.forces,
                               rtol=1e-7, atol=1e-15)
    np.testing.assert_allclose(matrix_free.regularisation_results,
                               explicit.regularisation_results, rtol=1e-7, atol=1e-25)


def test_reconfigure_and_reset_restore_legacy_matrix_assembly():
    solver, _ = _solver("surface")
    solver.K_glo = sparse.eye(12, format="csr")
    solver.localweight = np.ones(4)
    solver._compute_regularization_a_and_b(1.)
    assert isinstance(solver.A, LinearOperator)
    pr.configure_solver(solver, 1e4, 14)
    solver._compute_regularization_a_and_b(1.)
    assert isinstance(solver.A, LinearOperator)
    pr.reset_solver(solver)
    assert "_compute_regularization_a_and_b" not in solver.__dict__
    solver._compute_regularization_a_and_b(1.)
    assert sparse.issparse(solver.A)
    assert sparse.issparse(solver.KAK)
