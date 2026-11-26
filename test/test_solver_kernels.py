import numpy as np
import scipy.sparse as ssp

from saenopy import Solver
from saenopy.materials import SemiAffineFiberMaterial
from saenopy.multigrid_helper import create_box_mesh


def reference_f_and_K(M):
    """ The original einsum formulation of the force and stiffness computation """
    T = M.mesh.tetrahedra
    F = np.eye(3) + np.einsum("tmi,tmj->tij", M.mesh.displacements[T], M.mesh.Phi)
    s_bar = F @ M.s.T
    s_star = M.mesh.Phi @ M.s.T
    s = np.linalg.norm(s_bar, axis=1)
    eps, epsbar, epsbarbar = M.material_model_look_up(s - 1)
    V = M.mesh.volume[:, None] / M.N_b
    dEdsbar = -(epsbar / s) * V
    dEdsbarbar = ((s * epsbarbar - epsbar) / s ** 3) * V
    f = np.einsum("tmb,tib,tb->tmi", s_star, s_bar, dEdsbar)
    s_bar_s_bar = 0.5 * (np.einsum("tb,tib,tlb->tilb", dEdsbarbar, s_bar, s_bar)
                         - np.einsum("il,tb->tilb", np.eye(3), dEdsbar))
    K = np.einsum("tmb,trb,tilb->tmril", s_star, s_star, s_bar_s_bar)

    n = M.mesh.number_nodes
    forces = np.zeros((n, 3))
    np.add.at(forces, T, f)
    # K[t, m, r, i, l] belongs to row 3 * T[t, m] + i and column 3 * T[t, r] + l
    rows = np.broadcast_to(T[:, :, None, None, None] * 3 + np.arange(3)[:, None], K.shape)
    cols = np.broadcast_to(T[:, None, :, None, None] * 3 + np.arange(3), K.shape)
    keep = np.broadcast_to(M.mesh.movable[T][:, :, None, None, None], K.shape)
    K_glo = ssp.coo_matrix((K[keep], (rows[keep], cols[keep])), shape=(n * 3, n * 3)).tocsr()
    energy = np.mean(eps, axis=1) * M.mesh.volume
    return forces, K_glo, np.sum(energy[np.any(M.mesh.movable[T], axis=1)])


def make_solver(fix_border):
    R, T = create_box_mesh(np.linspace(-0.5, 0.5, 6))
    M = Solver()
    M.set_nodes(R)
    M.set_tetrahedra(T)
    M.set_material_model(SemiAffineFiberMaterial(900, 0.0004, 0.0075, 0.033))
    if fix_border:
        border = np.any(np.abs(R) == 0.5, axis=1)
        displacement = np.full(R.shape, np.nan)
        displacement[border] = 0
        force = np.zeros(R.shape)
        force[border] = np.nan
        M.set_boundary_condition(displacement, force)
    M.mesh.displacements[:] = np.random.default_rng(0).normal(scale=0.02, size=R.shape)
    M._check_relax_ready()
    M._prepare_temporary_quantities()
    M._update_glo_f_and_k()
    return M


def test_f_and_K_match_reference():
    for fix_border in [False, True]:
        M = make_solver(fix_border)
        forces, K_glo, energy = reference_f_and_K(M)
        np.testing.assert_allclose(M.mesh.forces, forces, rtol=0, atol=1e-12 * np.abs(forces).max())
        assert abs(M.mesh.strain_energy - energy) <= 1e-12 * abs(energy)
        assert abs(M.K_glo - K_glo).max() <= 1e-12 * abs(K_glo).max()


def test_regularization_operator_matches_explicit_matrix():
    M = make_solver(False)
    n = M.mesh.number_nodes
    M.mesh.displacements_target_mask = np.random.default_rng(1).random(n) > 0.3
    M._I_diag = np.repeat(M.mesh.displacements_target_mask, 3).astype(float)
    M.localweight = np.random.default_rng(2).uniform(0.1, 1, n)
    M.mesh.displacements_target = np.zeros((n, 3))
    M._compute_regularization_a_and_b(1e3)

    K = M.K_glo.tocsr()
    W = ssp.diags(np.repeat(M.localweight * 1e3, 3))
    A = ssp.diags(M._I_diag) + K @ W @ K
    x = np.random.default_rng(3).normal(size=n * 3)
    y = A @ x
    np.testing.assert_allclose(M.A @ x, y, rtol=0, atol=1e-12 * np.abs(y).max())
    np.testing.assert_allclose(K @ x, M.K_glo @ x, rtol=0, atol=1e-12 * np.abs(K @ x).max())
