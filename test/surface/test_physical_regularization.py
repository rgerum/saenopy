"""Unit tests for the geometry-only Classic/Surface alpha normalization."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from saenopy import physical_regularization as pr
from saenopy.solver import Solver


def _one_tetra_solver():
    solver = Solver()
    solver.set_nodes(np.array([
        [0.0, 0.0, 0.0],
        [10e-6, 0.0, 0.0],
        [0.0, 10e-6, 0.0],
        [0.0, 0.0, 10e-6],
    ]))
    solver.set_tetrahedra(np.array([[0, 1, 2, 3]], dtype=int))
    solver.mesh.regularisation_mask = np.ones(4, dtype=bool)
    return solver


def test_common_h5_mesh_factor():
    assert np.isclose(pr.mesh_scale_factor(7.0, 14.0), 32.0)
    assert np.isclose(
        pr.classic_internal_alpha(1e9, 7.0, 14.0), 3.2e10)


def test_configure_classic_uses_visible_alpha_at_reference_mesh():
    solver = _one_tetra_solver()
    info = pr.configure_solver(
        solver, alpha_visible=1e9, element_size_um=14.0)
    assert info["mode"] == "classic_volume_density_huber"
    assert info["internal_alpha"] == 1e9
    assert info["mesh_scale_factor"] == 1.0
    assert solver.physical_regularization["normalization"] == "geometry_h5"


def test_surface_extra_length_keeps_bulk_on_classic_scale():
    solver = _one_tetra_solver()
    info = pr.configure_solver(
        solver,
        alpha_visible=1e9,
        element_size_um=7.0,
        surface_mask=np.ones(4, dtype=bool),
        surface_area_m2=2e-10,
    )
    assert info["mode"] == "surface_traction_l2"
    assert info["surface_thickness_m"] > 0
    assert np.isclose(info["classic_equivalent_alpha"], 3.2e10)
    assert np.isclose(
        info["effective_bulk_alpha"], info["classic_equivalent_alpha"])
    assert np.isclose(
        info["internal_alpha"],
        info["classic_equivalent_alpha"] * info["surface_thickness_m"],
    )
