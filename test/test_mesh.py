from saenopy.solver import SolverMesh
from saenopy.multigrid_helper import get_scaled_mesh
import numpy as np
from saenopy.mesh import InvalidShape
import pytest


def test_mesh_shape():
    nodes, tetrahedra = get_scaled_mesh(3, 10, 10, [5, 5, 5], 1)

    mesh = SolverMesh()
    mesh.energy = np.random.rand(tetrahedra.shape[0] + 1)
    mesh.displacements = np.random.rand(tetrahedra.shape[0] + 1)
    mesh.regularisation_mask = np.random.rand(tetrahedra.shape[0] + 1)

    mesh = SolverMesh(nodes, tetrahedra)
    mesh.energy = np.random.rand(tetrahedra.shape[0])
    with pytest.raises(InvalidShape):
        mesh.energy = np.random.rand(tetrahedra.shape[0]+1)
    with pytest.raises(InvalidShape):
        mesh.energy = np.random.rand(tetrahedra.shape[0], 10)

    mesh.displacements = np.random.rand(nodes.shape[0], 3)
    with pytest.raises(InvalidShape):
        mesh.displacements = np.random.rand(nodes.shape[0], 4)
    with pytest.raises(InvalidShape):
        mesh.displacements = np.random.rand(nodes.shape[0]+1, 3)
    with pytest.raises(InvalidShape):
        mesh.displacements = np.random.rand(nodes.shape[0])

    mesh.regularisation_mask = np.random.rand(nodes.shape[0])
    with pytest.raises(InvalidShape):
        mesh.regularisation_mask = np.random.rand(nodes.shape[0]+1)
    with pytest.raises(InvalidShape):
        mesh.regularisation_mask = np.random.rand(nodes.shape[0], 3)
