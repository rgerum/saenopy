"""Execute exported PIV/remeshing stages, including their save/load path."""
from types import SimpleNamespace

import numpy as np
import pytest

import saenopy
from saenopy.get_deformations import PivMesh
from saenopy.stack import Stack
from saenopy.gui.solver.modules.DeformationDetector import DeformationDetector
from saenopy.gui.solver.modules.MeshCreator import MeshCreator


def stack():
    return Stack("z{z}.tif", (1, 1, 1), channels=["0"],
                 image_filenames=np.array([["z0.tif"]]), _shape=(8, 8, 1, 1))


@pytest.mark.parametrize("reference", [False, True])
def test_export_allocates_slots_and_invalidates_old_fits(tmp_path, monkeypatch, reference):
    result = saenopy.Result(stack=[stack(), stack()],
                           stack_reference=stack() if reference else None)
    count = 2 if reference else 1
    result.output = str(tmp_path / "export.saenopy")
    result.piv_parameters_tmp = dict(window_size=35, element_size=14,
                                     signal_to_noise=1.3, drift_correction=True)
    result.mesh_parameters_tmp = dict(reference_stack="first", element_size=14,
                                      mesh_size="piv")
    # Simulate a short/empty archive held in memory by an existing GUI session.
    result.mesh_piv = []
    result.solvers = [saenopy.Solver()]
    calls = []

    def piv(a, b, *params):
        if not calls:
            assert result.mesh_piv == [None] * count
        assert result.solvers == [None] * count
        calls.append((a, b, params))
        return PivMesh(nodes=np.array([[1., 2., 3.]]),
                       displacements_measured=np.array([[4., 5., 6.]]))

    monkeypatch.setattr(saenopy, "get_displacements_from_stacks", piv)
    imports, code = DeformationDetector.get_code(SimpleNamespace(result=result))
    exec(imports + code, {"saenopy": saenopy, "results": [result]})
    assert len(calls) == count
    assert calls[0][:2] == ((result.stack_reference, result.stacks[0]) if reference
                           else (result.stacks[0], result.stacks[1]))
    loaded = saenopy.Result.load(result.output)
    assert len(loaded.mesh_piv) == count
    assert loaded.solvers == [None] * count
    np.testing.assert_array_equal(loaded.mesh_piv[0].displacements_measured,
                                  [[4., 5., 6.]])

    # The remeshing export also works after the GUI has cleared its solver list.
    result.solvers = None
    monkeypatch.setattr(saenopy, "subtract_reference_state",
                        lambda meshes, ref: [m.displacements_measured for m in meshes])
    created = []

    def mesh(piv, displacement, params):
        solver = saenopy.Solver()
        solver.set_nodes(piv.nodes)
        created.append(solver)
        return solver

    monkeypatch.setattr(saenopy, "interpolate_mesh", mesh)
    imports, code = MeshCreator.get_code(SimpleNamespace(result=result))
    exec(imports + code, {"saenopy": saenopy, "results": [result]})
    assert len(created) == count
    loaded = saenopy.Result.load(result.output)
    assert len(loaded.solvers) == count
    np.testing.assert_array_equal(loaded.solvers[0].mesh.nodes, [[1., 2., 3.]])
