"""Result slots must exist before exported code assigns PIV and solver data."""
import numpy as np
import pytest
import tifffile

from saenopy import Result, Solver, get_stacks
from saenopy.get_deformations import PivMesh
from saenopy.stack import Stack


def stack():
    return Stack("stack_z{z}.tif", (1, 1, 1),
                 image_filenames=np.array([["stack_z0.tif"]]),
                 channels=["0"], _shape=(8, 8, 1, 1))


@pytest.mark.parametrize("count,reference,expected", [
    (0, False, 0), (1, True, 1), (3, True, 3), (3, False, 2),
])
def test_slots_follow_actual_stack_pairs(count, reference, expected):
    result = Result(stack=[stack() for _ in range(count)],
                    stack_reference=stack() if reference else None)
    assert result.mesh_piv == [None] * expected
    assert result.solvers == [None] * expected


@pytest.mark.parametrize("empty", [None, []])
def test_load_repairs_empty_slots_and_roundtrips(tmp_path, empty):
    path = tmp_path / "incomplete.saenopy"
    result = Result(output=path, stack=[stack()], stack_reference=stack())
    # Reproduce archives already written with the old constructor.
    result.mesh_piv = empty
    result.solvers = empty
    result.save()
    loaded = Result.load(path)
    assert loaded.mesh_piv == [None]
    assert loaded.solvers == [None]
    loaded.save()
    again = Result.load(path)
    assert again.mesh_piv == [None]
    assert again.solvers == [None]


def test_partial_results_are_preserved_when_missing_slots_are_added(tmp_path):
    piv = PivMesh(nodes=np.array([[1., 2., 3.]]),
                  displacements_measured=np.array([[4., 5., 6.]]))
    solver = Solver()
    solver.set_nodes(np.array([[7., 8., 9.]]))
    path = tmp_path / "partial.saenopy"
    result = Result(output=path, stack=[stack(), stack()], stack_reference=stack(),
                    mesh_piv=[piv], solvers=[solver])
    assert result.mesh_piv[0] is piv
    assert result.solvers[0] is solver
    assert result.mesh_piv[1] is None
    assert result.solvers[1] is None
    result.save()
    loaded = Result.load(path)
    np.testing.assert_array_equal(loaded.mesh_piv[0].displacements_measured,
                                  piv.displacements_measured)
    np.testing.assert_array_equal(loaded.solvers[0].mesh.nodes, solver.mesh.nodes)
    assert loaded.mesh_piv[1] is None
    assert loaded.solvers[1] is None


def test_get_stacks_new_and_existing_empty_reference_result(tmp_path):
    for folder in ("active", "reference"):
        (tmp_path / folder).mkdir()
        tifffile.imwrite(tmp_path / folder / "z0.tif", np.zeros((8, 8), np.uint8))
    args = dict(filename=tmp_path / "active" / "z{z}.tif",
                reference_stack=tmp_path / "reference" / "z{z}.tif",
                output_path=tmp_path / "results", voxel_size=(1, 1, 1),
                load_existing=True)
    result, = get_stacks(**args)
    assert result.mesh_piv == [None]
    assert result.solvers == [None]
    result.mesh_piv = []
    result.solvers = []
    result.save()
    loaded, = get_stacks(**args)
    assert loaded.mesh_piv == [None]
    assert loaded.solvers == [None]


def test_solver_only_results_are_not_discarded():
    solver = Solver()
    result = Result(solvers=[solver])
    assert result.solvers == [solver]
    assert result.mesh_piv == []
