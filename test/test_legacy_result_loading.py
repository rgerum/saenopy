import numpy as np
import pytest

from saenopy.result_file import Result
from saenopy.solver import Solver
from saenopy.stack import Stack


def make_stack_dict():
    return {
        "template": "stack_t0_z{z}.tif",
        "voxel_size": (0.18, 0.18, 1.007),
        "crop": None,
        "_shape": (1, 1, 1, 1),
        "image_filenames": [["stack_t0_z0.tif"]],
        "channels": ["0"],
        "packed_files": None,
    }


def test_load_v14_result_with_empty_solver_placeholder():
    data = {
        "stacks": [make_stack_dict(), make_stack_dict()],
        "stack_reference": None,
        "template": "stack_t{t}_z{z}.tif",
        "time_delta": 15.0,
        "piv_parameters": None,
        "mesh_piv": ["__NONE__"],
        "mesh_parameters": None,
        "material_parameters": None,
        "solve_parameters": None,
        "solvers": ["__NONE__"],
        "___save_name__": "Result",
        "___save_version__": "1.4",
    }

    result = Result.from_dict(data)

    assert result.___save_version__ == Result.___save_version__
    assert result.mesh_piv == [None]
    assert result.solvers == [None]
    assert all(
        isinstance(stack.image_filenames, np.ndarray)
        for stack in result.stacks
    )


@pytest.mark.parametrize("version", ["1.2", "1.3"])
def test_migration_reuses_large_image_and_mesh_arrays(version):
    stack = Stack(**make_stack_dict())
    stack.packed_files = np.arange(12, dtype=np.uint8).reshape(3, 4, 1, 1, 1)
    solver = Solver()
    solver.set_nodes(np.eye(3))
    data = Result(stack=[stack], solvers=[solver], solve_parameters={}).to_dict()
    images = data["stacks"][0]["packed_files"]
    nodes = data["solvers"][0]["mesh"]["nodes"]
    data["___save_version__"] = version
    if version == "1.2":
        data["stack"] = data.pop("stacks")
        data["solver"] = data.pop("solvers")

    loaded = Result.from_dict(data)

    assert np.shares_memory(loaded.stacks[0].packed_files, images)
    # Mesh construction later normalizes the node dtype; migration itself
    # must not allocate another copy of the solver's array payload.
    assert np.shares_memory(data["solvers"][0]["mesh"]["nodes"], nodes)
    np.testing.assert_array_equal(loaded.stacks[0].packed_files, images)
    np.testing.assert_array_equal(loaded.solvers[0].mesh.nodes, nodes)
