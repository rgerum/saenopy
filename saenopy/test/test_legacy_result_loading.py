import numpy as np

from saenopy.result_file import Result


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

    assert result.___save_version__ == "1.7"
    assert result.mesh_piv == [None]
    assert result.solvers == [None]
    assert all(
        isinstance(stack.image_filenames, np.ndarray)
        for stack in result.stacks
    )
