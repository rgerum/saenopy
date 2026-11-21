"""First save and Save As preserve the source directory of every image."""
from pathlib import Path

import numpy as np
import pytest
import tifffile

from saenopy import Result, add_image_source, get_image_source
from saenopy.stack import Stack


@pytest.mark.parametrize("packed", [False, True])
def test_first_save_and_save_as_keep_matrix_reference_and_sources(tmp_path, monkeypatch, packed):
    images = tmp_path / "images"
    images.mkdir()
    monkeypatch.chdir(images)
    matrix = np.arange(80, dtype=np.uint8).reshape(8, 10)
    reference = matrix + 1
    mask = (matrix > 40).astype(np.uint8)
    surface = np.stack([matrix, matrix + 2, matrix + 4])
    for name, data in (("matrix_z0.tif", matrix), ("reference_z0.tif", reference),
                       ("mask.tif", mask), ("surface.tif", surface)):
        tifffile.imwrite(name, data, photometric="minisblack")
    result = Result(template="matrix_z{z}.tif", stack=[Stack("matrix_z{z}.tif", (1, 1, 1))],
                    stack_reference=Stack("reference_z{z}.tif", (1, 1, 1)))
    mask_key = add_image_source(result, "mask.tif", kind="mask")
    surface_key = add_image_source(result, "surface.tif[z]", kind="surface")
    if packed:
        for stack in [*result.stacks, result.stack_reference,
                      *(source.stacks[0] for source in result.image_sources)]:
            stack.pack_files()

    first = tmp_path / "results" / "first.saenopy"
    result.save(first)
    assert Path(result.output) == first
    loaded = Result.load(first)

    # Save As must resolve against the old archive, even from another CWD.
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    loaded.save("second.saenopy")
    second = other / "second.saenopy"
    assert Path(loaded.output) == second
    if packed:
        for filename in images.glob("*.tif"):
            filename.unlink()
    moved = Result.load(second)
    assert Path(moved.get_absolute_path()).resolve() == images / "matrix_z{z}.tif"
    assert Path(moved.get_absolute_path_reference()).resolve() == images / "reference_z{z}.tif"
    np.testing.assert_array_equal(moved.stacks[0][:, :, 0, 0, 0], matrix)
    np.testing.assert_array_equal(moved.stack_reference[:, :, 0, 0, 0], reference)
    np.testing.assert_array_equal(get_image_source(moved, mask_key, 0)[:, :, 0, 0, 0], mask)
    np.testing.assert_array_equal(get_image_source(moved, surface_key, 0)[:, :, 0, :, 0],
                                  surface.transpose(1, 2, 0))
