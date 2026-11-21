"""Image import, saved sources and physical alignment across the two pipelines."""
from pathlib import Path
import shutil

import numpy as np
import pytest
import tifffile

import saenopy
from saenopy.image_sources import image_source_definitions
from saenopy.reconstruction import segment_with_params
from saenopy.stack import Stack


def result_with_frames(tmp_path, count=1, reference=True):
    filename = tmp_path / "matrix.tif"
    tifffile.imwrite(filename, np.zeros((21, 31), dtype=np.uint8))
    stacks = [Stack(str(filename), (2., 3., 4.),
                    image_filenames=np.array([[str(filename)]]), channels=[""])
              for _ in range(count)]
    return saenopy.Result(output=tmp_path / "results" / "sample.saenopy",
                          template=str(filename), stack=stacks,
                          stack_reference=stacks[0] if reference else None)


def write_surface(tmp_path, stem="cell", shift=0):
    image = np.zeros((21, 31, 13), dtype=np.uint8)
    image[6:15, 10 + shift:21 + shift, 3:10] = 100
    for z in range(image.shape[2]):
        tifffile.imwrite(tmp_path / f"{stem}_z{z}.tif", image[:, :, z])
    return str(tmp_path / f"{stem}_z{{z}}.tif"), image


def test_plain_2d_mask_defaults_and_series_natural_order(tmp_path):
    result = result_with_frames(tmp_path, count=2)
    for time, value in ((10, 255), (2, 0)):
        tifffile.imwrite(tmp_path / f"mask_t{time}.tif", np.full((21, 31), value, np.uint8))
    key = saenopy.add_image_source(result, str(tmp_path / "mask_t{t}.tif"),
                                   kind="mask", time_mode="series")
    result.mesh_parameters = {"custom_mask": key}
    assert saenopy.get_custom_mask(result, 0).voxel_size == (2., 3., 4.)
    assert not saenopy.get_custom_mask(result, 0)[:, :, 0, :, 0].any()
    assert saenopy.get_custom_mask(result, 1)[:, :, 0, :, 0].all()
    assert saenopy.get_custom_mask(result, 0, {}) is None
    single = saenopy.add_image_source(result, str(tmp_path / "mask_t10.tif"), kind="mask")
    assert saenopy.get_image_source(result, single, 0) is saenopy.get_image_source(result, single, 1)


def test_mask_series_can_match_deformation_pairs(tmp_path):
    result = result_with_frames(tmp_path, count=3, reference=False)
    for time in range(2):
        tifffile.imwrite(tmp_path / f"mask_t{time}.tif", np.zeros((21, 31), np.uint8))
    key = saenopy.add_image_source(result, str(tmp_path / "mask_t{t}.tif"), kind="mask", time_mode="series")
    assert len(result.image_sources[0].stacks) == len(result.mesh_piv) == 2
    assert saenopy.get_image_source(result, key, 1).shape[:2] == (21, 31)


def test_surface_uses_own_voxel_size_and_common_center(tmp_path):
    result = result_with_frames(tmp_path)
    path, image = write_surface(tmp_path)
    key = saenopy.add_image_source(result, path, voxel_size=(.5, 1.5, 2.))
    params = dict(seg_channel=key, seg_threshold_method="otsu", seg_threshold_factor=1.)
    stack, actual, body, shell, _ = segment_with_params(result, 0, params)
    np.testing.assert_array_equal(actual, image)
    assert stack.voxel_size == (.5, 1.5, 2.)
    # Symmetric fluorescent object stays at the matrix centre despite a zoom change.
    np.testing.assert_allclose((shell.min(axis=0) + shell.max(axis=0)) / 2, 0, atol=1e-15)
    indices = shell / np.array([1.5, .5, 2.]) / 1e-6 + (np.array(image.shape) - 1) / 2
    np.testing.assert_allclose(indices, np.round(indices), atol=1e-10)
    assert body.any()


def test_surface_series_segments_each_timepoint(tmp_path):
    result = result_with_frames(tmp_path, count=2)
    write_surface(tmp_path, "cell_t0")
    write_surface(tmp_path, "cell_t1", shift=3)
    key = saenopy.add_image_source(result, str(tmp_path / "cell_t{t}_z{z}.tif"), time_mode="series")
    bodies = [segment_with_params(result, i, {"seg_channel": key})[2] for i in range(2)]
    assert np.argwhere(bodies[1])[:, 1].mean() == pytest.approx(np.argwhere(bodies[0])[:, 1].mean() + 3, abs=.1)


def test_sources_survive_save_as_and_directory_relocation(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    result = result_with_frames(original)
    surface, image = write_surface(original)
    key = saenopy.add_image_source(result, surface, voxel_size=(.5, 1.5, 2.), name="zoom channel")
    mask_path = original / "ignore.tif"
    tifffile.imwrite(mask_path, np.ones((21, 31), np.uint8))
    mask = saenopy.add_image_source(result, mask_path, kind="mask")
    result.solve_parameters = {"seg_channel": key}
    result.mesh_parameters = {"custom_mask": mask}
    result.save()
    loaded = saenopy.Result.load(result.output)
    assert loaded.solve_parameters["seg_channel"] == key
    assert loaded.image_sources[0].name == "zoom channel"
    np.testing.assert_array_equal(saenopy.get_image_source(loaded, key, 0)[:, :, 0, :, 0], image)
    loaded.save(original / "other_results" / "copy.saenopy")
    relocated = tmp_path / "relocated"
    shutil.copytree(original, relocated)
    shutil.rmtree(original)
    moved = saenopy.Result.load(relocated / "other_results" / "copy.saenopy")
    np.testing.assert_array_equal(saenopy.get_image_source(moved, key, 0)[:, :, 0, :, 0], image)
    assert saenopy.get_custom_mask(moved, 0)[:, :, 0, :, 0].all()
    # Recreating sources for Python export also resolves relocated image paths.
    definitions = image_source_definitions(moved)
    moved.image_sources = []
    for definition in definitions:
        saenopy.add_image_source(moved, **definition)
    np.testing.assert_array_equal(saenopy.get_image_source(moved, key, 0)[:, :, 0, :, 0], image)


def test_format_18_migrates_without_sources(tmp_path):
    result = result_with_frames(tmp_path)
    data = result.to_dict()
    data.pop("image_sources")
    data["___save_version__"] = "1.8"
    loaded = saenopy.Result.from_dict(data)
    assert loaded.image_sources == []
    assert loaded.___save_version__ == "1.9"
    assert loaded.solve_parameters is None


@pytest.mark.parametrize("voxel", [(1, 1), (1, 0, 1), (1, np.inf, 1)])
def test_invalid_voxel_sizes_do_not_attach_source(tmp_path, voxel):
    result = result_with_frames(tmp_path)
    with pytest.raises(ValueError, match="Voxel size"):
        saenopy.add_image_source(result, tmp_path / "matrix.tif", kind="mask", voxel_size=voxel)
    assert result.image_sources == []


def test_mismatched_time_counts_and_multiplane_masks_rejected(tmp_path):
    result = result_with_frames(tmp_path, count=3)
    for t in range(2):
        tifffile.imwrite(tmp_path / f"mask_t{t}.tif", np.zeros((21, 31), np.uint8))
    with pytest.raises(ValueError, match="expected 3"):
        saenopy.add_image_source(result, tmp_path / "mask_t{t}.tif", kind="mask", time_mode="series")
    with pytest.raises(ValueError, match="one image/stack"):
        saenopy.add_image_source(result, tmp_path / "mask_t{t}.tif", kind="mask")
    surface, _ = write_surface(tmp_path)
    with pytest.raises(ValueError, match="must be 2D"):
        saenopy.add_image_source(result, surface, kind="mask")
    assert result.image_sources == []


def test_multipage_tiff_and_selected_channel(tmp_path):
    result = result_with_frames(tmp_path)
    path, image = write_surface(tmp_path)
    tifffile.imwrite(tmp_path / "volume.tif", image.transpose(2, 0, 1), photometric="minisblack")
    key = saenopy.add_image_source(result, tmp_path / "volume.tif[z]")
    np.testing.assert_array_equal(saenopy.get_image_source(result, key, 0)[:, :, 0, :, 0], image)
    for channel in (0, 1):
        tifffile.imwrite(tmp_path / f"mask_c{channel}.tif", np.full((21, 31), channel, np.uint8))
    key = saenopy.add_image_source(result, tmp_path / "mask_c{c:1}.tif", kind="mask")
    stack = saenopy.get_image_source(result, key, 0)
    assert stack.channels == ["1"]
    assert stack[:, :, 0, :, 0].all()


def test_missing_surface_slice_is_not_silently_recentered(tmp_path):
    result = result_with_frames(tmp_path, count=2)
    write_surface(tmp_path, "cell_t0")
    write_surface(tmp_path, "cell_t1")
    (tmp_path / "cell_t1_z0.tif").unlink()
    with pytest.raises(ValueError, match="every selected slice"):
        saenopy.add_image_source(result, tmp_path / "cell_t{t}_z{z}.tif", time_mode="series")
    assert result.image_sources == []
