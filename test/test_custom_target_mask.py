import numpy as np
import pytest

import saenopy
from saenopy.get_deformations import PivMesh
from saenopy.solver import _sample_custom_mask
from saenopy.stack import Stack


def test_mask_uses_centered_yx_coordinates_and_xyz_voxel_sizes():
    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[3, 1] = 255
    # Y offset = 1 * 3 µm; X offset = -2 * 2 µm; Z is unrestricted.
    nodes = np.array([[3, -4, 5], [0, 0, 0], [-4, 3, 5],
                      [30, -4, 5], [3, -4, 50]]) * 1e-6
    np.testing.assert_array_equal(
        _sample_custom_mask(nodes, mask, (2, 3, 5)),
        [True, False, False, False, True])


def test_even_shape_center_and_nearest_pixel_footprint():
    mask = np.zeros((4, 6), dtype=bool)
    mask[2, 3] = True
    # The origin is between central pixels; these sample the same voxel.
    nodes = np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9],
                      [-0.1, 0.1, 0.1], [1.1, 0.1, 0.1]]) * 1e-6
    np.testing.assert_array_equal(_sample_custom_mask(nodes, mask, (1, 1, 1)),
                                  [True, True, False, False])


def test_2d_and_single_plane_masks_exclude_complete_z_columns():
    mask = np.zeros((3, 5), dtype=bool)
    mask[1, 2] = True
    nodes = np.array([[0, 0, -20], [0, 0, 20], [2, 0, 0], [0, 0, 0]]) * 1e-6
    np.testing.assert_array_equal(_sample_custom_mask(nodes, mask, (1, 1, 1)),
                                  [True, True, False, True])
    np.testing.assert_array_equal(_sample_custom_mask(nodes, mask[:, :, None], (1, 1, 1)),
                                  [True, True, False, True])


def test_stack_uses_its_voxel_sizes(tmp_path):
    import tifffile
    image = np.zeros((5, 7), dtype=np.uint8)
    image[3, 1] = 255
    tifffile.imwrite(tmp_path / "mask.tif", image)
    stack = Stack(str(tmp_path / "mask.tif") + "[z]", (2, 3, 5))
    np.testing.assert_array_equal(
        _sample_custom_mask(np.array([[3, -4, 5], [0, 0, 0]]) * 1e-6, stack, None),
        [True, False])


@pytest.mark.parametrize("mask", [np.zeros(3), np.zeros((2, 2, 2)), np.zeros((2, 2, 2, 2)),
                                       np.zeros((0, 2)), np.array([[np.nan]]),
                                       np.array([[np.inf]]), np.array([["mask"]])])
def test_invalid_masks_are_rejected(mask):
    with pytest.raises(ValueError, match="Custom mask"):
        _sample_custom_mask(np.zeros((1, 3)), mask, (1, 1, 1))


@pytest.mark.parametrize("voxel", [None, (1, 2), (1, 0, 1), (1, -1, 1),
                                        (1, np.inf, 1), (1, np.nan, 1), "bad"])
def test_invalid_or_missing_voxel_sizes_are_rejected(voxel):
    with pytest.raises(ValueError, match="voxel size"):
        _sample_custom_mask(np.zeros((1, 3)), np.ones((3, 3)), voxel)


def test_exclusion_is_applied_after_interpolation_and_preserves_other_inputs():
    nodes = np.stack(np.meshgrid(*([np.arange(-4, 5) * 1e-6] * 3),
                                indexing="ij"), axis=-1).reshape(-1, 3)
    displacement = np.full_like(nodes, 1e-6)
    piv = PivMesh(nodes=nodes, displacements_measured=displacement)
    params = dict(mesh_size="piv", element_size=2)
    baseline = saenopy.interpolate_mesh(piv, displacement, params)
    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[3:6, 3:6] = 255
    mask_before = mask.copy()
    result = saenopy.interpolate_mesh(piv, displacement, params,
                                     custom_mask=mask, mask_voxel_size=(1, 1, 1))
    # Constant finite PIV data would interpolate finite vectors everywhere;
    # an interior image mask must still leave holes in the final targets.
    excluded = np.all(np.abs(result.mesh.nodes[:, :2]) < 1.5e-6, axis=1)
    assert excluded.any() and (~excluded).any()
    assert np.isfinite(baseline.mesh.displacements_target).all()
    assert np.isnan(result.mesh.displacements_target[excluded]).all()
    assert not result.mesh.displacements_target_mask[excluded].any()
    np.testing.assert_array_equal(result.mesh.displacements_target[~excluded],
                                  baseline.mesh.displacements_target[~excluded])
    np.testing.assert_array_equal(result.mesh.regularisation_mask,
                                  baseline.mesh.regularisation_mask)
    np.testing.assert_array_equal(piv.displacements_measured, np.full_like(nodes, 1e-6))
    np.testing.assert_array_equal(mask, mask_before)


def test_empty_mask_preserves_the_existing_interpolation():
    nodes = np.stack(np.meshgrid(*([np.arange(5) * 1e-6] * 3),
                                indexing="ij"), axis=-1).reshape(-1, 3)
    displacement = nodes.copy()
    displacement[10] = np.nan
    piv = PivMesh(nodes=nodes)
    params = dict(mesh_size="piv", element_size=1)
    baseline = saenopy.interpolate_mesh(piv, displacement, params)
    result = saenopy.interpolate_mesh(piv, displacement, params,
                                     custom_mask=np.zeros((5, 5)),
                                     mask_voxel_size=(1, 1, 1))
    np.testing.assert_array_equal(result.mesh.displacements_target,
                                  baseline.mesh.displacements_target)
    np.testing.assert_array_equal(result.mesh.displacements_target_mask,
                                  baseline.mesh.displacements_target_mask)
