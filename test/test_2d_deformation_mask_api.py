"""Image-file masks can be used directly in a Python interpolation pipeline."""
import numpy as np
import pytest
import tifffile

import saenopy
from saenopy.get_deformations import PivMesh


def test_image_mask_series_excludes_targets_without_a_result(tmp_path):
    nodes = np.stack(np.meshgrid(*([np.arange(-4, 5) * 1e-6] * 3),
                                indexing="ij"), axis=-1).reshape(-1, 3)
    displacements = np.full_like(nodes, 1e-6)
    piv = PivMesh(nodes=nodes, displacements_measured=displacements)
    empty = np.zeros((9, 9), np.uint8)
    masked = empty.copy()
    masked[3:6, 3:6] = 255
    tifffile.imwrite(tmp_path / "ignore_t2.tif", empty)
    tifffile.imwrite(tmp_path / "ignore_t10.tif", masked)
    masks = saenopy.load_2d_deformation_masks(tmp_path / "ignore_t{t}.tif", (1, 1, 2))
    params = dict(mesh_size="piv", element_size=2)
    solvers = [saenopy.interpolate_mesh(piv, displacements, params, ignore_mask_2d=mask)
               for mask in masks]
    assert len(solvers) == 2
    assert np.isfinite(solvers[0].mesh.displacements_target).all()
    mesh = solvers[1].mesh
    excluded = np.all(np.abs(mesh.nodes[:, :2]) < 1.5e-6, axis=1)
    assert excluded.any() and (~excluded).any()
    assert np.isnan(mesh.displacements_target[excluded]).all()
    assert np.isfinite(mesh.displacements_target[~excluded]).all()
    single = saenopy.load_2d_deformation_masks(tmp_path / "ignore_t10.tif", (1, 1, 2))
    assert len(single) == 1
    alias = saenopy.interpolate_mesh(piv, displacements, params, custom_mask=single[0])
    np.testing.assert_array_equal(alias.mesh.displacements_target, mesh.displacements_target)


def test_file_mask_loader_rejects_3d_stacks(tmp_path):
    tifffile.imwrite(tmp_path / "volume.tif", np.zeros((3, 9, 9), np.uint8), photometric="minisblack")
    with pytest.raises(ValueError, match="must be 2D"):
        saenopy.load_2d_deformation_masks(str(tmp_path / "volume.tif") + "[z]", (1, 1, 1))


@pytest.mark.parametrize("voxel", [(1, 0, 1), (1, np.inf, 1), (1, 1)])
def test_file_mask_loader_requires_physical_pixel_sizes(tmp_path, voxel):
    with pytest.raises(ValueError, match="Voxel size"):
        saenopy.load_2d_deformation_masks(tmp_path / "mask.tif", voxel)


def test_mask_aliases_cannot_silently_override_each_other():
    mask = np.ones((3, 3))
    with pytest.raises(ValueError, match="compatibility alias"):
        saenopy.interpolate_mesh(None, None, {}, ignore_mask_2d=mask, custom_mask=mask)
