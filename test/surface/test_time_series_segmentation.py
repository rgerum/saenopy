"""Every frame gets its own surface; warm starts transfer displacements only."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu

from saenopy import Solver, surface_regularization as sr
from saenopy.reconstruction import fit_result, segment_with_params
from test_release import example_result


class ImageStack:
    voxel_size = (1., 1., 2.)
    channels = ["matrix", "nuclei"]

    def __init__(self, image):
        self.image = image

    def __getitem__(self, key):
        assert key[4] == 1
        return self.image


def test_each_frame_uses_its_own_smoothed_threshold_and_surface():
    images = []
    for offset, brightness in [(0, 50.), (5, 150.)]:
        image = np.full((36, 40, 24), 5., dtype=np.float32)
        image[8:22, 8 + offset:22 + offset, 6:18] = brightness
        images.append(image)
    result = SimpleNamespace(stacks=[ImageStack(im) for im in images])
    params = dict(seg_channel=1, seg_smoothing_sigma_um=2.,
                  seg_threshold_method="otsu", seg_threshold_factor=.9)
    records = [segment_with_params(result, i, params) for i in range(2)]
    for image, (_, selected, body, shell, threshold) in zip(images, records):
        sigma = sr.smoothing_sigma_voxels(2., ImageStack.voxel_size)
        expected = .9 * threshold_otsu(gaussian_filter(image, sigma, truncate=2.))
        assert threshold == pytest.approx(expected)
        np.testing.assert_array_equal(selected, image)
        assert body.any() and not body.all() and len(shell) > 0
    assert records[1][-1] > 2 * records[0][-1]
    assert np.argwhere(records[1][2])[:, 1].mean() == pytest.approx(
        np.argwhere(records[0][2])[:, 1].mean() + 5, abs=.1)
    assert not np.array_equal(records[0][2], records[1][2])


@pytest.mark.parametrize("channel", [-1, .5, None, float("nan")])
def test_invalid_channel_does_not_silently_select_another(channel):
    result = SimpleNamespace(stacks=[ImageStack(None)])
    with pytest.raises(sr.SegmentationError, match="nonnegative integer"):
        segment_with_params(result, 0, {"seg_channel": channel})


@pytest.mark.parametrize("later_channels, message", [
    (["matrix"], "missing at time step 2"),
    (["nuclei", "matrix"], "different channel at time step 2"),
])
def test_inconsistent_series_fails_before_reading_any_images(later_channels, message):
    first, second = ImageStack(None), ImageStack(None)
    second.channels = later_channels
    result = SimpleNamespace(stacks=[first, second])
    with pytest.raises(sr.SegmentationError, match=message):
        segment_with_params(result, 0, {"seg_channel": 1})


def test_previous_field_warm_start_does_not_reuse_previous_surface(monkeypatch):
    import saenopy.reconstruction as reconstruction
    result = example_result()
    result.solvers.append(example_result().solvers[0])
    fields = [np.full((4, 3), 1e-8), np.full((4, 3), 2e-8)]
    geometry = [dict(mask=np.array(mask), inside=np.zeros(4, bool),
                     area_m2=area, threshold=threshold)
                for mask, area, threshold in [
                    ([True, False, False, False], 2e-10, 5.),
                    ([False, True, False, False], 3e-10, 12.)]]
    calls = []

    def prepare(res, index, params):
        assert res is result
        assert params["seg_smoothing_sigma_um"] == 12.
        assert params["seg_threshold_factor"] == 1.
        calls.append(index)
        return geometry[index]

    def solve(solver, **kwargs):
        index = result.solvers.index(solver)
        if index:
            np.testing.assert_array_equal(solver.mesh.displacements, fields[0])
        np.testing.assert_array_equal(solver.mesh.cell_boundary_mask, geometry[index]["mask"])
        solver.mesh.displacements[:] = fields[index]
        solver.regularisation_parameters = dict(kwargs)
        return []

    monkeypatch.setattr(reconstruction, "prepare_surface", prepare)
    monkeypatch.setattr(Solver, "solve_regularized", solve)
    params = dict(surface=True, prev_t_as_start=True,
                  seg_smoothing_sigma_um=12., seg_threshold_factor=1.)
    for i in range(2):
        fit_result(result, i, parameters=params)
        assert result.solvers[i].regularisation_parameters["segmentation_threshold"] == geometry[i]["threshold"]
    assert calls == [0, 1]
    np.testing.assert_array_equal(result.solvers[0].mesh.cell_boundary_mask, geometry[0]["mask"])
