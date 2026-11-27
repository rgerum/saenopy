"""Convergence failures must never silently become force-reconstruction masks."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_li

from saenopy import surface_regularization as sr
from saenopy.reconstruction import segment_with_params


def test_li_agrees_with_float64_reference():
    rng = np.random.default_rng(8)
    image = rng.normal(12, 2, (25, 27, 23)).astype(np.float32)
    image[8:18, 7:20, 5:17] += 40
    sm = gaussian_filter(image, sigma=2, truncate=2)
    reference = threshold_li(sm.astype(np.float64), tolerance=1e-7) * .6
    assert sr.auto_threshold(image) == pytest.approx(reference, abs=1e-3)


def test_float32_cycle_is_within_stopping_tolerance(monkeypatch):
    # The four-value cycle measured on Pos008/ch00 used to run forever.
    cycle = [16.204492568969727, 16.204504013061523,
             16.20450210571289, 16.20450782775879]
    def fake_li(image, *, tolerance, iter_callback):
        assert tolerance > np.ptp(cycle)
        iter_callback(cycle[0])
        return cycle[0]
    monkeypatch.setattr('skimage.filters.threshold_li', fake_li)
    image = np.linspace(0, 32, 100, dtype=np.float32).reshape(10, 10)
    assert np.isfinite(sr.auto_threshold(image))


def test_nonconvergent_li_is_bounded_and_creates_no_body(monkeypatch):
    values = []
    monkeypatch.setattr(sr, 'LI_MAX_ITERATIONS', 3)
    def oscillating(image, *, tolerance, iter_callback):
        for i in range(100):
            values.append(i)
            iter_callback(1.0 + i % 2)
        pytest.fail('iteration limit was ignored')
    monkeypatch.setattr('skimage.filters.threshold_li', oscillating)
    image = np.arange(64).reshape(4, 4, 4)
    def unexpected(*a, **kw):
        pytest.fail('no morphology allowed after failed threshold')
    monkeypatch.setattr(sr, 'segment_cell', unexpected)
    stack = SimpleNamespace(voxel_size=(1, 1, 1))
    class Stack:
        voxel_size = stack.voxel_size
        def __getitem__(self, key):
            return image
    with pytest.raises(sr.SegmentationError, match='iteration limit'):
        segment_with_params(SimpleNamespace(stacks=[Stack()]), 0, {})
    assert len(values) == 4  # initial guess + 3 iterations


def test_li_deadline(monkeypatch):
    times = iter([0., 31.])
    monkeypatch.setattr(sr, 'monotonic', lambda: next(times))
    with pytest.raises(sr.SegmentationError, match='timed out'):
        sr.auto_threshold(np.arange(125).reshape(5, 5, 5))


@pytest.mark.parametrize('image', [np.ones((3, 3, 3)), np.full((3, 3, 3), np.nan), np.array([])])
def test_no_contrast_or_invalid_image_fails(image):
    with pytest.raises(sr.SegmentationError):
        sr.auto_threshold(image)


@pytest.mark.parametrize('factor', [0, -1, np.nan, np.inf])
def test_invalid_factor(factor):
    with pytest.raises(sr.SegmentationError, match='factor'):
        sr.auto_threshold(np.arange(27).reshape(3, 3, 3), factor=factor)


@pytest.mark.parametrize('surface', [False, True])
def test_failed_surface_preparation_preserves_completed_fit(monkeypatch, surface):
    import saenopy.reconstruction as reconstruction
    from saenopy.saveable import flatten_dict
    from test_release import example_result

    result = example_result()
    geometry = dict(mask=np.array([True, False, False, False]),
                    inside=np.array([True, False, False, False]),
                    area_m2=1e-10, threshold=42., open_surface=False)
    reconstruction.fit_result(result, parameters=dict(surface=True), prepared_surface=geometry)
    solver = result.solvers[0]
    before = {key: np.asarray(value).copy()
              for key, value in flatten_dict(result.to_dict()).items()}
    original_method = solver._compute_regularization_a_and_b
    original_weights = solver.physical_data_weights
    original_material = solver.material_model

    # Exercise the real no-active-nodes failure after image segmentation.
    body = np.zeros((4, 4, 4), bool)
    monkeypatch.setattr(reconstruction, 'segment_with_params', lambda *args: (
        SimpleNamespace(voxel_size=(1., 1., 1.)), body, body, body, 42.))
    monkeypatch.setattr(sr, 'surface_node_mask', lambda *args, **kwargs: np.zeros(4, bool))
    with pytest.raises(ValueError, match='no active surface nodes'):
        reconstruction.fit_result(
            result, parameters=dict(surface=surface, exclude_cell_interior=True, prev_t_as_start=True),
            material_parameters=dict(k=2000.))

    after = flatten_dict(result.to_dict())
    assert after.keys() == before.keys()
    for key, value in before.items():
        np.testing.assert_array_equal(after[key], value, err_msg=key)
    assert solver._compute_regularization_a_and_b is original_method
    assert solver.physical_data_weights is original_weights
    assert solver.material_model is original_material


@pytest.mark.parametrize('prepared', [False, True])
def test_surface_geometry_is_prepared_only_once(monkeypatch, prepared):
    import saenopy.reconstruction as reconstruction
    from test_release import example_result

    result = example_result()
    geometry = dict(mask=np.array([True, False, False, False]),
                    inside=np.array([True, False, False, False]),
                    area_m2=1e-10, threshold=42., open_surface=False)
    calls = []
    monkeypatch.setattr(reconstruction, 'prepare_surface', lambda *args: calls.append(True) or geometry)
    solver = reconstruction.fit_result(
        result, parameters=dict(surface=True), prepared_surface=geometry if prepared else None)
    assert len(calls) == (0 if prepared else 1)
    np.testing.assert_array_equal(solver.mesh.cell_boundary_mask, geometry['mask'])
    assert not solver.mesh.displacements_target_mask[0]
    assert solver.regularisation_parameters['segmentation_threshold'] == 42.
