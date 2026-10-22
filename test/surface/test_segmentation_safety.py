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
