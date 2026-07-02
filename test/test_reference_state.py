from types import SimpleNamespace

import numpy as np

from saenopy.solver import subtract_reference_state


def create_meshes(displacements):
    return [
        SimpleNamespace(displacements_measured=values)
        for values in displacements
    ]


def test_cumulative_reference_ignores_nans_and_requires_ten_percent_valid():
    displacements = np.full((20, 2, 3), np.nan)
    displacements[:, 0] = 1
    displacements[5, 0] = np.nan
    displacements[0, 1] = 1

    result = subtract_reference_state(create_meshes(displacements), "cumul.")

    np.testing.assert_allclose(result[-1, 0], 19)
    assert np.all(np.isnan(result[:, 1]))


def test_last_reference_ignores_nans_and_requires_ten_percent_valid():
    displacements = np.full((20, 2, 3), np.nan)
    displacements[:, 0] = 1
    displacements[5, 0] = np.nan
    displacements[0, 1] = 1

    result = subtract_reference_state(create_meshes(displacements), "last")

    np.testing.assert_allclose(result[-1, 0], 0)
    assert np.all(np.isnan(result[:, 1]))


def test_median_uses_only_valid_steps_and_requires_ten_percent_valid():
    displacements = np.full((20, 2, 3), np.nan)
    displacements[0, 0] = 1
    displacements[10, 0] = 1
    displacements[0, 1] = 1

    result = subtract_reference_state(create_meshes(displacements), "median")

    np.testing.assert_allclose(result[0, 0], -0.5)
    np.testing.assert_allclose(result[10, 0], 0.5)
    assert np.all(np.isnan(result[:, 1]))
