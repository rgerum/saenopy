"""Preserve measured targets and prescribed displacements through refitting."""
import numpy as np
import pytest

from saenopy.reconstruction import fit_result
from test_release import example_result


@pytest.mark.parametrize("excluded", [[np.nan, np.nan, np.nan], [np.nan, 1e-7, -2e-7]])
def test_target_center_does_not_restore_excluded_measurements(excluded):
    result = example_result()
    solver = result.solvers[0]
    solver.mesh.displacements_target[0] = excluded
    solver.mesh.displacements_target_mask[0] = False
    before = solver.mesh.displacements_target.copy()

    assert np.isfinite(solver.get_center("deformation_target")).all()
    np.testing.assert_array_equal(solver.mesh.displacements_target, before)
    fit_result(result)
    assert not solver.mesh.displacements_target_mask[0]
    np.testing.assert_array_equal(solver.mesh.displacements_target, before)
    assert np.isfinite(solver.mesh.forces).all()


@pytest.mark.parametrize("frames", [1, 2])
def test_warm_start_preserves_prescribed_fixed_displacements(frames):
    result = example_result()
    solver = result.solvers[0]
    solver.mesh.movable[-1] = False
    solver.mesh.displacements[-1] = [1e-8, -2e-8, 3e-8]
    fixed = solver.mesh.displacements[-1].copy()
    initial = solver.mesh.displacements_target.copy()
    if frames == 2:
        previous = example_result().solvers[0]
        previous.mesh.displacements[:] = previous.mesh.nodes * 0.002
        initial = previous.mesh.displacements.copy()
        result.solvers = [previous, solver]
    initial[-1] = fixed
    starts = []

    fit_result(result, frames - 1, parameters=dict(prev_t_as_start=True),
               callback=lambda solver, records, *args:
               starts.append(solver.mesh.displacements.copy()) if not records else None)

    np.testing.assert_array_equal(starts[0], initial)
    np.testing.assert_array_equal(solver.mesh.displacements[-1], fixed)
    assert np.isfinite(solver.mesh.forces).all()


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_warm_start_zeroes_entire_invalid_target_vectors(invalid):
    result = example_result()
    solver = result.solvers[0]
    solver.mesh.displacements_target[0] = [invalid, 2e-7, -3e-7]
    target_before = solver.mesh.displacements_target.copy()
    starts = []

    fit_result(result, parameters=dict(prev_t_as_start=True),
               callback=lambda solver, records, *args:
               starts.append(solver.mesh.displacements.copy()) if not records else None)

    np.testing.assert_array_equal(starts[0][0], np.zeros(3))
    np.testing.assert_array_equal(starts[0][1:], target_before[1:])
    np.testing.assert_array_equal(solver.mesh.displacements_target, target_before)
    assert not solver.mesh.displacements_target_mask[0]
    assert np.isfinite(solver.mesh.displacements).all()
    assert np.isfinite(solver.mesh.forces).all()
