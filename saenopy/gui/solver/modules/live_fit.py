"""Transient, independent field snapshots for the reconstruction viewers."""
from copy import copy

import numpy as np


def solver_snapshot(solver):
    snapshot = copy(solver)
    snapshot.mesh = copy(solver.mesh)
    # Topology is constant during fitting. Copy the displayed mutable fields,
    # not tetrahedral tensors, sparse matrices or material caches.
    for name in ("forces", "forces_border", "displacements", "displacements_target",
                 "displacements_target_mask", "regularisation_mask", "cell_boundary_mask", "movable"):
        value = getattr(solver.mesh, name, None)
        if isinstance(value, np.ndarray):
            setattr(snapshot.mesh, name, value.copy())
    return snapshot


def displayed_solver(result, frame):
    if getattr(result, "_live_fit_active", False):
        return result._live_fit_solvers.get(frame)
    return result.solvers[frame]


def live_fit_available(result, frame):
    return bool(getattr(result, "_live_fit_active", False)
                and getattr(result, "_live_fit_solvers", {}).get(frame) is not None)
