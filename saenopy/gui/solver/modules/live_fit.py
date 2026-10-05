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


def fit_status_label(result, description):
    """Describe the displayed fit without calling an intermediate field final."""
    state = getattr(result, "solve_parameters_state", "")
    status = {"scheduled": "queued...", "running": "in progress...",
              "cancelling": "cancelling...", "failed": "fit failed"}.get(state)
    if getattr(result, "_live_fit_active", False) and state not in (
            "scheduled", "running", "cancelling"):
        # Snapshots are enabled just before a new fit is queued, including
        # when restarting a result whose previous state was finished/failed.
        status = "in progress..."
    if status:
        return f"{description} <b>({status})</b>"
    return description
