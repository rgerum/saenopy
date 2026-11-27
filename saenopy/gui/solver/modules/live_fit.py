"""Transient, independent field snapshots for the reconstruction viewers."""
import numpy as np
from saenopy import Solver


def solver_snapshot(solver, *, pending=False):
    # The save/load construction excludes computation caches and bound overrides
    # that would otherwise keep the worker solver alive after clear_cache().
    data = solver.to_dict()
    # Display snapshots may have incomplete metadata while a fit is starting.
    data["regularisation_parameters"] = None
    if pending:
        data["regularisation_results"] = None
    snapshot = Solver.from_dict(data)
    if not pending and solver.regularisation_parameters is not None:
        snapshot.regularisation_parameters = dict(solver.regularisation_parameters)
    # Independently own the mutable fields used by the viewers.
    for name in ("forces", "forces_border", "displacements", "displacements_target",
                 "displacements_target_mask", "regularisation_mask", "cell_boundary_mask", "movable"):
        value = getattr(solver.mesh, name, None)
        if isinstance(value, np.ndarray):
            setattr(snapshot.mesh, name, value.copy())
    preview = getattr(solver.mesh, "_segmentation_preview", None)
    if preview is not None:
        snapshot.mesh._segmentation_preview = tuple(value.copy() for value in preview)
    return snapshot


def displayed_solver(result, frame):
    if getattr(result, "_live_fit_active", False):
        return getattr(result, "_live_fit_solvers", {}).get(frame)
    solvers = getattr(result, "solvers", None) or []
    return solvers[frame] if 0 <= frame < len(solvers) else None


def live_fit_available(result, frame):
    return bool(getattr(result, "_live_fit_active", False)
                and getattr(result, "_live_fit_solvers", {}).get(frame) is not None)


def fit_status_label(result, description, frame=0):
    """Describe the displayed fit without calling an intermediate field final."""
    state = getattr(result, "solve_parameters_state", "")
    status = {"scheduled": "queued...", "running": "in progress...",
              "cancelling": "cancelling...", "failed": "fit failed"}.get(state)
    if getattr(result, "_live_fit_active", False) and state not in (
            "scheduled", "running", "cancelling"):
        # Snapshots are enabled just before a new fit is queued, including
        # when restarting a result whose previous state was finished/failed.
        status = "in progress..."
    if status is None:
        solvers = getattr(result, "solvers", None) or []
        if 0 <= frame < len(solvers):
            params = getattr(solvers[frame], "regularisation_parameters", None) or {}
            if params.get("cancelled", False):
                status = "fit cancelled; partial result"
    if status:
        if getattr(result, "_live_fit_active", False) and not live_fit_available(result, frame):
            status += "; waiting for this frame"
        return f"{description} <b>({status})</b>"
    return description
