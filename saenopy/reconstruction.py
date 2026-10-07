"""Shared reconstruction entry points for the GUI, exported Python and sweeps."""
from __future__ import annotations

import numpy as np

from . import physical_regularization as pr
from . import surface_regularization as sr
from .materials import SemiAffineFiberMaterial


def segment_with_params(result, index, params):
    """Return stack, YXZ image, body, shell coordinates and threshold."""
    stack = result.stacks[index]
    channel = int(params.get("seg_channel", sr.DEFAULT_SEG_CHANNEL))
    image = np.asarray(stack[:, :, 0, :, channel])
    threshold = sr.auto_threshold(
        image, method=params.get("seg_threshold_method", sr.DEFAULT_THRESHOLD_METHOD),
        factor=float(params.get("seg_threshold_factor", sr.DEFAULT_THRESHOLD_FACTOR)))
    body, shell = sr.segment_cell(image, stack.voxel_size, threshold=threshold)
    return stack, image, body, shell, threshold


def segmented_surface_area(body, voxel_size):
    """Area in m² of the observed marching-cubes cell/gel interface (YXZ).

    Voxel sizes are XYZ in micrometres. A body touching the image edge is
    reported as open; no artificial end cap or missing surface is added.
    """
    from skimage.measure import marching_cubes, mesh_surface_area
    body = np.asarray(body, dtype=bool)
    if body.ndim != 3 or not np.any(body):
        raise ValueError("segmentation must be a nonempty 3D body")
    if any(np.any(np.take(body, [0, -1], axis=axis)) for axis in range(3)):
        import warnings
        warnings.warn("Cell segmentation touches the image boundary: normalizing to the observed OPEN interface only; no end caps added.",
                      RuntimeWarning, stacklevel=2)
    voxel = np.asarray(voxel_size, dtype=float)
    if voxel.shape != (3,) or not np.all(np.isfinite(voxel)) or np.any(voxel <= 0):
        raise ValueError("voxel_size must contain three positive finite values")
    vertices, triangles, _, _ = marching_cubes(
        body.astype(np.float32), level=0.5,
        spacing=tuple(voxel[[1, 0, 2]] * 1e-6))
    return float(mesh_surface_area(vertices, triangles))


def prepare_surface(result, index=0, parameters=None, *, include_piv=False):
    """Compute segmentation and masks once, without modifying the solver."""
    params = dict(result.solve_parameters or {})
    params.update(parameters or {})
    stack, image, body, shell, threshold = segment_with_params(result, index, params)
    solver = result.solvers[index]
    mask = sr.surface_node_mask(
        solver.mesh.nodes, shell, float(result.mesh_parameters["element_size"]),
        dilate_layers=int(params.get("seg_dilate_layers", 1)))
    mask &= solver.mesh.regularisation_mask & solver.mesh.movable
    if not mask.any():
        raise ValueError("segmentation selected no active surface nodes")
    area = segmented_surface_area(body, stack.voxel_size)
    geometry = dict(mask=mask, inside=sr.inside_cell_mask(
        solver.mesh.nodes, body, stack.voxel_size, image.shape),
        area_m2=area, threshold=float(threshold),
        open_surface=any(np.any(np.take(body, [0, -1], axis=axis)) for axis in range(3)))
    if include_piv:
        nodes = result.mesh_piv[index].nodes
        nodes = nodes - (nodes.max(axis=0) + nodes.min(axis=0)) / 2
        geometry["piv_nodes"] = nodes
        geometry["piv_inside"] = sr.inside_cell_mask(nodes, body, stack.voxel_size, image.shape)
    return geometry


def fit_result(result, index=0, *, parameters=None, material_parameters=None,
               prepared_surface=None, callback=None, cancel_signal=None,
               verbose=False, resume=False):
    """Fit one existing interpolated solver; does not save or clear its cache.

    Missing ``physical_normalization`` retains the legacy behavior. New GUI
    analyses set it explicitly. For a fair A/B fit, set
    ``exclude_cell_interior=True`` in BOTH modes and reuse ``prepared_surface``.
    ``resume=True`` retains the current displacement field, bypassing
    ``prev_t_as_start``. It starts a new iteration budget and diagnostic history
    from that field; it does not restore an interrupted inner CG solve.
    """
    params = dict(result.solve_parameters or {})
    params.update(parameters or {})
    for obsolete in ("surface_area_method",):
        params.pop(obsolete, None)
    if params.get("surface", False):
        params["physical_normalization"] = True
    params.setdefault("alpha_reference_element_size_um", pr.DEFAULT_REFERENCE_ELEMENT_SIZE_UM)
    params.setdefault("scaling_exponent", pr.DEFAULT_SCALING_EXPONENT)
    params.setdefault("surface_min_iterations", sr.DEFAULT_MIN_ITERATIONS)
    from .solver import DEFAULT_CG_MAXITER_FACTOR
    params.setdefault("cg_maxiter_factor", DEFAULT_CG_MAXITER_FACTOR)
    params.setdefault("solver_precision", 1e-18)
    material = dict(result.material_parameters or {})
    material.update(material_parameters or {})
    material = {key: None if value == "None" else value for key, value in material.items()}
    solver = result.solvers[index]
    pr.reset_solver(solver)
    solver.mesh.cell_boundary_mask = None
    # A vector target needs all three finite components. Partial NaNs would
    # otherwise enter the linear system and invalidate the entire force field.
    solver.mesh.displacements_target_mask = np.all(
        np.isfinite(solver.mesh.displacements_target), axis=1)
    solver.set_material_model(SemiAffineFiberMaterial(
        material["k"], material["d_0"], material["lambda_s"], material["d_s"]))
    if params.get("prev_t_as_start", False) and not resume:
        if index > 0:
            previous = result.solvers[index - 1].mesh
            if not np.array_equal(previous.nodes, solver.mesh.nodes):
                raise ValueError("previous time step has a different mesh; disable prev_t_as_start")
            solver.mesh.displacements[:] = previous.displacements
        elif len(result.solvers) == 1:
            solver.mesh.displacements[:] = np.nan_to_num(solver.mesh.displacements_target)
    surface = bool(params.get("surface", False))
    geometry = None
    if surface or params.get("exclude_cell_interior", False):
        geometry = prepared_surface if prepared_surface is not None else prepare_surface(result, index, params)
        sr.drop_targets(solver, geometry["inside"])
    if surface:
        sr.set_surface_regularization(solver, geometry["mask"])
    kwargs = dict(step_size=float(params.get("step_size", 0.2)),
                  cg_maxiter_factor=params["cg_maxiter_factor"],
                  solver_precision=float(params["solver_precision"]),
                  max_iterations=int(params.get("max_iterations", 300)),
                  rel_conv_crit=float(params.get("rel_conv_crit", 0.01)),
                  callback=callback, cancel_signal=cancel_signal, verbose=verbose)
    if surface:
        kwargs["i_min"] = min(int(params["surface_min_iterations"]), kwargs["max_iterations"])
    if params.get("physical_normalization", False):
        pr.solve_regularized(
            solver, alpha=float(params.get("alpha", 1e10)),
            element_size_um=float(result.mesh_parameters["element_size"]),
            surface_mask=geometry["mask"] if surface else None,
            surface_area_m2=geometry["area_m2"] if surface else None,
            reference_element_size_um=float(params["alpha_reference_element_size_um"]),
            exponent=float(params["scaling_exponent"]), **kwargs)
    else:
        solver.solve_regularized(alpha=float(params.get("alpha", 1e10)), **kwargs)
    result.solve_parameters = params
    result.material_parameters = material
    if geometry is not None:
        solver.regularisation_parameters["segmentation_threshold"] = geometry["threshold"]
        solver.regularisation_parameters["surface_is_open"] = bool(geometry.get("open_surface", False))
    solver.regularisation_parameters["surface_min_iterations"] = kwargs.get("i_min", 12)
    return solver
