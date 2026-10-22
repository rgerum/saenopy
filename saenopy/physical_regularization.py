"""Shared geometry-normalized backend for Classic and Surface regularization.

This module is not a second physical solver and it is not bulk-only.  It
installs the objective weights used by both modes and dispatches according to
whether a surface mask is supplied: no mask means Classic volume-density
Huber, a mask means Surface traction L2 with the derived bulk penalty.

This module contains no fitted calibration constant.  It keeps saenopy's
forces as nodal forces in newtons and only changes the quadrature weights of
the inverse objective on an individual solver instance.

The user-facing alpha is mapped as

    classic: alpha_h = alpha * (h_ref / h)**5
    surface: alpha_h = alpha * (h_ref / h)**5 * d

where ``V_ref = V_regularized / (N_bulk + N_surface)`` and
``A_ref = A_cell / N_surface``. Thus
``d = V_ref / A_ref = (V_regularized / A_cell) * N_surface / (N_bulk + N_surface)``
is computed from the current mesh and measured interface in metres. The
surface solver uses ``bulk_factor = 1/d`` so the resulting bulk coefficient is
again exactly ``alpha * (h_ref / h)**5``.  Hence the only classic/surface
bridge is the geometric nodal thickness, not an empirical number.
"""
from __future__ import annotations

from types import MethodType

import numpy as np
import scipy.sparse as ssp
from scipy.spatial import cKDTree


DEFAULT_SCALING_EXPONENT = 5.0
DEFAULT_REFERENCE_ELEMENT_SIZE_UM = 14.0


def mesh_scale_factor(element_size_um: float, reference_element_size_um: float,
                      exponent: float = DEFAULT_SCALING_EXPONENT) -> float:
    """Return the mesh-refinement correction ``(h_ref / h)**exponent``."""
    h = float(element_size_um)
    h_ref = float(reference_element_size_um)
    if not np.all(np.isfinite([h, h_ref, exponent])) or h <= 0 or h_ref <= 0:
        raise ValueError("element sizes must be finite and positive; exponent must be finite")
    return (h_ref / h) ** float(exponent)


def tetrahedral_nodal_volumes(nodes: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    """Return lumped nodal volumes (one quarter of each tetrahedron per vertex)."""
    nodes = np.asarray(nodes, dtype=float)
    tetrahedra = np.asarray(tetrahedra, dtype=int)
    xyz = nodes[tetrahedra]
    matrices = xyz[:, 1:] - xyz[:, :1]
    tet_volume = np.abs(np.linalg.det(matrices)) / 6.0
    volumes = np.zeros(len(nodes), dtype=float)
    np.add.at(volumes, tetrahedra.ravel(), np.repeat(tet_volume / 4.0, 4))
    if np.any(volumes <= 0) or not np.all(np.isfinite(volumes)):
        raise ValueError("every mesh node needs a finite positive lumped volume")
    return volumes


def surface_areas_from_node_spacing(nodes: np.ndarray, surface_mask: np.ndarray,
                                    neighbours: int = 6) -> np.ndarray:
    """Estimate tributary surface areas from surface-node neighbour spacing.

    The estimate is purely geometric: for each selected node, the median
    distance to its nearest selected neighbours is squared.  No fitted scale
    factor is applied.  Exact nodal areas should be supplied instead whenever
    a triangulated surface is available (as in the synthetic cube).
    """
    nodes = np.asarray(nodes, dtype=float)
    surface_mask = np.asarray(surface_mask, dtype=bool)
    if surface_mask.shape != (len(nodes),):
        raise ValueError("surface_mask must contain one value per node")
    ids = np.flatnonzero(surface_mask)
    if len(ids) < 4:
        raise ValueError("at least four surface nodes are required")
    points = nodes[ids]
    k = min(int(neighbours) + 1, len(points))
    distances, _ = cKDTree(points).query(points, k=k)
    if distances.ndim == 1:
        distances = distances[:, None]
    spacing = np.median(distances[:, 1:], axis=1)
    if np.any(spacing <= 0) or not np.all(np.isfinite(spacing)):
        raise ValueError("surface-node spacing must be finite and positive")
    areas = np.zeros(len(nodes), dtype=float)
    areas[ids] = spacing**2
    return areas


def reference_measures(nodal_volumes: np.ndarray, volume_mask: np.ndarray,
                       surface_areas: np.ndarray | None = None) -> dict[str, float]:
    """Return V_ref, A_ref and d = V_ref / A_ref (in metres).

    V_ref = V_regularized / (N_bulk + N_surface) and
    A_ref = A_cell / N_surface. Thus
    d = (V_regularized / A_cell) * N_surface / (N_bulk + N_surface).
    This is the normalization convention, not a fitted Classic/Surface match.
    """
    volumes = np.asarray(nodal_volumes, dtype=float)
    volume_mask = np.asarray(volume_mask, dtype=bool)
    if volumes.shape != volume_mask.shape or not np.any(volume_mask):
        raise ValueError("volume_mask must select at least one nodal volume")
    if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0):
        raise ValueError("nodal volumes must be finite and positive")
    volume_reference = float(np.mean(volumes[volume_mask]))
    result = {"volume_per_node_m3": volume_reference,
              "regularized_node_count": int(volume_mask.sum()),
              "regularized_volume_m3": float(volumes[volume_mask].sum())}
    if surface_areas is not None:
        areas = np.asarray(surface_areas, dtype=float)
        if areas.shape != volumes.shape:
            raise ValueError("surface_areas must contain one value per node")
        surface = areas > 0
        if not np.all(np.isfinite(areas)) or np.any(areas < 0) or np.any(surface & ~volume_mask):
            raise ValueError("surface areas must be finite, nonnegative and inside volume_mask")
        if not np.any(surface):
            raise ValueError("surface_areas must select at least one node")
        area_reference = float(np.mean(areas[surface]))
        thickness = volume_reference / area_reference
        result.update({
            "surface_node_count": int(surface.sum()),
            "bulk_node_count": int(np.count_nonzero(volume_mask & ~surface)),
            "surface_area_m2": float(areas[surface].sum()),
            "area_per_surface_node_m2": area_reference,
            "surface_thickness_m": thickness,
            "surface_thickness_um": thickness * 1e6,
            "bulk_factor_per_m": 1.0 / thickness,
        })
    return result


def classic_internal_alpha(alpha_visible: float, element_size_um: float,
                           reference_element_size_um: float,
                           exponent: float = DEFAULT_SCALING_EXPONENT) -> float:
    """Map visible alpha to Classic using only the common h**5 correction."""
    if not np.isfinite(alpha_visible) or alpha_visible < 0:
        raise ValueError("alpha_visible must be finite and nonnegative")
    return float(alpha_visible) * mesh_scale_factor(
        element_size_um, reference_element_size_um, exponent
    )


def surface_internal_alpha(alpha_visible: float, element_size_um: float,
                           reference_element_size_um: float,
                           surface_thickness_m: float,
                           exponent: float = DEFAULT_SCALING_EXPONENT) -> float:
    """Map visible alpha to Surface with the additional geometric length ``d``."""
    d = float(surface_thickness_m)
    if d <= 0 or not np.isfinite(d):
        raise ValueError("surface_thickness_m must be finite and positive")
    return classic_internal_alpha(
        alpha_visible, element_size_um, reference_element_size_um, exponent
    ) * d


def configure_solver(solver, alpha_visible: float, element_size_um: float,
                     surface_mask: np.ndarray | None = None,
                     reference_element_size_um: float = DEFAULT_REFERENCE_ELEMENT_SIZE_UM,
                     exponent: float = DEFAULT_SCALING_EXPONENT,
                     *, surface_area_m2: float | None = None,
                     surface_areas: np.ndarray | None = None
                     ) -> dict[str, float | str | int]:
    """Install the geometry-normalized objective and return its provenance.

    ``surface_mask=None`` configures the Classic volume-density Huber objective.
    Passing a surface mask configures the Surface traction objective and derives
    its additional length and bulk suppression entirely from the current mesh.
    The returned ``internal_alpha`` is the value that must be passed to
    :meth:`Solver.solve_regularized`.
    """
    volumes = tetrahedral_nodal_volumes(
        solver.mesh.nodes, solver.mesh.tetrahedra
    )
    volume_mask = np.ones(len(volumes), dtype=bool)
    if solver.mesh.regularisation_mask is not None:
        volume_mask &= np.asarray(solver.mesh.regularisation_mask, dtype=bool)
    if solver.mesh.movable is not None:
        volume_mask &= np.asarray(solver.mesh.movable, dtype=bool)

    mesh_factor = mesh_scale_factor(
        element_size_um, reference_element_size_um, exponent
    )
    common_alpha = classic_internal_alpha(
        alpha_visible, element_size_um, reference_element_size_um, exponent
    )
    provenance: dict[str, float | str | int] = {
        "normalization": "geometry_h5",
        "alpha_visible": float(alpha_visible),
        "element_size_um": float(element_size_um),
        "reference_element_size_um": float(reference_element_size_um),
        "scaling_exponent": float(exponent),
        "mesh_scale_factor": float(mesh_factor),
        "classic_equivalent_alpha": float(common_alpha),
    }

    reset_solver(solver)
    if surface_mask is None:
        solver.mesh.cell_boundary_mask = None
        measures = reference_measures(volumes, volume_mask)
        install_volume_density_huber(
            solver, volumes, measures["volume_per_node_m3"]
        )
        internal_alpha = common_alpha
        provenance.update({
            "mode": "classic_volume_density_huber",
            "reference_volume_m3": measures["volume_per_node_m3"],
            "internal_alpha": float(internal_alpha),
            "effective_bulk_alpha": float(common_alpha),
        })
    else:
        surface_mask = np.asarray(surface_mask, dtype=bool)
        if surface_mask.shape != volume_mask.shape:
            raise ValueError("surface_mask must contain one value per node")
        # The outer boundary remains excluded, exactly as in Classic Saenopy.
        surface_mask = surface_mask & volume_mask
        if not np.any(surface_mask):
            raise ValueError("no active surface nodes inside the regularisation mask")
        if surface_areas is not None:
            areas = np.asarray(surface_areas, dtype=float).copy()
            if (areas.shape != volumes.shape or not np.all(np.isfinite(areas))
                    or np.any(areas[surface_mask] <= 0) or np.any(areas < 0)):
                raise ValueError("surface_areas must be finite, positive on every active surface node")
            areas[~surface_mask] = 0
            area_method = "nodal_areas"
        elif surface_area_m2 is not None:
            if not np.isfinite(surface_area_m2) or surface_area_m2 <= 0:
                raise ValueError("surface_area_m2 must be finite and positive")
            areas = np.zeros(len(volumes))
            areas[surface_mask] = float(surface_area_m2) / int(surface_mask.sum())
            area_method = "measured_area_equal_partition"
        else:
            raise ValueError("provide surface_area_m2 or surface_areas from the cell geometry")
        solver.mesh.cell_boundary_mask = surface_mask.copy()
        measures = reference_measures(volumes, volume_mask, areas)
        install_surface_traction_l2(
            solver,
            volumes,
            areas,
            measures["volume_per_node_m3"],
            measures["area_per_surface_node_m2"],
        )
        internal_alpha = surface_internal_alpha(
            alpha_visible,
            element_size_um,
            reference_element_size_um,
            measures["surface_thickness_m"],
            exponent,
        )
        provenance.update({
            "mode": "surface_traction_l2",
            "reference_volume_m3": measures["volume_per_node_m3"],
            "reference_area_m2": measures["area_per_surface_node_m2"],
            "surface_thickness_m": measures["surface_thickness_m"],
            "surface_thickness_um": measures["surface_thickness_um"],
            "bulk_factor_per_m": measures["bulk_factor_per_m"],
            "internal_alpha": float(internal_alpha),
            "effective_bulk_alpha": float(
                internal_alpha * measures["bulk_factor_per_m"]
            ),
            "surface_node_count": int(np.count_nonzero(surface_mask)),
            "surface_area_method": area_method,
        })

    provenance.update(measures)
    solver.physical_regularization.update(provenance)
    return provenance


_OVERRIDDEN_METHODS = ("_update_local_regularization_weigth",
                       "_compute_regularization_a_and_b", "_record_regularization_status")


def reset_solver(solver):
    """Restore methods replaced by this module; preserve masks and boundary policy."""
    previous = solver.__dict__.pop("_physical_original_methods", None)
    if previous is not None:
        for name, value in previous.items():
            if value is None:
                solver.__dict__.pop(name, None)
            else:
                setattr(solver, name, value)
    for name in ("physical_regularization", "physical_data_weights"):
        solver.__dict__.pop(name, None)


def solve_regularized(solver, alpha: float, element_size_um: float, *,
                      surface_mask=None, surface_area_m2=None, surface_areas=None,
                      reference_element_size_um=14.0,
                      exponent=5.0, **kwargs):
    """Public normalized Classic/Surface API. Coordinates/areas use SI units.

    Classic uses density-Huber; Surface uses L2 and a 60-iteration floor,
    capped by the requested iteration budget. The original boundary mask is kept.
    """
    from .surface_regularization import DEFAULT_MIN_ITERATIONS
    if kwargs.get("method", "huber") != "huber":
        raise ValueError("normalized API uses Classic Huber / Surface L2; omit method")
    kwargs["method"] = "huber"
    if surface_mask is not None:
        kwargs.setdefault("i_min", min(DEFAULT_MIN_ITERATIONS, int(kwargs.get("max_iterations", 300))))
    info = configure_solver(
        solver, alpha, element_size_um, surface_mask, reference_element_size_um,
        exponent, surface_area_m2=surface_area_m2, surface_areas=surface_areas)
    history = solver.solve_regularized(alpha=info["internal_alpha"], **kwargs)
    solver.regularisation_parameters.update(info)
    return history


def _huber_irls(values: np.ndarray, active: np.ndarray, k: float = 1.345) -> np.ndarray:
    weights = np.ones(len(values), dtype=float)
    median = float(np.median(values[active])) if np.any(active) else 0.0
    high = (values > k * median) & active
    if median > 0:
        weights[high] = k * median / values[high]
    elif np.any(high):
        weights[high] = 1e-10
    weights[active] = np.maximum(weights[active], 1e-10)
    return weights


def install_volume_density_huber(solver, nodal_volumes: np.ndarray,
                                 reference_volume_m3: float):
    """Install volume-weighted Classic Huber regularization on ``solver``."""
    volumes = np.asarray(nodal_volumes, dtype=float)
    volume_reference = float(reference_volume_m3)
    data_weights = volumes / volume_reference
    force_weights = volume_reference / volumes

    def update_weights(self, method: str):
        active = self.mesh.movable & self.mesh.regularisation_mask
        density = np.linalg.norm(self.mesh.forces, axis=1) / volumes
        robust = _huber_irls(density, active) if method == "huber" else np.ones(len(volumes))
        self.localweight[:] = robust * force_weights
        self.localweight[~self.mesh.regularisation_mask] = 0.0

    _install_weighted_objective(solver, data_weights, update_weights)
    solver.physical_regularization = {
        "mode": "volume_density_huber",
        "reference_volume_m3": volume_reference,
    }
    return solver


def install_surface_traction_l2(solver, nodal_volumes: np.ndarray,
                                surface_areas: np.ndarray,
                                reference_volume_m3: float,
                                reference_area_m2: float,
                                bulk_factor: float | None = None):
    """Install area-weighted Surface and volume-weighted Bulk regularization.

    If ``bulk_factor`` is omitted it is derived as ``A_ref / V_ref = 1/d``.
    Together with :func:`surface_internal_alpha`, this makes the effective bulk
    coefficient identical to the Classic internal alpha.
    """
    volumes = np.asarray(nodal_volumes, dtype=float)
    areas = np.asarray(surface_areas, dtype=float)
    volume_reference = float(reference_volume_m3)
    area_reference = float(reference_area_m2)
    surface = areas > 0
    if volumes.shape != areas.shape or not np.any(surface):
        raise ValueError("volumes and areas must match and contain surface nodes")
    thickness = volume_reference / area_reference
    if bulk_factor is None:
        bulk_factor = 1.0 / thickness
    bulk_factor = float(bulk_factor)
    data_weights = volumes / volume_reference
    bulk_weights = volume_reference / volumes
    surface_weights = np.zeros(len(areas), dtype=float)
    surface_weights[surface] = area_reference / areas[surface]

    def update_weights(self, method: str):
        self.localweight[:] = bulk_factor * bulk_weights
        self.localweight[surface] = surface_weights[surface]
        self.localweight[~self.mesh.regularisation_mask] = 0.0

    _install_weighted_objective(solver, data_weights, update_weights)
    solver.physical_regularization = {
        "mode": "surface_traction_l2",
        "reference_volume_m3": volume_reference,
        "reference_area_m2": area_reference,
        "surface_thickness_m": thickness,
        "bulk_factor_per_m": bulk_factor,
    }
    return solver


def _install_weighted_objective(solver, data_weights: np.ndarray, update_function):
    data_weights = np.asarray(data_weights, dtype=float)
    # number_nodes is a runtime cache, still zero immediately after deserialization.
    if data_weights.shape != (len(solver.mesh.nodes),):
        raise ValueError("data_weights must contain one value per solver node")

    def compute_a_and_b(self, alpha: float):
        coefficient = np.repeat(self.localweight * alpha, 3)
        KA = self.K_glo.multiply(coefficient[None, :])
        target = self.mesh.displacements_target_mask & self.mesh.movable
        q = data_weights * target
        self.I = ssp.diags(np.repeat(q, 3), format="csr")
        self.KAK = KA @ self.K_glo
        self.A = self.I + self.KAK
        self.b = (KA @ self.mesh.forces.ravel()).reshape(self.mesh.forces.shape)
        residual = self.mesh.displacements_target - self.mesh.displacements
        residual[~target] = 0.0
        self.b += q[:, None] * residual

    def record_status(self, relrec: list, alpha: float, relrecname: str = None):
        target = self.mesh.displacements_target_mask & self.mesh.movable
        residual = self.mesh.displacements_target - self.mesh.displacements
        residual[~target] = 0.0
        displacement_term = float(np.sum(data_weights[:, None] * residual**2))
        forces = np.zeros_like(self.mesh.forces)
        forces[self.mesh.movable] = self.mesh.forces[self.mesh.movable]
        force_term = float(np.sum(
            np.sum(forces**2, axis=1) * self.localweight * self.mesh.movable
        ))
        objective = displacement_term + alpha * force_term
        relrec.append((objective, displacement_term, force_term))
        if relrecname is not None:
            np.savetxt(relrecname, relrec)

    if "_physical_original_methods" not in solver.__dict__:
        solver._physical_original_methods = {
            name: solver.__dict__.get(name) for name in _OVERRIDDEN_METHODS}
    solver._update_local_regularization_weigth = MethodType(update_function, solver)
    solver._compute_regularization_a_and_b = MethodType(compute_a_and_b, solver)
    solver._record_regularization_status = MethodType(record_status, solver)
    solver.physical_data_weights = data_weights
    return solver
