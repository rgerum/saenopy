"""Segmentation and surface-node masks for cell-surface regularization.

Current fits use :func:`saenopy.reconstruction.fit_result` or
:func:`saenopy.physical_regularization.solve_regularized`. Their geometric
conversion is implemented in `physical_regularization.reference_measures`::

    N_reg = N_bulk + N_surface             # active regularized nodes
    V_ref = sum(V_i over active nodes) / N_reg
    A_ref = measured_cell_area / N_surface
    d = V_ref / A_ref                      # numerical length in metres
    alpha_h = alpha * (h_ref / h)**5
    alpha_surface = alpha_h * d
    bulk_factor = 1 / d

The cell area comes from the segmented interface, not N_surface * h**2.
There is no fitted bridge reference. The two objectives remain different;
equal optimal alpha or equal reconstructed contractility is not guaranteed.

"""
from __future__ import annotations
import numpy as np

DEFAULT_MIN_ITERATIONS = 60


# ---------------------------------------------------------------------------
# 1. segmentation: cell body + surface shell (saenopy coordinate convention)
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLD_METHOD = "li"     # "li" | "otsu" | "yen"
DEFAULT_THRESHOLD_FACTOR = 0.6


def auto_threshold(image, method=DEFAULT_THRESHOLD_METHOD,
                   factor=DEFAULT_THRESHOLD_FACTOR, smooth=2.0):
    """Automatic intensity threshold = ``factor * <method>(smoothed image)``.

    The selected method is fixed, not chosen automatically. Check the resulting
    segmentation against the cell image; no factor is suitable for every stain.
    """
    from scipy.ndimage import gaussian_filter
    from skimage.filters import threshold_otsu, threshold_yen, threshold_li
    fn = {"otsu": threshold_otsu, "yen": threshold_yen, "li": threshold_li}[method]
    sm = gaussian_filter(np.asarray(image).astype(np.float32), sigma=smooth, truncate=2.0)
    return float(fn(sm)) * float(factor)


def segment_cell(image, voxel_size, threshold="auto", smooth=2.0, close_radius=1,
                 largest_component=True, fill=True,
                 threshold_method=DEFAULT_THRESHOLD_METHOD,
                 threshold_factor=DEFAULT_THRESHOLD_FACTOR):
    """Segment a filled cell body and its surface shell from a 3D stack.

    Parameters
    ----------
    image : ndarray (Y, X, Z)
        Single-channel fluorescence stack of the cell label/stain.
    voxel_size : (du, dv, dw)
        Physical voxel size in micrometres, i.e. ``stack.voxel_size`` (X, Y, Z).
    threshold : float or "auto"
        Absolute intensity threshold, or ``"auto"`` to use
        ``threshold_factor * <threshold_method>(image)`` (see :func:`auto_threshold`).
    threshold_method, threshold_factor : used when ``threshold == "auto"``.
    smooth, close_radius : pre-smoothing sigma / morphological closing radius
        (closing is expensive on big stacks; 1 is usually enough, raise it to 2-3
        for hollow objects whose bright rim needs bridging).
    largest_component : keep only the biggest connected component (rejects debris).
    fill : fill interior holes so the body is solid.

    Returns
    -------
    body : ndarray(bool) (Y, X, Z)   filled cell/object volume.
    shell_xyz : ndarray (N, 3)       surface voxels, physical, CENTERED, in METRES,
                                     columns (Y, X, Z) -- the same frame as the
                                     saenopy solver mesh nodes.
    """
    from scipy.ndimage import gaussian_filter, binary_fill_holes, label
    from skimage.morphology import erosion, ball, closing

    image = np.asarray(image)
    du, dv, dw = [float(v) for v in voxel_size]
    if isinstance(threshold, str):
        threshold = auto_threshold(image, method=threshold_method,
                                   factor=threshold_factor, smooth=smooth)
    imf = gaussian_filter(image.astype(np.float32), sigma=smooth, truncate=2.0)
    vol = imf > threshold
    if close_radius and close_radius > 0:
        vol = closing(vol, ball(int(close_radius)))
    if largest_component:
        lab, n = label(vol)
        if n > 1:
            sizes = np.bincount(lab.ravel()); sizes[0] = 0
            vol = lab == sizes.argmax()
    if fill:
        vol = binary_fill_holes(vol)
    shell = vol & ~erosion(vol)

    # Only allocate coordinates for the shell, not six full-volume arrays.
    shell_xyz = (np.argwhere(shell) - (np.asarray(image.shape) - 1) / 2)
    shell_xyz = shell_xyz * np.array([dv, du, dw]) * 1e-6
    return vol, shell_xyz


# ---------------------------------------------------------------------------
# 2. per-node masks on a solver mesh
# ---------------------------------------------------------------------------
def surface_node_mask(nodes, shell_xyz, element_size_um, capture_radius_um=None,
                      dilate_layers=0):
    """Boolean per-node mask: mesh nodes within ``capture_radius`` of the cell
    surface (default = element_size / 2, i.e. ~one node layer).

    ``dilate_layers`` grows the mask by that many node shells (binary dilation on the
    mesh point cloud) -- useful for THIN single cells whose segmentation captures only a
    handful of surface nodes, so the traction has more nodes to spread over instead of
    concentrating on a few. See :func:`dilate_surface_mask`.
    """
    from scipy.spatial import cKDTree
    r = (element_size_um / 2.0 if capture_radius_um is None else capture_radius_um)
    dist, _ = cKDTree(shell_xyz).query(np.asarray(nodes), k=1)
    mask = dist < r * 1e-6
    if dilate_layers:
        mask = dilate_surface_mask(nodes, mask, element_size_um, layers=dilate_layers)
    return mask


def dilate_surface_mask(nodes, mask, element_size_um, layers=1):
    """Grow a surface-node mask by ``layers`` node shells (binary dilation on the mesh
    point cloud): add every mesh node within ~one element spacing of an existing surface
    node. For thin cells that otherwise have too few surface nodes to carry the traction.
    """
    from scipy.spatial import cKDTree
    mask = np.asarray(mask, bool).copy()
    if layers <= 0 or not mask.any():
        return mask
    nodes = np.asarray(nodes)
    tree = cKDTree(nodes)
    r = element_size_um * 1e-6 * 1.25          # ~one node spacing (diagonal-safe)
    for _ in range(int(layers)):
        grow = np.zeros(len(nodes), bool)
        for lst in tree.query_ball_point(nodes[mask], r=r):
            grow[lst] = True
        mask |= grow
    return mask


def distance_to_surface(nodes, shell_xyz):
    """Per-node distance (metres) to the nearest cell-surface voxel."""
    from scipy.spatial import cKDTree
    dist, _ = cKDTree(shell_xyz).query(np.asarray(nodes), k=1)
    return dist


def inside_cell_mask(nodes, body_vol, voxel_size, image_shape):
    """Boolean per-node mask: mesh nodes that fall INSIDE the filled cell body.

    Uses the same centred coordinate convention as :func:`segment_cell`.
    """
    du, dv, dw = [float(v) for v in voxel_size]
    Ny, Nx, Nz = [int(v) for v in image_shape]
    nodes = np.asarray(nodes)
    iy = np.round(nodes[:, 0] * 1e6 / dv + (Ny - 1) / 2).astype(int)
    ix = np.round(nodes[:, 1] * 1e6 / du + (Nx - 1) / 2).astype(int)
    iz = np.round(nodes[:, 2] * 1e6 / dw + (Nz - 1) / 2).astype(int)
    ok = (iy >= 0) & (iy < Ny) & (ix >= 0) & (ix < Nx) & (iz >= 0) & (iz < Nz)
    inside = np.zeros(len(nodes), bool)
    inside[ok] = np.asarray(body_vol)[iy[ok], ix[ok], iz[ok]]
    return inside


# ---------------------------------------------------------------------------
# 3. apply to a solver and reconstruct
# ---------------------------------------------------------------------------
def set_surface_regularization(solver, surface_mask):
    """Attach the surface node mask to the solver (as ``mesh.cell_boundary_mask``)."""
    surface_mask = np.asarray(surface_mask, dtype=bool)
    assert surface_mask.shape[0] == solver.mesh.nodes.shape[0], \
        "surface_mask length must equal the number of mesh nodes"
    solver.mesh.cell_boundary_mask = surface_mask


def drop_targets(solver, drop_mask):
    """Remove deformation-fit targets at ``drop_mask`` nodes (e.g. inside the
    cell, where the PIV displacement is unreliable). Returns #dropped."""
    drop_mask = np.asarray(drop_mask, dtype=bool)
    before = int(solver.mesh.displacements_target_mask.sum())
    solver.mesh.displacements_target_mask[drop_mask] = False
    return before - int(solver.mesh.displacements_target_mask.sum())
