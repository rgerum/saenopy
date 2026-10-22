"""Local normalized-median outlier detection.

Mesh creation tests signed PIV components in a centered 5x5x5 node window
before interpolation. A failure in any component excludes the whole vector.
The original nearest-neighbour vector test remains available to direct callers.
"""
import numpy as np

DEFAULT_OUTLIER_K = 12  # legacy vector mode only
DEFAULT_COMPONENT_WINDOW_SIZE = 5
DEFAULT_OUTLIER_THRESH = 4.0
DEFAULT_OUTLIER_MIN_MULT = 0.5
DEFAULT_OUTLIER_MIN_ABS_UM = 0.3


def _grid_indices(nodes):
    """Map arbitrary row order onto a complete Cartesian grid, including NaN nodes."""
    if nodes.ndim != 2 or nodes.shape[1] != 3 or not np.isfinite(nodes).all():
        raise ValueError("nodes must be a finite (N, 3) Cartesian grid")
    axes_and_indices = [np.unique(nodes[:, j], return_inverse=True) for j in range(3)]
    shape = tuple(len(axis) for axis, _ in axes_and_indices)
    indices = np.column_stack([index for _, index in axes_and_indices])
    if int(np.prod(shape)) != len(nodes):
        raise ValueError("The component filter requires a complete Cartesian node grid; "
                         "keep missing measurements as NaN vectors, not missing nodes")
    flat = np.ravel_multi_index(indices.T, shape)
    if len(np.unique(flat)) != len(nodes):
        raise ValueError("The component filter requires unique Cartesian grid nodes")
    return shape, indices


def _component_neighbourhoods(nodes, u, valid, chunk_size=2048):
    """Yield center IDs and the 124 fixed offsets; absent/invalid vectors are NaN."""
    shape, indices = _grid_indices(nodes)
    radius = DEFAULT_COMPONENT_WINDOW_SIZE // 2
    grid = np.full(tuple(n + 2 * radius for n in shape) + (3,), np.nan)
    ids = np.flatnonzero(valid)
    grid[tuple((indices[ids] + radius).T)] = u[ids]
    offsets = np.stack(np.meshgrid(*([np.arange(-radius, radius + 1)] * 3),
                                  indexing="ij"), axis=-1).reshape(-1, 3)
    offsets = offsets[np.any(offsets != 0, axis=1)]
    for start in range(0, len(ids), chunk_size):
        centers = ids[start:start + chunk_size]
        neighbours = indices[centers, None, :] + radius + offsets[None, :, :]
        yield centers, grid[neighbours[..., 0], neighbours[..., 1], neighbours[..., 2]]


def local_outliers(nodes, u, base_mask, k=DEFAULT_OUTLIER_K,
                   thresh=DEFAULT_OUTLIER_THRESH, min_mult=DEFAULT_OUTLIER_MIN_MULT,
                   min_abs_um=DEFAULT_OUTLIER_MIN_ABS_UM, eps_um=0.3, verbose=False,
                   componentwise=False):
    """Flag local outliers without changing input vectors.

    Component mode uses exactly the index offsets [-2, 2] in each grid axis,
    excluding the center. At edges the window is truncated; invalid vectors are
    ignored. The window never expands across missing data. There is no minimum
    neighbour count beyond requiring at least one finite neighbour to define a
    median. A center with no valid neighbours is left unflagged. All supported
    centers use the same decision limits; very sparse support is less reliable.

    For each signed component c, let m_c be the neighbour median, s_c the median
    of |u_neighbour,c - m_c|, and a_c the median neighbour absolute amplitude.
    Flag the whole vector if, for any c, both conditions hold:
        |u_c - m_c| > thresh * (s_c + eps_um * 1e-6)
        |u_c - m_c| > max(min_abs_um * 1e-6, min_mult * a_c).
    All decisions use the same original field (one simultaneous pass).

    Coordinates and displacements are in metres. ``k`` applies only to the
    unchanged legacy vector mode, which uses the k nearest valid vectors.
    """
    nodes = np.asarray(nodes)
    u = np.asarray(u)
    out = np.zeros(len(nodes), bool)
    if componentwise:
        import warnings
        valid = np.asarray(base_mask, dtype=bool) & np.isfinite(u).all(axis=1)
        if valid.sum() < 2:
            return out
        for ids, neighbours in _component_neighbourhoods(nodes, u, valid):
            count = np.isfinite(neighbours).all(axis=2).sum(axis=1)
            sufficient = count > 0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(neighbours, axis=1)
                scatter = np.nanmedian(np.abs(neighbours - med[:, None, :]), axis=1)
                amplitude = np.nanmedian(np.abs(neighbours), axis=1)
            residual = np.abs(u[ids] - med)
            floor = np.maximum(min_abs_um * 1e-6, min_mult * amplitude)
            # Multiplication defines zero scatter + zero epsilon without division.
            out[ids] = sufficient & np.any(
                (residual > thresh * (scatter + eps_um * 1e-6)) &
                (residual > floor), axis=1)
        support_description = "window=5x5x5"
        count = int(valid.sum())
    else:
        from scipy.spatial import cKDTree
        idx = np.where(base_mask)[0]
        if len(idx) < k + 1:
            return out
        P, U = nodes[idx], u[idx]
        _, nn = cKDTree(P).query(P, k=k + 1)
        nb = nn[:, 1:]
        med = np.median(U[nb], axis=1)
        res = np.linalg.norm(U - med, axis=1)
        nb_dev = np.linalg.norm(U[nb] - med[:, None, :], axis=2)
        scatter = np.median(nb_dev, axis=1)
        norm = res / (scatter + eps_um * 1e-6)
        nb_mag = np.median(np.linalg.norm(U[nb], axis=2), axis=1)
        min_res = np.maximum(min_abs_um * 1e-6, min_mult * nb_mag)
        out[idx[(norm > thresh) & (res > min_res)]] = True
        support_description = f"k={k}"
        count = len(idx)
    if verbose:
        print(f"  [outlier] {support_description} thresh={thresh} min_mult={min_mult} "
              f"min_abs={min_abs_um}um -> flagged {int(out.sum())} / {count} outliers")
    return out


def filter_target_outliers(nodes, u_target, k=DEFAULT_OUTLIER_K,
                           thresh=DEFAULT_OUTLIER_THRESH, min_mult=DEFAULT_OUTLIER_MIN_MULT,
                           min_abs_um=DEFAULT_OUTLIER_MIN_ABS_UM, verbose=False, *,
                           componentwise=False, eps_um=0.3):
    """Return a copy with whole outlier vectors set to NaN, plus the outlier mask.

    Despite the historical name, mesh creation calls this on reference-corrected
    PIV vectors. Retained values and existing invalid entries are not replaced.
    """
    u_target = np.asarray(u_target, dtype=float)
    valid = np.isfinite(u_target).all(axis=1)
    outlier = local_outliers(nodes, u_target, valid, k=k, thresh=thresh,
                             min_mult=min_mult, min_abs_um=min_abs_um, verbose=verbose,
                             componentwise=componentwise, eps_um=eps_um)
    cleaned = u_target.copy()
    cleaned[outlier] = np.nan
    return cleaned, outlier
