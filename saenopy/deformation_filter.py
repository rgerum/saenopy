"""
saenopy.deformation_filter
==========================
Robust, scale-adaptive outlier removal for the interpolated target-deformation
field, applied once at the *interpolate mesh* step (before the force fit). A single
huge PIV spike would otherwise be fitted by a large, wrong, localized force; removing
it up-front cleans the reconstruction for BOTH the classic and the surface-restricted
regularization and speeds up convergence.

The test is a 3D Westerweel-style local normalized-median (median/MAD-like) test:
each valid node is compared to the MEDIAN of its ``k`` nearest neighbours (a robust
local expectation), the deviation normalised by the local robust scatter. Because it
is median-based, a single extreme spike cannot mask itself, and because the
normalisation is scale-free (and the absolute floor is tied to the field's own median
displacement) the same settings work whether a cell pulls ~1 um or ~30 um.

Ported from the surface-regularization experiment (``surfreg_filters.local_outliers``),
where it removed every deformation spike on the organoid data on its own.
"""
import numpy as np

# defaults validated on the organoid + single-cell data
DEFAULT_OUTLIER_K = 12          # nearest neighbours for the local statistic
DEFAULT_OUTLIER_THRESH = 4.0    # normalized-median threshold (higher = more permissive)
DEFAULT_OUTLIER_MIN_MULT = 0.5  # LOCAL floor = min_mult * neighbour-median |u| (per node)
DEFAULT_OUTLIER_MIN_ABS_UM = 0.3  # hard lower bound of that floor, in micrometres


def local_outliers(nodes, u, base_mask, k=DEFAULT_OUTLIER_K, thresh=DEFAULT_OUTLIER_THRESH,
                   min_mult=DEFAULT_OUTLIER_MIN_MULT, min_abs_um=DEFAULT_OUTLIER_MIN_ABS_UM,
                   eps_um=0.3, verbose=False):
    """Robust, scale-adaptive 3D normalized-median outlier test (Westerweel-style).

    A node is flagged when its displacement deviates from the MEDIAN of its ``k``
    nearest neighbours by more than ``thresh`` times the local robust scatter (a
    median-of-deviations, MAD-like) AND the deviation exceeds a LOCAL floor. The floor
    is per-node ``max(min_abs_um, min_mult * neighbour-median|u|)`` -- crucially LOCAL,
    not tied to the whole-field median. That is what makes the test work on a
    heterogeneous single-cell field: in the quiet gel the floor is tiny so isolated PIV
    spikes are caught, while a coherent high-deformation region near the cell edge (where
    a node AND its neighbours are large, so the deviation is small) is left untouched. An
    earlier version used a global ``min_mult * field_median`` floor (~5 um here), which
    swamped ``thresh`` and flagged nothing on thin cells.

    Parameters
    ----------
    nodes : ndarray (N, 3)   node coordinates in metres.
    u : ndarray (N, 3)       displacement at each node in metres.
    base_mask : ndarray (N,) bool   nodes to consider (valid / non-nan targets).
    k : int                  nearest neighbours for the local median/scatter.
    thresh : float           flag if the normalized deviation exceeds this (lower = more sensitive).
    min_mult : float         local floor = min_mult * neighbour-median |u| (relative protection
                             of coherent large regions).
    min_abs_um : float       hard lower bound of the floor, in micrometres.
    eps_um : float           small regulariser (um) on the scatter to avoid division by zero.

    Returns
    -------
    ndarray (N,) bool        full-length mask of nodes flagged as outliers (to remove).
    """
    from scipy.spatial import cKDTree

    nodes = np.asarray(nodes)
    u = np.asarray(u)
    idx = np.where(base_mask)[0]
    out = np.zeros(len(nodes), bool)
    if len(idx) < k + 1:
        return out
    P = nodes[idx]
    U = u[idx]
    _, nn = cKDTree(P).query(P, k=k + 1)                        # col 0 == self
    nb = nn[:, 1:]
    med = np.median(U[nb], axis=1)                              # (n,3) local expectation
    res = np.linalg.norm(U - med, axis=1)                       # node deviation
    nb_dev = np.linalg.norm(U[nb] - med[:, None, :], axis=2)    # (n,k) neighbour devs
    scatter = np.median(nb_dev, axis=1)                         # robust local scatter
    norm = res / (scatter + eps_um * 1e-6)
    nb_mag = np.median(np.linalg.norm(U[nb], axis=2), axis=1)   # (n,) LOCAL magnitude scale
    min_res = np.maximum(min_abs_um * 1e-6, min_mult * nb_mag)  # per-node LOCAL floor
    flag = (norm > thresh) & (res > min_res)
    out[idx[flag]] = True
    if verbose:
        print(f"  [outlier] k={k} thresh={thresh} min_mult={min_mult} "
              f"min_abs={min_abs_um}um -> flagged {int(flag.sum())} / {len(idx)} outliers")
    return out


def filter_target_outliers(nodes, u_target, k=DEFAULT_OUTLIER_K, thresh=DEFAULT_OUTLIER_THRESH,
                           min_mult=DEFAULT_OUTLIER_MIN_MULT, min_abs_um=DEFAULT_OUTLIER_MIN_ABS_UM,
                           verbose=False):
    """Return a copy of ``u_target`` with outlier rows set to NaN.

    Operates on the currently valid (finite) target nodes; flagged nodes are set to
    NaN so the existing ``displacements_target_mask`` logic drops them from the fit.
    Also returns the boolean outlier mask.
    """
    u_target = np.asarray(u_target, dtype=float)
    valid = np.all(np.isfinite(u_target), axis=1)
    outlier = local_outliers(nodes, np.nan_to_num(u_target), valid, k=k, thresh=thresh,
                             min_mult=min_mult, min_abs_um=min_abs_um, verbose=verbose)
    cleaned = u_target.copy()
    cleaned[outlier] = np.nan
    return cleaned, outlier
