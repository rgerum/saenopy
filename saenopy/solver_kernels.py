"""
Numba kernels for the per-tetrahedron force and stiffness computation of the solver.

The tetrahedra are processed in parallel. For each tetrahedron the deformed beams, the local forces (4x3) and the
local stiffness block (4x4x3x3) are computed in one pass over the beams, without large temporary arrays.
"""
import numpy as np
from numba import njit, prange

# the 10 unique (m, r) corner pairs and the 6 unique (i, l) direction pairs of the symmetric stiffness block
_MR = np.array([(m, r) for m in range(4) for r in range(m, 4)], dtype=np.int64)
_IL = np.array([(i, l) for i in range(3) for l in range(i, 3)], dtype=np.int64)


@njit(parallel=True, cache=True)
def compute_s_bar(displacements, tetrahedra, Phi, beams, t0, t1, s_bar, s_norm):
    """
    For the tetrahedra t0:t1 compute the deformed beams s'_tib = F_tij * s_jb and their lengths |s'_tb|.

    F_tij = d_ij + u_tmi * Phi_tmj
    """
    n_b = beams.shape[0]
    for tt in prange(t1 - t0):
        t = t0 + tt
        F = np.eye(3)
        for m in range(4):
            c = tetrahedra[t, m]
            for i in range(3):
                u = displacements[c, i]
                for j in range(3):
                    F[i, j] += u * Phi[t, m, j]
        for b in range(n_b):
            norm = 0.0
            for i in range(3):
                v = F[i, 0] * beams[b, 0] + F[i, 1] * beams[b, 1] + F[i, 2] * beams[b, 2]
                s_bar[tt, i, b] = v
                norm += v * v
            s_norm[tt, b] = np.sqrt(norm)


@njit(parallel=True, cache=True)
def compute_f_and_K(Phi, beams, t0, t1, s_bar, s_norm, epsbar, epsbarbar, V_over_Nb, f_out, K_out):
    """
    Compute the local forces and stiffness blocks of the tetrahedra t0:t1.

    dEdsbar_tb    = - eps'_tb / s_tb * V_t / N_b
    dEdsbarbar_tb = (s_tb * eps''_tb - eps'_tb) / s_tb^3 * V_t / N_b
    f_tmi   = s*_tmb * s'_tib * dEdsbar_tb
    K_tmril = s*_tmb * s*_trb * 0.5 * (dEdsbarbar_tb * s'_tib * s'_tlb - delta_il * dEdsbar_tb)
    with s*_tmb = Phi_tmj * s_jb
    """
    n_b = beams.shape[0]
    MR = _MR
    IL = _IL
    for tt in prange(t1 - t0):
        t = t0 + tt
        V = V_over_Nb[t]
        f = np.zeros((4, 3))
        K = np.zeros((10, 6))
        ss = np.zeros(4)
        mr = np.zeros(10)
        il = np.zeros(6)
        for b in range(n_b):
            s = s_norm[tt, b]
            d1 = -(epsbar[tt, b] / s) * V
            d2 = ((s * epsbarbar[tt, b] - epsbar[tt, b]) / (s * s * s)) * V

            for m in range(4):
                ss[m] = Phi[t, m, 0] * beams[b, 0] + Phi[t, m, 1] * beams[b, 1] + Phi[t, m, 2] * beams[b, 2]

            sb0 = s_bar[tt, 0, b]
            sb1 = s_bar[tt, 1, b]
            sb2 = s_bar[tt, 2, b]
            for m in range(4):
                w = ss[m] * d1
                f[m, 0] += w * sb0
                f[m, 1] += w * sb1
                f[m, 2] += w * sb2

            for k in range(10):
                mr[k] = ss[MR[k, 0]] * ss[MR[k, 1]]
            h = 0.5 * d2
            hd = 0.5 * d1
            il[0] = h * sb0 * sb0 - hd
            il[1] = h * sb0 * sb1
            il[2] = h * sb0 * sb2
            il[3] = h * sb1 * sb1 - hd
            il[4] = h * sb1 * sb2
            il[5] = h * sb2 * sb2 - hd
            for k in range(10):
                w = mr[k]
                for q in range(6):
                    K[k, q] += w * il[q]

        for m in range(4):
            for i in range(3):
                f_out[t, m, i] = f[m, i]
        for k in range(10):
            m = MR[k, 0]
            r = MR[k, 1]
            for q in range(6):
                i = IL[q, 0]
                l = IL[q, 1]
                v = K[k, q]
                K_out[t, m, r, i, l] = v
                K_out[t, m, r, l, i] = v
                K_out[t, r, m, i, l] = v
                K_out[t, r, m, l, i] = v


@njit(cache=True)
def _bsr3_row(data, indices, indptr, x, row, out):
    """ out = (K @ x)[3 * row: 3 * row + 3] for a block sparse matrix K with 3x3 blocks """
    y0 = 0.0
    y1 = 0.0
    y2 = 0.0
    for k in range(indptr[row], indptr[row + 1]):
        c = indices[k] * 3
        x0 = x[c]
        x1 = x[c + 1]
        x2 = x[c + 2]
        y0 += data[k, 0, 0] * x0 + data[k, 0, 1] * x1 + data[k, 0, 2] * x2
        y1 += data[k, 1, 0] * x0 + data[k, 1, 1] * x1 + data[k, 1, 2] * x2
        y2 += data[k, 2, 0] * x0 + data[k, 2, 1] * x1 + data[k, 2, 2] * x2
    out[0] = y0
    out[1] = y1
    out[2] = y2


@njit(parallel=True, cache=True)
def bsr3_matvec(data, indices, indptr, x):
    """ y = K @ x for a block sparse matrix K with 3x3 blocks, parallelized over the block rows """
    n = indptr.shape[0] - 1
    y = np.empty(n * 3)
    for row in prange(n):
        _bsr3_row(data, indices, indptr, x, row, y[row * 3:row * 3 + 3])
    return y


@njit(parallel=True, cache=True)
def regularization_matvec(data, indices, indptr, weight, I_diag, x):
    """ y = diag(I) @ x + K @ (diag(W) @ (K @ x)) for a block sparse matrix K with 3x3 blocks """
    n = indptr.shape[0] - 1
    tmp = np.empty(n * 3)
    for row in prange(n):
        _bsr3_row(data, indices, indptr, x, row, tmp[row * 3:row * 3 + 3])
        for i in range(row * 3, row * 3 + 3):
            tmp[i] *= weight[i]
    y = np.empty(n * 3)
    for row in prange(n):
        _bsr3_row(data, indices, indptr, tmp, row, y[row * 3:row * 3 + 3])
        for i in range(row * 3, row * 3 + 3):
            y[i] += I_diag[i] * x[i]
    return y
