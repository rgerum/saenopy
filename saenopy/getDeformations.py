# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:41:50 2020

@author: david
"""

import numpy as np
from openpiv.pyprocess3D import extended_search_area_piv3D
from scipy import interpolate


def get_displacements_from_stacks(stack_relaxed, stack_deformed, win_um, elementsize, signoise_filter, drift_correction):
    fac_overlap = 1 - (elementsize/win_um)
    voxel_size1 = stack_deformed.voxel_size
    voxel_size2 = stack_relaxed.voxel_size

    np.testing.assert_equal(voxel_size1, voxel_size2, f"The two stacks do not have the same voxel size. {voxel_size1}, {voxel_size2}")

    np.testing.assert_equal(stack_deformed.shape, stack_relaxed.shape, f"The two stacks do not have the same voxel count. {stack_deformed.shape}, {stack_relaxed.shape}")

    # mean over the rgb channels
    stack_deformed = np.mean(np.array(stack_deformed), axis=2)
    stack_relaxed = np.mean(np.array(stack_relaxed), axis=2)
    M = getDisplacementsFromStacks_old(stack_deformed, stack_relaxed, voxel_size1,
                                        win_um=win_um,
                                        fac_overlap=fac_overlap,
                                        signoise_filter=signoise_filter,
                                        drift_correction=drift_correction,
                                        return_mesh=True)
    # center
    M.R = (M.R - np.min(M.R, axis=0)) - (np.max(M.R, axis=0) - np.min(M.R, axis=0)) / 2
    return M


def sig2noise_filtering( u, v, sig2noise, w=None, threshold = 1.3):
    """
    As integrted into OpenPiv Jun 19, 2020.
    Since OpenPIV changed this function lateron, we use this version
    to replace outliers with np.nan dependend on the signal2noise ratio here
    """

    ind = sig2noise < threshold

    u[ind] = np.nan
    v[ind] = np.nan
    if isinstance(w, np.ndarray):
        w[ind] = np.nan
        return u, v, w, ind

    return u, v, ind


def replace_outliers( u, v, w=None, method='localmean', max_iter=5, tol=1e-3, kernel_size=1):
    """
    As integrted into OpenPiv Jun 19, 2020.
    Since OpenPIV changed several functions lateron, we use this version
    to replace outliers with np.nan dependend on the signal2noise ratio here

    Replace invalid vectors in an velocity field using an iterative image inpainting algorithm.

    The algorithm is the following:

    1) For each element in the arrays of the ``u`` and ``v`` components, replace it by a weighted average
       of the neighbouring elements which are not invalid themselves. The weights depends
       of the method type. If ``method=localmean`` weight are equal to 1/( (2*kernel_size+1)**2 -1 )

    2) Several iterations are needed if there are adjacent invalid elements.
       If this is the case, inforation is "spread" from the edges of the missing
       regions iteratively, until the variation is below a certain threshold.

    Parameters
    ----------

    u : 2d or 3d np.ndarray
        the u velocity component field

    v : 2d or 3d  np.ndarray
        the v velocity component field
    w : 2d or 3d  np.ndarray
        the w velocity component field

    max_iter : int
        the number of iterations
    fil
    kernel_size : int
        the size of the kernel, default is 1

    method : str
        the type of kernel used for repairing missing vectors

    Returns
    -------
    uf : 2d or 3d np.ndarray
        the smoothed u velocity component field, where invalid vectors have been replaced

    vf : 2d or 3d np.ndarray
        the smoothed v velocity component field, where invalid vectors have been replaced
    wf : 2d or 3d np.ndarray
        the smoothed w velocity component field, where invalid vectors have been replaced

    """
    uf = replace_nans_py(u, method=method, max_iter=max_iter, tol=tol, kernel_size=kernel_size)
    vf = replace_nans_py(v, method=method, max_iter=max_iter, tol=tol, kernel_size=kernel_size)

    if isinstance(w, np.ndarray):
        wf =  replace_nans_py(w, method=method, max_iter=max_iter, tol=tol, kernel_size=kernel_size)
        return uf, vf, wf

    return uf, vf

def get_dist(kernel,kernel_size):
    """
    As integrted into OpenPiv Jun 19, 2020.
    Since OpenPIV changed several functions lateron, we use this version
    """
    # generates a map of distances to the center of the kernel. This is later used to generate disk-shaped kernels and
    # fill in distance based weights

    if len(kernel.shape) == 2:
        # x and y coordinates for each points
        xs, ys = np.indices(kernel.shape)
        # maximal distance form center - distance to center (of each point)
        dist = np.sqrt((ys - kernel_size) ** 2 + (xs - kernel_size) ** 2)
        dist_inv = np.sqrt(2) * kernel_size - dist

    if len(kernel.shape) == 3:
        xs, ys, zs = np.indices(kernel.shape)
        dist = np.sqrt((ys - kernel_size) ** 2 + (xs - kernel_size) ** 2 + (zs - kernel_size) ** 2)
        dist_inv = np.sqrt(3) * kernel_size - dist

    return dist, dist_inv


def replace_nans_py(array, max_iter, tol, kernel_size = 2, method = 'disk'):
    """
    As integrted into OpenPiv Jun 19, 2020.
    Since OpenPIV changed several functions lateron, we use this version
    to replace outliers with np.nan dependend on the signal2noise ratio here


    Replace NaN elements in an array using an iterative image inpainting algorithm.
      The algorithm is the following:
      1) For each element in the input array, replace it by a weighted average
         of the neighbouring elements which are not NaN themselves. The weights
         depend on the method type. See Methods below.
      2) Several iterations are needed if there are adjacent NaN elements.
         If this is the case, information is "spread" from the edges of the missing
         regions iteratively, until the variation is below a certain threshold.
      Methods:
      localmean - A square kernel where all elements have the same value,
                  weights are equal to n/( (2*kernel_size+1)**2 -1 ),
                  where n is the number of non-NaN elements.
      disk - A circular kernel where all elements have the same value,
             kernel is calculated by::
                 if ((S-i)**2 + (S-j)**2)**0.5 <= S:
                     kernel[i,j] = 1.0
                 else:
                     kernel[i,j] = 0.0
             where S is the kernel radius.
      distance - A circular inverse distance kernel where elements are
                 weighted proportional to their distance away from the
                 center of the kernel, elements farther away have less
                 weight. Elements outside the specified radius are set
                 to 0.0 as in 'disk', the remaining of the weights are
                 calculated as::
                     maxDist = ((S)**2 + (S)**2)**0.5
                     kernel[i,j] = -1*(((S-i)**2 + (S-j)**2)**0.5 - maxDist)
                 where S is the kernel radius.
      Parameters
      ----------
      array : 2d or 3d np.ndarray
          an array containing NaN elements that have to be replaced
      max_iter : int
          the number of iterations
      tol : float
          On each iteration check if the mean square difference between
          values of replaced elements is below a certain tolerance `tol`
      kernel_size : int
          the size of the kernel, default is 1
      method : str
          the method used to replace invalid values. Valid options are
          `localmean`, `disk`, and `distance`.
      Returns
      -------
      filled : 2d or 3d np.ndarray
          a copy of the input array, where NaN elements have been replaced.
      """

    DTYPEf = float
    DTYPEi = int

    filled = array.copy()
    n_dim = len(array.shape)

    # generating the kernel
    kernel = np.zeros([2 * kernel_size + 1] * len(array.shape), dtype=int)
    if method == 'localmean':
        kernel += 1
    elif method == 'disk':
        dist, dist_inv = get_dist(kernel, kernel_size)
        kernel[dist <= kernel_size] = 1
    elif method == 'distance':
        dist, dist_inv = get_dist(kernel, kernel_size)
        kernel[dist <= kernel_size] = dist_inv[dist <= kernel_size]
    else:
        raise ValueError('method not valid. Should be one of `localmean`, `disk` or `distance`.')

    # list of kernel array indices
    kernel_indices = np.indices(kernel.shape)
    kernel_indices = np.reshape(kernel_indices, (n_dim, (2 * kernel_size + 1) ** n_dim), order="C").T

    # indices where array is NaN
    nan_indices = np.array(np.nonzero(np.isnan(array))).T.astype(DTYPEi)

    # number of NaN elements
    n_nans = len(nan_indices)

    # arrays which contain replaced values to check for convergence
    replaced_new = np.zeros(n_nans, dtype=DTYPEf)
    replaced_old = np.zeros(n_nans, dtype=DTYPEf)

    # make several passes
    # until we reach convergence
    for it in range(max_iter):
        # note: identifying new nan indices and looping other the new indices would give slightly different result

        # for each NaN element
        for k in range(n_nans):
            ind = nan_indices[k] #2 or 3 indices indicating the position of a nan element
            # init to 0.0
            replaced_new[k] = 0.0
            n = 0.0

            # generating a list of indices of the convolution window in the array
            slice_indices = np.array(np.meshgrid(*[range(i-kernel_size,i+kernel_size+1) for i in ind]))
            slice_indices = np.reshape(slice_indices,( n_dim, (2 * kernel_size + 1) ** n_dim), order="C").T

            # loop over the kernel
            for s_index, k_index in zip(slice_indices, kernel_indices):
                s_index = tuple(s_index) # this is necessary for numpy array indexing
                k_index = tuple(k_index)

                # skip if we are outside of array boundaries, if the array element is nan or if the kernel element is zero
                if all([s >= 0 and s < bound for s, bound  in zip(s_index, filled.shape)]):
                    if not np.isnan(filled[s_index]) and kernel[k_index] != 0:
                    # convolve kernel with original array
                        replaced_new[k] = replaced_new[k] + filled[s_index] * kernel[k_index]
                        n = n + kernel[k_index]

                    # divide value by effective number of added elements
            if n > 0:
                replaced_new[k] = replaced_new[k] / n
            else:
                replaced_new[k] = np.nan

        # bulk replace all new values in array
        for k in range(n_nans):
            filled[tuple(nan_indices[k])] = replaced_new[k]

        # elements is below a certain tolerance
        if np.mean((replaced_new - replaced_old) ** 2) < tol:
            break
        else:
                replaced_old = replaced_new
    return filled


# Full 3D Deformation analysis
def getDisplacementsFromStacks_old(stack_deformed, stack_relaxed, voxel_size, win_um=12, fac_overlap=0.6,
                               signoise_filter=1.3, drift_correction=True, return_mesh=False):
    from saenopy.multigridHelper import createBoxMesh
    from saenopy import Solver

    # set properties
    voxel_size = np.array(voxel_size)
    window_size = (win_um / voxel_size).astype(int)
    overlap = ((fac_overlap * win_um) / voxel_size).astype(int)
    du, dv, dw = voxel_size
    print("Calculate Deformations", stack_relaxed.shape, window_size, overlap)

    # calculate deformations
    u, v, w, sig2noise = extended_search_area_piv3D(stack_relaxed, stack_deformed,
                                                    window_size=window_size,
                                                    overlap=overlap,
                                                    dt=(1 / du, 1 / dv, 1 / dw),
                                                    search_area_size=window_size,
                                                    subpixel_method='gaussian',
                                                    sig2noise_method='peak2peak',
                                                    width=2,
                                                    nfftx=None, nffty=None)

    # correcting stage drift between the field of views
    if drift_correction:
        u -= np.nanmean(u)
        v -= np.nanmean(v)
        w -= np.nanmean(w)

    # filter deformations
    uf, vf, wf, mask = sig2noise_filtering(u, v,sig2noise, w=w, threshold=signoise_filter)
    uf, vf, wf = replace_outliers(uf, vf, wf, max_iter=1, tol=100, kernel_size=2, method='disk')

    # get coordinates (by multiplication with the ratio of image dimension and deformation grid)
    y, x, z = np.indices(u.shape)
    y, x, z = (y * stack_deformed.shape[0] * dv / u.shape[0],
               x * stack_deformed.shape[1] * du / u.shape[1],
               z * stack_deformed.shape[2] * dw / u.shape[2])

    # create a box mesh - convert to meters for saenopy conversion
    R, T = createBoxMesh(np.unique(y.ravel()) * 1e-6,  # saeno x
                         np.unique(x.ravel()) * 1e-6,  # saeno y
                         np.unique(z.ravel()) * 1e-6)  # saeno z

    # bring deformations in right order (switch from OpenPIV conversion ot saenopy conversion)
    # - convert to meters for saenopy conversion
    U = np.vstack([v.ravel() * 1e-6, -u.ravel() * 1e-6, w.ravel() * 1e-6]).T

    if return_mesh is True:
        from saenopy.solver import Mesh
        M = Mesh(R, T)
        M.setNodeVar("U_measured", U)
        return M
    M = Solver()
    # provide the node data
    M.setNodes(R)
    # and the tetrahedron data
    M.setTetrahedra(T)
    # set the deformations
    M.setTargetDisplacements(U)
    return M


def center_field(U, R):
    # find center of deformation field analog to force field in Saeno/Saenopy for deformation field
    # U = U[~np.isnan(U)]
    # R = R[~np.isnan(R)]
    Usum = np.sum(U, axis=0)
    B1 = np.einsum("kj,ki->j", R, U ** 2)
    B2 = np.einsum("kj,ki,ki->j", U, R, U)
    A = np.sum(np.einsum("ij,kl,kl->kij", np.eye(3), U, U) - np.einsum("ki,kj->kij", U, U), axis=0)
    B = B1 - B2
    center = np.linalg.inv(A) @ B
    return center


def interpolate_different_mesh(R, U, Rnew):
    """
    Interpolate Deformations (or any quantity) from one mesh to another.
    
    Nonoverlapping regimes between meshes are filled with nans - linear interpolation.

    Parameters
    ----------
    R : Old coordinates (saenopy format: M.R)
    U : Old deformations (saenopy format: M.U)
    Rnew: New coordinates

    Returns
    -------
    Unew : New interpolated deformations 

    """

    # find unique coordinates
    points = [np.unique(R[:, i]) for i in range(3)]

    # shape off data array for reshaping
    arr_shape = (points[0].shape[0], points[1].shape[0], points[2].shape[0])

    # Interpolate deformations to new mesh
    Unew = np.array([interpolate.interpn(points, U[:, i].reshape(arr_shape), Rnew, method='linear', bounds_error=False, fill_value=np.nan) for i in range(3)]).T

    # if the mesh contains nans, interpolate nearest instead of linear to prevent the nans from spreading
    if np.any(np.isnan(Unew)):
        Unew2 = np.array([interpolate.interpn(points, U[:, i].reshape(arr_shape), Rnew, method='nearest', bounds_error=False, fill_value=np.nan) for i in range(3)]).T
        Unew[np.isnan(Unew[:, 0]), :] = Unew2[np.isnan(Unew[:, 0]), :]

    return Unew
