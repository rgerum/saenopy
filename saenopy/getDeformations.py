# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:41:50 2020

@author: david
"""

import sys
import glob as glob
from tqdm import tqdm
import numpy as np
from openpiv.pyprocess3D import extended_search_area_piv3D
from openpiv.validation import sig2noise_val
from openpiv.filters import replace_outliers
from openpiv.lib import replace_nans
from skimage import io
from scipy import interpolate
 
# Full 3D Deformation analysis
def getDisplacementsFromStacks(stack_deformed, stack_relaxed, voxel_size, win_um=12, fac_overlap=0.6, signoise_filter=1.3, drift_correction=True):
    from saenopy.multigridHelper import createBoxMesh
    from saenopy import Solver

    # set properties
    voxel_size = np.array(voxel_size)
    window_size = (win_um/voxel_size).astype(int)
    overlap = ((fac_overlap * win_um)/voxel_size).astype(int)
    du, dv, dw = voxel_size
    print("Calculate Deformations")

    # calculate deformations
    u, v, w, sig2noise = extended_search_area_piv3D(stack_relaxed,stack_deformed,
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
    uf, vf, wf, mask = sig2noise_val(u, v, w=w, sig2noise=sig2noise, threshold=signoise_filter)
    uf, vf, wf = replace_outliers(uf, vf, wf, max_iter=1, tol=100, kernel_size=2, method='disk')

    # get coordinates (by multiplication with the ratio of image dimension and deformation grid)
    y, x, z = np.indices(u.shape) 
    y, x, z = (y * stack_deformed.shape[0] * dv/ u.shape[0], 
               x * stack_deformed.shape[1] * du/ u.shape[1],
               z * stack_deformed.shape[2] * dw/ u.shape[2])

    # create a box mesh - convert to meters for saenopy conversion
    R, T = createBoxMesh(np.unique(y.ravel())*1e-6,      #saeno x
                         np.unique(x.ravel())*1e-6,      #saeno y
                         np.unique(z.ravel())*1e-6)      #saeno z
                        
    # bring deformations in right order (switch from OpenPIV conversion ot saenopy conversion)
    # - convert to meters for saenopy conversion
    U = np.vstack([v.ravel()*1e-6, -u.ravel()*1e-6, w.ravel()*1e-6]).T
    M = Solver()
    # provide the node data
    M.setNodes(R)
    # and the tetrahedron data
    M.setTetrahedra(T)
    # set the deformations
    M.setTargetDisplacements(U)
    return M

# read in image stack
def getStack(filename):
    images = glob.glob(filename)
    im = io.imread(images[0], as_gray=True)
    stack = np.zeros((im.shape[0], im.shape[1], len(images)), dtype=im.dtype)
    for i, im in enumerate(images):
        stack[:, :, i] = io.imread(im, as_gray=True)
    return stack


def center_field(U,R):
# find center of deformation field analog to force field in Saeno/Saenopy for deformation field
        # U = U[~np.isnan(U)]
        # R = R[~np.isnan(R)]
        Usum = np.sum(U, axis=0)
        B1 = np.einsum("kj,ki->j", R, U**2)
        B2 = np.einsum("kj,ki,ki->j", U, R, U)
        A = np.sum(np.einsum("ij,kl,kl->kij", np.eye(3), U,U) - np.einsum("ki,kj->kij", U, U), axis=0)
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
