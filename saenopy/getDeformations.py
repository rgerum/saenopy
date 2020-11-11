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


 
# Full 3D Deformation analysis
def getDisplacementsFromStacks(stack_deformed, stack_relaxed, voxel_size, win_um=12, fac_overlap=0.6, signoise_filter=1.3, drift_correction=True):
    from saenopy.multigridHelper import createBoxMesh
    from saenopy import Solver
    # set properties
    voxel_size = np.array(voxel_size)
    window_size = (win_um/voxel_size).astype(int)
    overlap = ((fac_overlap * win_um)/voxel_size).astype(int)
    du, dv, dw = voxel_size
    print ("Calculate Deformations")
    # calculate deformations
    u, v, w, sig2noise = extended_search_area_piv3D(stack_relaxed,stack_deformed,
                                                window_size=window_size,
                                                overlap=overlap,
                                                dt=(1 / du, 1 / dv, 1 / dw),
                                                search_area_size=window_size,       
                                                subpixel_method='gaussian',
                                                sig2noise_method='peak2peak',
                                                width=2,
                                                nfftx=None,nffty=None)

    # correcting stage drift between the field of views
    if drift_correction:
        u -= np.nanmean(u)
        v -= np.nanmean(v)
        w -= np.nanmean(w)
          
    # filter deformations
    uf, vf, wf, mask = sig2noise_val(u, v, w=w, sig2noise=sig2noise, threshold=signoise_filter)
    uf, vf, wf = replace_outliers(uf, vf, wf, max_iter=1, tol=100, kernel_size=2, method='disk')

    # get coodrinates (by multiplication with the ratio of image dimension and deformation grid)
    y, x, z = np.indices(u.shape) 
    y, x, z = (y * stack_deformed.shape[0] * dv/ u.shape[0], 
               x * stack_deformed.shape[1] * du/ u.shape[1],
               z * stack_deformed.shape[2] * dw/ u.shape[2])
    

    # create a box mesh - convert to meters for saenopy conversion
    R, T = createBoxMesh(np.unique(x.ravel())*1e-6,
                         np.unique(y.ravel())*1e-6,
                         np.unique(z.ravel())*1e-6)
                        
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





# Example:

# stack_deformed = getStack(r"..\Mark_and_Find_001_Pos002_S001_t02_z*_RAW_ch00.tif")[:,:,30:-30]  # cut upper lower part here
# stack_relaxed = getStack(r"..\\Mark_and_Find_001_Pos002_S001_t21_z*_RAW_ch00.tif")[:,:,30:-30]

# deformation = getDisplacementsFromStacks(stack_deformed, stack_relaxed, (0.2407,0.2407,1), win_um=12, fac_overlap=0.6, signoise_filter=1.3)  #win  12

# deformation.save("test")















