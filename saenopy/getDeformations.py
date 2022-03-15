# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:41:50 2020

@author: david
"""

import sys
import glob as glob
from tqdm import tqdm
import numpy as np
from pathlib import Path
import re
from openpiv.pyprocess3D import extended_search_area_piv3D
from openpiv.validation import sig2noise_val
from openpiv.filters import replace_outliers
from openpiv.lib import replace_nans
from skimage import io
import natsort
import pandas as pd
import os
from scipy import interpolate
from saenopy.loadHelpers import Saveable


def double_glob(text):
    glob_string = text.replace("?", "*")
    print("globbing", glob_string)
    files = glob.glob(glob_string)

    output_base = glob_string
    while "*" in str(output_base):
        output_base = Path(output_base).parent

    regex_string = re.escape(text).replace("\*", "(.*)").replace("\?", ".*")

    results = []
    for file in files:
        file = os.path.normpath(file)
        print(file, regex_string)
        match = re.match(regex_string, file).groups()
        reconstructed_file = regex_string
        for element in match:
            reconstructed_file = reconstructed_file.replace("(.*)", element, 1)
        reconstructed_file = reconstructed_file.replace(".*", "*")
        reconstructed_file = re.sub(r'\\(.)', r'\1', reconstructed_file)
        if reconstructed_file not in results:
            results.append(reconstructed_file)
    return results, output_base


def format_glob(pattern):
    pattern = str(Path(pattern))
    regexp_string = re.sub(r"\\{([^}]*)\\}", r"(?P<\1>.*)", re.escape(pattern).replace("\\*\\*", ".*").replace("\\*", ".*"))
    regexp_string3 = ""
    replacement = ""
    count = 1
    for part in re.split("(\([^)]*\))", regexp_string):
        if part.startswith("("):
            regexp_string3 += part
            replacement += f"{{{part[4:-4]}}}"
            count += 1
        else:
            regexp_string3 += f"({part})"
            replacement += f"\\{count}"
            count += 1

    regexp_string2 = re.compile(regexp_string)
    glob_string = re.sub(r"({[^}]*})", "*", pattern)

    output_base = glob_string
    while "*" in str(output_base):
        output_base = Path(output_base).parent

    file_list = []
    for file in output_base.rglob(str(Path(glob_string).relative_to(output_base))):#glob.glob(glob_string, recursive=True):
        file = str(Path(file))
        group = regexp_string2.match(file).groupdict()
        template_name = re.sub(regexp_string3, replacement, file)
        group["filename"] = file
        group["template"] = template_name
        file_list.append(group)
    return pd.DataFrame(file_list), output_base


def getDisplacementsFromStacks2(stack_relaxed, stack_deformed, win_um, fac_overlap, signoise_filter, drift_correction):
    voxel_size1 = stack_deformed.voxel_size
    voxel_size2 = stack_relaxed.voxel_size

    np.testing.assert_equal(voxel_size1, voxel_size2, f"The two stacks do not have the same voxel size. {voxel_size1}, {voxel_size2}")

    np.testing.assert_equal(stack_deformed.shape, stack_relaxed.shape, f"The two stacks do not have the same voxel count. {stack_deformed.shape}, {stack_relaxed.shape}")

    stack_deformed = np.array(stack_deformed)
    stack_relaxed = np.array(stack_relaxed)
    M = getDisplacementsFromStacks_old(stack_deformed, stack_relaxed, voxel_size1,
                                        win_um=win_um,
                                        fac_overlap=fac_overlap,
                                        signoise_filter=signoise_filter,
                                        drift_correction=drift_correction,
                                        return_mesh=True)
    # center
    M.R = (M.R - np.min(M.R, axis=0)) - (np.max(M.R, axis=0) - np.min(M.R, axis=0)) / 2
    return M

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
    print("Calculate Deformations")

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
    uf, vf, wf, mask = sig2noise_val(u, v,sig2noise, w=w, threshold=signoise_filter)
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


# read in image stack
def getStack(filename):
    if isinstance(filename, str):
        images = glob.glob(filename)
    else:
        images = list(filename)
    im = io.imread(images[0], as_gray=True)
    stack = np.zeros((im.shape[0], im.shape[1], len(images)), dtype=im.dtype)
    for i, im in enumerate(images):
        stack[:, :, i] = io.imread(im, as_gray=True)
    return stack

class Stack(Saveable):
    __save_parameters__ = ['filename', 'voxel_size', 'shape']
    images: list = None
    input: str = ""
    _shape = None
    voxel_size: tuple = None

    def __init__(self, filename, voxel_size, shape=None):
        if shape is not None:
            self._shape = shape
        if isinstance(filename, str):
            self.filename = str(filename)
            self.images = natsort.natsorted(glob.glob(filename))
        else:
            self.filename = list(filename)
            self.images = list(filename)
        self.voxel_size = voxel_size

    def description(self, z):
        return f"shape {self.shape}px\nsize {np.array(self.shape)*np.array(self.voxel_size)}μm\nvoxel size {self.voxel_size}μm\n{self.images[z]}"

    @property
    def shape(self) -> tuple:
        if self._shape is None:
            im = io.imread(self.images[0])
            self._shape = tuple(list(im.shape[:2]) + [len(self.images)])
        return self._shape

    def __getitem__(self, index) -> np.ndarray:
        try:
            im = io.imread(self.images[index[2]])
        except (IndexError, IOError) as err:
            print("ERROR", err)
            im = np.zeros(self.shape[:2])
        if len(im.shape) == 3:
            im = im[:, :, 0]
        return im[index[0], index[1]]

    def __array__(self) -> np.ndarray:
        return getStack(self.images)

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
