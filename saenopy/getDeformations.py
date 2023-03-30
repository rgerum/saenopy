# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:41:50 2020

@author: david
"""

import glob as glob
import numpy as np
from pathlib import Path
import re
from openpiv.pyprocess3D import extended_search_area_piv3D
from skimage import io
import natsort
import pandas as pd
import os
import tifffile
from scipy import interpolate
from saenopy.loadHelpers import Saveable
import imageio


def double_glob(text):
    glob_string = text.replace("?", "*")
    print("globbing", glob_string)
    files = glob.glob(glob_string)

    output_base = glob_string
    while "*" in str(output_base):
        output_base = Path(output_base).parent

    regex_string = re.escape(text).replace(r"\*", "(.*)").replace(r"\?", ".*")

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

    match = re.match(r"(.*\.tif)\[(.*)\]", pattern)
    page = 0
    if match:
        pattern, page = match.groups()

    regexp_string = re.sub(r"(?<!{)\\{([^{}]*)\\}(?!})", r"(?P<\1>.*)", re.escape(pattern).replace("\\*\\*", ".*").replace("\\*", ".*"))
    regexp_string3 = ""
    replacement = ""
    count = 1
    for part in re.split(r"(\([^)]*\))", regexp_string):
        if part.startswith("("):
            regexp_string3 += part
            replacement += f"{{{part[4:-4]}}}"
            count += 1
        else:
            regexp_string3 += f"({part})"
            replacement += f"\\{count}"
            count += 1

    regexp_string = regexp_string.replace("\\{\\{", "\\{").replace("\\}\\}", "\\}")
    regexp_string2 = re.compile(regexp_string)
    glob_string = re.sub(r"({[^}]*})", "*", pattern)

    output_base = Path(glob_string).parent
    while "*" in str(output_base):
        output_base = Path(output_base).parent

    file_list = []
    for file in output_base.rglob(str(Path(glob_string).relative_to(output_base))):#glob.glob(glob_string, recursive=True):
        file = str(Path(file))
        group = regexp_string2.match(file).groupdict()
        template_name = re.sub(regexp_string3, replacement, file.replace("{", "{{").replace("}", "}}"))
        group["filename"] = file
        group["template"] = template_name
        try:
            page = int(page)
            if page != 0:
                group["filename"] = file+f"[{page}]"
                group["template"] = template_name + "[" + page + "]"
            file_list.append(group)
        except ValueError:
            tif = tifffile.TiffReader(file)
            group["template"] = template_name + "[" + page + "]"
            for i in range(len(tif.pages)):
                group["filename"] = file+f"[{i}]"
                group[page] = i
                file_list.append(group.copy())
    return pd.DataFrame(file_list), output_base


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

    DTYPEf = np.float
    DTYPEi = np.int

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


def filenames_to_channel_template(filenames):
    for i in range(len(filenames[0])):
        for file in filenames[1:]:
            if file[:i] != filenames[0][:i]:
                break
        else:
            continue
        i -= 1
        while filenames[0][i].isdigits():
            i -= 1
        i += 1
        break
    for j in range(len(filenames[0]) - 1, 0, -1):
        for file in filenames[1:]:
            if file[j:] != filenames[0][j:]:
                break
        else:
            continue
        j += 1
        while filenames[0][j].isdigits():
            j += 1
        break
    template = filenames[0][:i] + "{c:" + filenames[0][i:j] + "}" + filenames[0][j:]
    return template


def template_to_array(filename, crop):
    from saenopy.result_file import get_channel_placeholder
    filename, channel1 = get_channel_placeholder(filename)
    results1, output_base = format_glob(filename)
    for (template, d1) in results1.groupby("template"):
        if template.endswith("[z]"):
            template = template.replace("[z]", "[{z}]")
        z_indices = natsort.natsorted(d1.z.unique())
        if crop is not None and "z" in crop:
            z_indices = z_indices[slice(*crop["z"])]
        if channel1 is not None:
            c_indices = natsort.natsorted(d1.c.unique())
            c_indices.remove(channel1)
            c_indices = [channel1] + c_indices
            image_filenames = []
            for z in z_indices:
                image_filenames.append([])
                for c in c_indices:
                    image_filenames[-1].append(template.format(z=z, c=c))
        else:
            image_filenames = []
            for z in z_indices:
                image_filenames.append([template.format(z=z)])
            c_indices = [""]
    return image_filenames, c_indices


def readTiff(image_filenames):
    if re.match(f".*\.tiff?(\[.*\])?$", str(image_filenames)):
        image_filenames = str(image_filenames)
        page = 0
        if image_filenames.endswith("]"):
            image_filenames, page = re.match(r"(.*)\[(\d*)\]", image_filenames).groups()
            page = int(page)
        tif = tifffile.TiffReader(image_filenames)
        page = tif.pages[page]
        if isinstance(page, list):
            page = page[0]
        return page.asarray()
    im = imageio.imread(image_filenames)
    return im


def load_image_files_to_nparray(image_filenames, crop=None):
    if isinstance(image_filenames, str):
        im = readTiff(image_filenames)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if crop is not None and "x" in crop:
            im = im[:, slice(*crop["x"])]
        if crop is not None and "y" in crop:
            im = im[slice(*crop["y"])]
        return im
    else:
        return [load_image_files_to_nparray(i, crop) for i in image_filenames]


class Stack(Saveable):
    __save_parameters__ = ['template', 'image_filenames', 'filename', 'voxel_size', 'shape', 'channels', 'crop', 'packed_files']
    template: str = None
    image_filenames: list = None

    packed_files: list = None

    images: list = None
    input: str = ""
    _shape = None
    voxel_size: tuple = None
    channels: list = None
    images_channels: list = None

    leica_file = None

    def __init__(self, template=None, voxel_size=None, filename=None, shape=None, channels=None, image_filenames=None, crop=None, **kwargs):
        print("stack",template)
        if template is None:
            if isinstance(filename, list):
                template = filenames_to_channel_template(filename)
            else:
                template = filename
        if template is not None:
            template = template.replace("*", "{z}")
        self.template = template
        self.crop = crop
        if image_filenames is None and template is not None:
            match = re.match(r"(.*)\{f\:(\d*)\}\{c\:(\d*)\}(?:\{t\:(\d*)\})?.lif", template)
            if match:
                from saenopy.gui.lif_reader import LifFile
                self.leica_filename, self.leica_folder, self.leica_channel, self.leica_time = match.groups()
                if self.leica_time is None:
                    self.leica_time = 0
                else:
                    self.leica_time = int(self.leica_time)
                self.leica_channel = int(self.leica_channel)
                self.leica_file = LifFile(self.leica_filename + ".lif").get_image(self.leica_folder)
                self.channels = [str(self.leica_channel)]
                for i in range(0, self.leica_file.channels):
                    if i != self.leica_channel:
                        self.channels.append(str(i))
            else:
                self.image_filenames, self.channels = template_to_array(template, crop)
        else:
            self.image_filenames = image_filenames
            self.channels = channels
        if shape is not None:
            self._shape = shape
        if 0:
            self.channels = channels
            if channels is not None:
                self.filename = filename
                self.images = natsort.natsorted(glob.glob(filename[0]))
                self.images_channels = []
                for filename_pattern in filename[1:]:
                    self.images_channels.append(natsort.natsorted(glob.glob(filename_pattern)))
            elif isinstance(filename, str):
                self.filename = str(filename)
                self.images = natsort.natsorted(glob.glob(filename))
            else:
                self.filename = list(filename)
                self.images = list(filename)
        self.voxel_size = voxel_size
        super().__init__(**kwargs)

    def pack_files(self):
        images = np.array(self.image_filenames)[:, :]
        images = np.asarray(load_image_files_to_nparray(images, self.crop)).T

        self.packed_files = images

    def description(self, z):
        try:
            return f"shape {self.shape}px\nsize {np.array(self.shape[:3])*np.array(self.voxel_size)}μm\nvoxel size {self.voxel_size}μm\n{self.image_filenames[z][0]}"
        except (IndexError, TypeError):
            return ""

    @property
    def shape(self) -> tuple:
        if self.leica_file is not None:
            return (self.leica_file.dims.y, self.leica_file.dims.x, self.leica_file.dims.z, 1)
        if self._shape is None:
            im = readTiff(self.image_filenames[0][0])
            if self.crop is not None and "x" in self.crop:
                im = im[:, slice(*self.crop["x"])]
            if self.crop is not None and "y" in self.crop:
                im = im[slice(*self.crop["y"])]
            self._shape = tuple(list(im.shape[:2]) + list(np.array(self.image_filenames).shape))
        return self._shape

    def get_image(self, z, channel):
        return io.imread(self.image_filenames[z][channel])

    def __getitem__(self, index) -> np.ndarray:
        """ axes are y, x, rgb, z, c """
        if self.leica_file is not None:
            if isinstance(index[3], slice):
                z_min = 0
                if index[3].start is not None:
                    z_min = index[3].start
                z_max = self.shape[3]
                if index[3].stop is not None:
                    z_max = index[3].stop
            else:
                z_min = index[3]
                z_max = z_min + 1
            images = []
            for z in range(z_min, z_max):
                im = np.asarray(self.leica_file.get_frame(z, t=self.leica_time, c=int(self.channels[index[4]])))
                images.append(im)
            images = np.asarray(images)
            #np.asarray([self.leica_file.get_frame(z) for z in range(z_min, z_max)])
            images = images.transpose(1, 2, 0)[:, :, None, :]
            if isinstance(index[3], int):
                images = images[:, :, :, 0]
            return images[index[0], index[1], index[2]]
        if self.packed_files is None:
            images = np.array(self.image_filenames)[index[3], index[4]]
            images = np.asarray(load_image_files_to_nparray(images, self.crop)).T
        else:
            images = self.packed_files[:, :, :, index[4], index[3]]
        images = np.swapaxes(images, 0, 2)
        return images[index[0], index[1], index[2]]

        images = self.images
        channel = 0
        if len(index) == 4 and index[3] > 0 and self.images_channels is not None:
            images = self.images_channels[index[3]-1]
            channel = index[3]
        if isinstance(index[2], slice):
            return np.array([self.__getitem__(tuple([index[0], index[1], i, channel])) for i in range(index[2].start, index[2].stop, index[2].step if index[2].step is not None else 1)]).transpose(1, 2, 0)
        try:
            im = io.imread(images[index[2]])
        except (IndexError, IOError) as err:
            print("ERROR", err)
            im = np.zeros(self.shape[:2])
        if len(im.shape) == 3:
            im = im[:, :, 0]
        return im[index[0], index[1]]

    def __array__(self) -> np.ndarray:
        return self[:, :, :, :, 0]
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
