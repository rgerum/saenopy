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
    regexp_string = re.sub(r"\\{([^}]*)\\}", r"(?P<\1>.*)", re.escape(pattern).replace("\\*\\*", ".*").replace("\\*", ".*"))
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


def get_displacements_from_stacks(stack_relaxed, stack_deformed, win_um, elementsize, signoise_filter, drift_correction):
    fac_overlap = 1 - (elementsize/win_um)
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


def filenames_to_channel_template(filenames):
    for i in range(len(filenames[0])):
        for file in filenames[1:]:
            if file[:i] != filenames[0][:i]:
                break
        else:
            continue
        i -= 1
        while filenames[0][i] in "0123456789":
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
        while filenames[0][j] in "0123456789":
            j += 1
        break
    template = filenames[0][:i] + "{c:" + filenames[0][i:j] + "}" + filenames[0][j:]
    return template


def template_to_array(filename):
    from saenopy.result_file import get_channel_placeholder
    filename, channel1 = get_channel_placeholder(filename)
    results1, output_base = format_glob(filename)
    for (template, d1) in results1.groupby("template"):
        z_indices = natsort.natsorted(d1.z.unique())
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


def load_image_files_to_nparray(image_filenames):
    if isinstance(image_filenames, str):
        return io.imread(image_filenames)
    else:
        return [to_stack(i) for i in image_filenames]


class Stack(Saveable):
    __save_parameters__ = ['template', 'image_filenames', 'filename', 'voxel_size', 'shape', 'channels']
    template: str = None
    image_filenames: list = None

    images: list = None
    input: str = ""
    _shape = None
    voxel_size: tuple = None
    channels: list = None
    images_channels: list = None

    def __init__(self, template=None, voxel_size=None, filename=None, shape=None, channels=None, image_filenames=None):
        if template is None:
            if isinstance(filename, list):
                template = filenames_to_channel_template(filename)
            else:
                template = filename
            template = template.replace("*", "{z}")
        self.template = template
        if image_filenames is None:
            self.image_filenames, self.channels = template_to_array(template)
        else:
            self.image_filenames = image_filenames
            self.channels = channels
        print("stack", filename, voxel_size, shape, channels)
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

    def description(self, z):
        try:
            return f"shape {self.shape}px\nsize {np.array(self.shape[:3])*np.array(self.voxel_size)}μm\nvoxel size {self.voxel_size}μm\n{self.image_filenames[z][0]}"
        except IndexError:
            return ""

    @property
    def shape(self) -> tuple:
        if self._shape is None:
            im = io.imread(self.image_filenames[0][0])
            self._shape = tuple(list(im.shape[:2]) + list(np.array(self.image_filenames).shape))
        return self._shape

    def get_image(self, z, channel):
        return io.imread(self.image_filenames[z][channel])

    def __getitem__(self, index) -> np.ndarray:
        """ axes are y, x, z, c """
        images = np.array(self.image_filenames)[index[2], index[3]]
        return np.swapaxes(np.asarray(load_image_files_to_nparray(images)).T, 1, 0)[index[0], index[1]]

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


if __name__ == "__main__":
    from saenopy.result_file import get_channel_placeholder
    import tifffile
    def get_iterator(values, name="", iter=None):
        if iter is None:
            for v in values:
                yield {name: v}
        else:
            for v in values:
                for t in iter:
                    t[name] = v
                    yield t

    def template_to_array(filename):
        filename, channel1 = get_channel_placeholder(filename)
        results1, output_base = format_glob(filename)
        for (template, d1) in results1.groupby("template"):
            z_indices = natsort.natsorted(d1.z.unique())
            if channel1 is not None:
                c_indices = natsort.natsorted(d1.c.unique())
                c_indices.remove(channel1)
                c_indices = [channel1] + c_indices
                image_filenames = []
                for c in c_indices:
                    image_filenames.append([])
                    for z in z_indices:
                        image_filenames[-1].append(template.format(z=z, c=c))
            else:
                image_filenames = [[]]
                for z in z_indices:
                    image_filenames.append(template.format(z=z))
                c_indices = [""]
        return image_filenames, c_indices

    if 0:
        image_filenames, c_indices = template_to_array("/home/richard/.local/share/saenopy/2_DynamicalSingleCellTFM/data/Pos*_S*_t00_z{z}_ch{c:01}.tif")
        for im in image_filenames:
            print("channel")
            for i in im[:3]:
                print(i)
            print("...")
        print(c_indices)
        image_filenames, c_indices = template_to_array("/home/richard/.local/share/saenopy/1_ClassicSingleCellTFM/Deformed/Mark_and_Find_001/Pos003_S001_z{z}_ch{c:01}.tif")
        for im in image_filenames:
            print("channel")
            for i in im[:3]:
                print(i)
            print("...")
        image_filenames = np.array(image_filenames)[0:2, 1:5]
        def to_stack(image_filenames):
            if isinstance(image_filenames, str):
                return io.imread(image_filenames)
            else:
                return [to_stack(i) for i in image_filenames]
        image_stack = np.swapaxes(np.asarray(to_stack(image_filenames)).T, 1, 0)
        shape = list(image_stack.shape)
        shape[0]
        print(image_stack.shape)
        print(image_filenames)
        print(c_indices)
        exit()

    def process_line(filename, output_path):
        results = []
        filename, channel1 = get_channel_placeholder(filename)
        results1, output_base = format_glob(filename)
        for (r1, d1) in results1.groupby("template"):
            counts = {"z": len(d1.z.unique())}
            iterator = get_iterator(d1.z.unique(), "z")
            if channel1 is not None:
                r1 = r1.replace("{c}", "{c:"+channel1+"}")
                counts["c"] = len(d1.c.unique())
                iterator = get_iterator(d1.c.unique(), "c", iterator)
            if "t" in d1.columns:
                counts["t"] = len(d1.t.unique())
                iterator = get_iterator(d1.t.unique(), "t", iterator)
            #print(r1, counts, np.prod([c for c in counts.values()]), len(d1))
            #assert np.prod([c for c in counts.values()]) == len(d1)
            # check if all files exist and have the same shape
            template = d1.iloc[0].template
            shape = None
            for props in iterator:
                filename = template.format(**props)
                if not Path(filename).exists():
                    raise FileNotFoundError()
                f = tifffile.TiffFile(filename)
                if shape is None:
                    shape = f.pages[0].shape
                else:
                    if f.pages[0].shape != shape:
                        raise ValueError()

            # create the output path
            output = Path(output_path) / os.path.relpath(r1, output_base)
            output = output.parent / output.stem
            output = Path(str(output).replace("*", "").replace("{c}", "{c:" + str(channel1) + "}") + ".npz")

            results.append({"filename": r1, "dimensions": counts, "output": output})
            if "t" in d1.columns:
                results[-1]["times"] = natsort.natsorted(d1.t.unique())
        return results

    from saenopy.result_file import Result
    def get_stacksX(filename, output_path, voxel_size, time_delta=None, reference_stack=None, exist_overwrite_callback=None):
        results1 = process_line(filename, output_path)
        results = []
        if reference_stack is not None:
            results2 = process_line(reference_stack, output_path)
            if len(results1) != len(results2):
                raise ValueError(f"Number of active stacks ({len(results1)}) does not match the number of reference stacks ({len(results2)}).")
            for r1, r2 in zip(results1, results2):
                if r1["dimensions"]["z"] != r2["dimensions"]["z"]:
                    raise ValueError("active and reference stack need the same number of z slices")
                if "t" in r2["dimensions"]:
                    raise ValueError("the reference stack is not allowed to have different time points")
                if "c" in r1["dimensions"]:
                    if "c" not in r2["dimensions"]:
                        raise ValueError("if the active stack has channels the reference stack also needs channels")
                    if r1["dimensions"]["c"] != r2["dimensions"]["c"]:
                        raise ValueError("the active stack and the reference stack also need the same number of channels")

                if "t" in r1["dimensions"]:
                    stacks = []
                    for t in r1["times"]:
                        stacks.append(Stack(r1["filename"].replace("{t}", t), voxel_size))
                else:
                    stacks = [Stack(r1["filename"], voxel_size)]

                output = r1["output"]
                if output.exists() and exist_overwrite_callback is not None:
                    mode = exist_overwrite_callback(output)
                    if mode == 0:
                        break
                    if mode == "read":
                        print('exists', output)
                        data = Result.load(output)
                        results.append(data)
                        continue

                data = Result(
                    output=output,
                    template=r1["filename"],
                    stack=stacks,
                    stack_reference=Stack(r2["filename"], voxel_size),
                    time_delta=time_delta,
                )
                data.save()
                results.append(data)
        else:
            for r1 in results1:
                if "t" in r1["dimensions"]:
                    stacks = []
                    for t in r1["times"]:
                        stacks.append(Stack(r1["filename"].replace("{t}", t), voxel_size))
                else:
                    stacks = [Stack(r1["filename"], voxel_size)]

                output = r1["output"]
                if output.exists() and exist_overwrite_callback is not None:
                    mode = exist_overwrite_callback(output)
                    if mode == 0:
                        break
                    if mode == "read":
                        print('exists', output)
                        data = Result.load(output)
                        results.append(data)
                        continue

                data = Result(
                    output=r1["output"],
                    template=r1["filename"],
                    stack=stacks,
                    time_delta=time_delta,
                )
                data.save()
                results.append(data)
        return results

    if 0:
        filenames = ['/home/richard/.local/share/saenopy/1_ClassicSingleCellTFM/Deformed/Mark_and_Find_001/Pos003_S001_z*_ch000.tif', '/home/richard/.local/share/saenopy/1_ClassicSingleCellTFM/Deformed/Mark_and_Find_001/Pos003_S001_z*_ch020.tif', '/home/richard/.local/share/saenopy/1_ClassicSingleCellTFM/Deformed/Mark_and_Find_001/Pos003_S001_z*_ch010.tif']
        for i in range(len(filenames[0])):
            for file in filenames[1:]:
                if file[:i] != filenames[0][:i]:
                    break
            else:
                continue
            i -= 1
            while filenames[0][i] in "0123456789":
                i -= 1
            i += 1
            break
        for j in range(len(filenames[0])-1, 0, -1):
            for file in filenames[1:]:
                print(file[j:], filenames[0][j:], file[j:] != filenames[0][j:])
                if file[j:] != filenames[0][j:]:
                    break
            else:
                continue
            j += 1
            while filenames[0][j] in "0123456789":
                j += 1
            break
        template = filenames[0][:i] + "{c:" + filenames[0][i:j] + "}" + filenames[0][j:]
        print(i, j, filenames[0][:i], filenames[0][i:j], filenames[0][j:])
        print(template)
        exit(0)
    #process_line("/home/richard/.local/share/saenopy/2_DynamicalSingleCellTFM/data/Pos*_S*_t*_z{z}_ch{c:01}.tif")
    from saenopy import get_stacks
    get_stacks("/home/richard/.local/share/saenopy/2_DynamicalSingleCellTFM/data/Pos002_S001_t00_z{z}_ch{c:01}.tif",
               "/home/richard/.local/share/saenopy/2_DynamicalSingleCellTFM/output",
               [0.1, 0.1, 0.1],
               time_delta=1,
               #reference_stack="/home/richard/.local/share/saenopy/2_DynamicalSingleCellTFM/data/Pos002_S001_t00_z{z}_ch{c:01}.tif",
               )
    #process_line("/home/richard/.local/share/saenopy/2_DynamicalSingleCellTFM/data/Pos002_S001_t00_z{z}_ch01.tif")
    #process_line("/home/richard/.local/share/saenopy/2_DynamicalSingleCellTFM/data/Pos002_S001_t{t}_z{z}_ch01.tif")
    #process_line("/home/richard/.local/share/saenopy/2_DynamicalSingleCellTFM/data/Pos002_S001_t*_z{z}_ch01.tif")
    #Stack("/home/richard/.local/share/saenopy/2_DynamicalSingleCellTFM/data/Pos002_S001_t00_z{z}_ch01.tif", [0.1, 0.1, 0.1])
