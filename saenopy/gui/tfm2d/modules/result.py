import io
import glob
from pathlib import Path
import os
import numpy as np
import tifffile
import imageio
import re
import matplotlib.pyplot as plt
import traceback

from saenopy.saveable import Saveable
from saenopy.result_file import make_path_absolute
from .draw import get_mask_using_gui

from saenopy.pyTFM.plotting import plot_continuous_boundary_stresses
from saenopy.pyTFM.plotting import show_quiver


def read_tiff(image_filenames):
    if re.match(r".*\.tiff?(\[.*\])?$", str(image_filenames)):
        image_filenames = str(image_filenames)
        page = 0
        if image_filenames.endswith("]"):
            image_filenames, page = re.match(r"(.*)\[(\d*)\]", image_filenames).groups()
            page = int(page)
        with tifffile.TiffReader(image_filenames) as tif:
            page = tif.pages[page]
            if isinstance(page, list):  # pragma: no cover
                page = page[0]
            return page.asarray()
    im = imageio.v2.imread(image_filenames)
    if len(im.shape) == 3:
        im = np.mean(im[:, :, :3], axis=2)
    return im


class Result2D(Saveable):
    __save_parameters__ = ['bf', 'input', 'reference_stack', 'output', 'pixel_size', 'u', 'v', 'mask_val', 'mask_std',
                           'tx', 'ty', 'fx', 'fy',
                           'shape', 'mask', 'res_dict',
                           'piv_parameters', 'force_parameters',
                           'borders_inter_shape', 'borders_edge_lines', 'lt', 'min_v', 'max_v',
                           '___save_name__', '___save_version__']
    ___save_name__ = "Result2D"
    ___save_version__ = "1.0"

    input: str = None
    reference_stack: str = None
    output: str = None
    state: bool = False
    pixel_size: float = None

    u: np.ndarray = None
    v: np.ndarray = None
    mask_val: np.ndarray = None
    mask_std: np.ndarray = None

    tx: np.ndarray = None
    ty: np.ndarray = None

    fx: np.ndarray = None
    fy: np.ndarray = None

    drift_parameters: dict = {}
    piv_parameters: dict = {}
    force_parameters: dict = {}
    force_gen_parameters: dict = {}
    stress_parameters: dict = {}

    shape: tuple = None
    mask: np.ndarray = None

    res_dict: dict = None

    im_displacement: np.ndarray = None
    im_force: np.ndarray = None
    im_tension: np.ndarray = None

    borders_inter_shape: tuple = None
    borders_edge_lines: list = None
    lt: dict = None
    min_v: float = None
    max_v: float = None

    def __init__(self, output, bf, input, reference_stack, pixel_size, **kwargs):
        self.bf = bf
        self.input = input
        self.reference_stack = reference_stack
        self.pixel_size = pixel_size
        self.output = output
        if "res_dict" not in kwargs:
            kwargs["res_dict"] = {}
        self.res_dict = {}

        path_b = Path(self.input)
        path_a = Path(self.reference_stack)
        path_b = path_b.parent / (path_b.stem + "_corrected" + path_b.suffix)
        path_a = path_a.parent / (path_a.stem + "_corrected" + path_a.suffix)
        self.input_corrected = str(path_b)
        self.reference_stack_corrected = str(path_a)

        self.state = False

        self.get_image(0)

        super().__init__(**kwargs)

    def get_image(self, index, corrected=True):
        try:
            if index == 0:
                if corrected:
                    try:
                        im = read_tiff(self.input_corrected)
                    except FileNotFoundError:
                        im = read_tiff(self.input)
                else:
                    im = read_tiff(self.input)
            elif index == -1:
                im = read_tiff(self.bf)
            else:
                if corrected:
                    try:
                        im = read_tiff(self.reference_stack_corrected)
                    except FileNotFoundError:
                        im = read_tiff(self.reference_stack)
                else:
                    im = read_tiff(self.reference_stack)
        except FileNotFoundError as err:
            traceback.print_exception(err)
            h = 255
            w = 255
            if self.shape is not None:
                h, w = self.shape[:2]
            im = np.zeros([h, w, 3], dtype=np.uint8)
            im[:, :, 0] = 255
            im[:, :, 2] = 255
        if self.shape is None:
            self.shape = im.shape
        return im

    def get_deformation_field(self):
        if self.im_displacement is None:
            fig1, ax = show_quiver(self.u, self.v, cbar_str="deformations\n[pixels]")
            self.im_displacement = fig_to_numpy(fig1, self.shape)
        return self.im_displacement

    def get_force_field(self):
        if self.im_force is None:
            fig1, ax = show_quiver(self.tx, self.ty, cbar_str="tractions\n[Pa]")
            self.im_force = fig_to_numpy(fig1, self.shape)
        return self.im_force

    def get_line_tensions(self):
        if self.im_tension is None:
            fig3, ax = plot_continuous_boundary_stresses([self.borders_inter_shape, self.borders_edge_lines, self.lt, self.min_v, self.max_v],
                                                         cbar_style="outside")
            self.im_tension = fig_to_numpy(fig3, self.shape)
        return self.im_tension

    def save(self, file_name=None):
        if file_name is None:
            file_name = self.output
        Path(self.output).parent.mkdir(exist_ok=True, parents=True)
        super().save(file_name)

    def on_load(self, filename: str):
        self.output = str(Path(filename))

    def get_mask(self):
        if self.mask is None:
            self.mask = get_mask_using_gui(self.bf)
        return self.mask

    def get_absolute_path(self):
        return make_path_absolute(self.input_corrected, Path(self.output).parent)

    def get_absolute_path_reference(self):
        return make_path_absolute(self.reference_stack, Path(self.output).parent)

    def get_absolute_path_bf(self):
        return make_path_absolute(self.bf, Path(self.output).parent)

    def get_data_structure(self):
        if self.shape is None:
            self.get_image(0)
        return {
            "dimensions": 2,
            "z_slices_count": 1,
            "im_shape": [self.shape[0], self.shape[1], 1],
            "time_point_count": 1,
            "has_reference": True,
            "voxel_size": [self.pixel_size, self.pixel_size, 1],
            "time_delta": None,
            "channels": ["cells", "beads"],
            "fields": {
                "deformation": {
                    "type": "vector",
                    "measure": "deformation",
                    "unit": "pixel",
                    "name": "displacements_measured",
                },
                "forces": {
                    "type": "vector",
                    "measure": "force",
                    "unit": "pixel",
                    "name": "force",
                }
            }
        }

    def get_image_data(self, time_point, channel="default", use_reference=False):
        if channel == "cells":
            im = self.get_image(-1)
        else:
            if use_reference:
                im = self.get_image(0)
            else:
                im = self.get_image(1)
        if len(im.shape) == 2:
            return im[:, :, None, None]
        return im[:, :, :, None]

    def get_field_data(self, name, time_point):
        class Mesh2D:
            pass

        vx = None
        vy = None
        vf = 1

        if name == "deformation":
            vx = self.u
            vy = self.v
            vf = 10
        if name == "forces":
            print("do force")
            vx = self.tx
            vy = self.ty
            vf = 0.1

        if vx is not None:
            mesh = Mesh2D()
            mesh.units = "pixels"
            f = self.shape[0] / vx.shape[0]
            x, y = np.meshgrid(np.arange(vx.shape[1]), np.arange(vx.shape[0]))
            x = x * f
            y = y * f
            y = self.shape[0] - y
            mesh.nodes = np.array([x.ravel(), y.ravel()]).T
            mesh.displacements_measured = np.array([vx.ravel(), -vy.ravel()]).T * vf
            return mesh, mesh.displacements_measured
        return None, None


def fig_to_numpy(fig1, shape):
    fig1.axes[0].set_position([0, 0, 1, 1])
    fig1.axes[1].set_position([1, 1, 0.1, 0.1])
    fig1.set_dpi(100)
    fig1.set_size_inches(shape[1] / 100, shape[0] / 100)
    with io.BytesIO() as buff:
        plt.savefig(buff, format="png")
        buff.seek(0)
        return plt.imread(buff)


def get_stacks2D(output_path, bf_stack, active_stack, reference_stack, pixel_size,
               exist_overwrite_callback=None,
               load_existing=False):
    output_base = Path(bf_stack).parent
    while "*" in str(output_base):
        output_base = Path(output_base).parent

    bf_stack = sorted(glob.glob(str(bf_stack)))
    output_path = str(output_path)
    active_stack = sorted(glob.glob(str(active_stack)))
    reference_stack = sorted(glob.glob(str(reference_stack)))

    if len(bf_stack) == 0:
        raise ValueError("no bf image selected")
    if len(active_stack) == 0:
        raise ValueError("no active image selected")
    if len(reference_stack) == 0:
        raise ValueError("no reference image selected")

    if len(bf_stack) != len(active_stack):
        raise ValueError(f"the number of bf images ({len(bf_stack)}) does not match the number of active images {len(active_stack)}")
    if len(bf_stack) != len(reference_stack):
        raise ValueError(f"the number of bf images ({len(bf_stack)}) does not match the number of reference images {len(reference_stack)}")

    results = []
    for i in range(len(bf_stack)):
        im0 = bf_stack[i]
        im1 = active_stack[i]
        im2 = reference_stack[i]

        output = Path(output_path) / os.path.relpath(im0, output_base)
        output = output.parent / output.stem
        output = Path(str(output) + ".saenopy2D")

        if output.exists():
            if exist_overwrite_callback is not None:
                mode = exist_overwrite_callback(output)
                if mode == 0:
                    break
                if mode == "read":
                    data = Result2D.load(output)
                    data.is_read = True
                    results.append(data)
                    continue
            elif load_existing is True:
                data = Result2D.load(output)
                data.is_read = True
                results.append(data)
                continue

        data = Result2D(
            output=str(output),
            bf=str(im0),
            input=str(im1),
            reference_stack=str(im2),
            pixel_size=float(pixel_size),
        )
        data.save()
        results.append(data)

    return results
