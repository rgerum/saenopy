import io
import matplotlib.pyplot as plt
from saenopy.saveable import Saveable
import numpy as np
from tifffile import imread
from pyTFM.plotting import show_quiver


class Result2D(Saveable):
    __save_parameters__ = ['bf', 'input', 'reference_stack', 'output', 'pixel_size', 'u', 'v', 'mask_val', 'mask_std',
                           'tx', 'ty', 'fx', 'fy',
                           'shape', 'mask',
                           'piv_parameters', 'force_parameters',
                           '___save_name__', '___save_version__']
    ___save_name__ = "Result2D"
    ___save_version__ = "1.0"

    input: str = None
    reference_stack: str = None
    output: str = None
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

    im_displacement: np.ndarray = None
    im_force: np.ndarray = None
    im_tension: np.ndarray = None

    def __init__(self, output, bf, input, reference_stack, pixel_size, **kwargs):
        self.bf = bf
        self.input = input
        self.reference_stack = reference_stack
        self.pixel_size = pixel_size
        self.output = output

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
        if index == 0:
            if corrected:
                try:
                    im = imread(self.input_corrected)
                except FileNotFoundError:
                    im = imread(self.input)
            else:
                im = imread(self.input)
            print(im.shape, self.input)
        elif index == -1:
            im = imread(self.bf)
        else:
            if corrected:
                try:
                    im = imread(self.reference_stack_corrected)
                except FileNotFoundError:
                    im = imread(self.reference_stack)
            else:
                im = imread(self.reference_stack)
            print(im.shape, self.reference_stack)
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

    def save(self, file_name=None):
        if file_name is None:
            file_name = self.output
        Path(self.output).parent.mkdir(exist_ok=True, parents=True)
        super().save(file_name)


def fig_to_numpy(fig1, shape):
    fig1.axes[0].set_position([0, 0, 1, 1])
    fig1.axes[1].set_position([1, 1, 0.1, 0.1])
    fig1.set_dpi(100)
    fig1.set_size_inches(shape[1] / 100, shape[0] / 100)
    with io.BytesIO() as buff:
        plt.savefig(buff, format="png")
        buff.seek(0)
        return plt.imread(buff)

import glob
from pathlib import Path
import os
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

        print("output", output)
        print("im0", im0)
        print("im1", im1)
        print("im2", im2)
        print("pixel_size", pixel_size)
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
