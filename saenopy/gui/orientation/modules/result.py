import numpy as np
import io
from typing import List, TypedDict, Tuple
import traceback
from natsort import natsorted
import re

from tifffile import imread
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import os

from saenopy.gui.tfm2d.modules.result import read_tiff
from saenopy.saveable import Saveable
from saenopy.result_file import make_path_absolute, make_path_relative



class ResultOrientation(Saveable):
    __save_parameters__ = ['image_cell', 'image_fiber', 'pixel_size', 'output',
                           '___save_name__', '___save_version__']
    ___save_name__ = "ResultOrientation"
    ___save_version__ = "1.0"

    image_cell: str = None
    image_fiber: str = None
    output: str = None
    pixel_size: float = None

    piv_parameters: dict = {}

    state: bool = False

    shape: Tuple[int, int] = None

    def __init__(self, output, image_cell, image_fiber, pixel_size, **kwargs):
        self.image_cell = image_cell
        self.image_fiber = image_fiber
        self.pixel_size = pixel_size
        self.output = output

        super().__init__(**kwargs)

    def save(self, file_name=None):
        if file_name is None:
            file_name = self.output
        Path(self.output).parent.mkdir(exist_ok=True, parents=True)
        super().save(file_name)

    def on_load(self, filename: str):
        self.output = str(Path(filename))

    def get_absolute_path_cell(self):
        return make_path_absolute(self.image_cell, Path(self.output).parent)

    def get_absolute_path_fiber(self):
        return make_path_absolute(self.image_fiber, Path(self.output).parent)

    def get_image(self, index, corrected=True):
        try:
            if index == 0:
                im = read_tiff(self.get_absolute_path_cell())
            else:
                im = read_tiff(self.get_absolute_path_fiber())
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

    def get_data_structure(self):
        if self.shape is None:
            self.get_image(0)
        return {
            "dimensions": 2,
            "z_slices_count": 1,
            "im_shape": [self.shape[0], self.shape[1], 1],
            "time_point_count": 1,
            "has_reference": False,
            "voxel_size": [self.pixel_size, self.pixel_size, 1],
            "time_delta": None,
            "channels": ["cells", "fibers"],
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
            im = self.get_image(0)
        else:
            im = self.get_image(1)
        if len(im.shape) == 2:
            return im[:, :, None, None]
        return im[:, :, :, None]

    def get_field_data(self, name, time_point):
        class Mesh2D:
            pass

        return None, None

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


def get_orientation_files(output_path, fiber_list_string, cell_list_string, pixel_size,
                    exist_overwrite_callback=None,
                    load_existing=False):
    output_base = Path(fiber_list_string).parent
    while "*" in str(output_base):
        output_base = Path(output_base).parent

    fiber_list_string = sorted(glob.glob(str(fiber_list_string)))
    output_path = str(output_path)
    cell_list_string = sorted(glob.glob(str(cell_list_string)))

    if len(fiber_list_string) == 0:
        raise ValueError("no fiber image selected")
    if len(cell_list_string) == 0:
        raise ValueError("no cell image selected")

    if len(fiber_list_string) != len(cell_list_string):
        raise ValueError(f"the number of fiber images ({len(fiber_list_string)}) does not match the number of cell images {len(cell_list_string)}")

    results = []
    for i in range(len(fiber_list_string)):
        im0 = fiber_list_string[i]
        im1 = cell_list_string[i]

        output = Path(output_path) / os.path.relpath(im0, output_base)
        output = output.parent / output.stem
        output = Path(str(output) + ".saenopyOrientation")

        if output.exists():
            if exist_overwrite_callback is not None:
                mode = exist_overwrite_callback(output)
                if mode == 0:
                    break
                if mode == "read":
                    data = ResultOrientation.load(output)
                    data.is_read = True
                    results.append(data)
                    continue
            elif load_existing is True:
                data = ResultOrientation.load(output)
                data.is_read = True
                results.append(data)
                continue

        data = ResultOrientation(
            output=str(output),
            image_fiber=str(im0),
            image_cell=str(im1),
            pixel_size=float(pixel_size),
        )
        data.save()
        results.append(data)

    return results
