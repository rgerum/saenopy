import numpy as np
import io
from typing import List
import traceback
from natsort import natsorted
import re

from tifffile import imread
import matplotlib.pyplot as plt

from saenopy.saveable import Saveable
from saenopy.result_file import make_path_absolute, make_path_relative



class Mesh2D:
    pass


class ResultSpheroid(Saveable):
    __save_parameters__ = ['template', 'images', 'output', 'pixel_size', 'time_delta',
                           'thresh_segmentation', 'continuous_segmentation',
                           'custom_mask', 'n_min', 'n_max', 'shape',

                           'piv_parameters', 'force_parameters',

                           'segmentations', 'displacements',
                            'res_data', 'res_angles',
                           '___save_name__', '___save_version__']
    ___save_name__ = "ResultSpheroid"
    ___save_version__ = "1.0"

    template: str = None
    images: list = None
    output: str = None
    state : bool = False

    pixel_size: float = None
    time_delta: float = None

    thresh_segmentation = None
    continuous_segmentation = None
    custom_mask = None
    n_min = None
    n_max = None

    piv_parameters: dict = {}
    force_parameters: dict = {}

    segmentations: list = None
    displacements: list = None

    res_data: dict = None
    res_angles: dict = None

    shape = None

    def __init__(self, template, images, output, **kwargs):
        self.template = template
        self.images = images
        self.output = str(output)
        if "shape" not in kwargs:
            kwargs["shape"] = None
        if "res_data" not in kwargs:
            kwargs["res_data"] = {}
        self.res_data = {}
        if "res_angles" not in kwargs:
            kwargs["res_angles"] = {}
        self.res_angles = {}

        super().__init__(**kwargs)

    def save(self, filename: str = None):
        if filename is None:
            self.template = make_path_absolute(self.template, Path(self.output).parent)
            for i in range(len(self.images)):
                self.images[i] = make_path_absolute(self.images[i], Path(self.output).parent)

            filename = self.output

            self.template = make_path_relative(self.template, Path(self.output).parent)
            for i in range(len(self.images)):
                self.images[i] = make_path_relative(self.images[i], Path(self.output).parent)
        Path(self.output).parent.mkdir(exist_ok=True, parents=True)
        super().save(filename)

    def on_load(self, filename: str):
        self.template = make_path_relative(self.template, Path(self.output).parent)
        for i in range(len(self.images)):
            self.images[i] = make_path_relative(self.images[i], Path(self.output).parent)

        self.output = str(Path(filename))

        self.template = make_path_absolute(self.template, Path(self.output).parent)
        for i in range(len(self.images)):
            self.images[i] = make_path_absolute(self.images[i], Path(self.output).parent)

    def get_absolute_path(self):
        return make_path_absolute(self.template, Path(self.output).parent)

    def get_data_structure(self):
        if self.shape is None:
            self.shape = list(imread(self.images[0]).shape[:2]) + [1]
        return {
            "dimensions": 2,
            "z_slices_count": 1,
            "im_shape": self.shape,
            "time_point_count": len(self.images),
            "has_reference": False,
            "voxel_size": [self.pixel_size, self.pixel_size, 1],
            "time_delta": self.time_delta,
            "channels": ["default"],
            "fields": {
                "deformation": {
                    "type": "vector",
                    "measure": "deformation",
                    "unit": "pixel",
                    "name": "displacements_measured",
                }
            }
        }

    def get_image_data(self, time_point, channel="default", use_reference=False, return_filename=False):
        try:
            if return_filename:
                return make_path_absolute(self.images[time_point], Path(self.output).parent)
            im = imread(make_path_absolute(self.images[time_point], Path(self.output).parent))
        except FileNotFoundError as err:
            traceback.print_exception(err)
            h = 255
            w = 255
            if self.shape is not None:
                h, w = self.shape[:2]
            im = np.zeros([h, w, 3], dtype=np.uint8)
            im[:, :, 0] = 255
            im[:, :, 2] = 255
        if len(im.shape) == 2:
            return im[:, :, None, None]
        return im[:, :, :, None]

    def get_field_data(self, name, time_point):
        if self.displacements is not None and time_point > 0:
            try:
                disp = self.displacements[time_point - 1]
                mesh = Mesh2D()
                mesh.units = "pixels"
                mesh.nodes = np.array([disp["x"].ravel(), disp["y"].ravel()]).T
                mesh.displacements_measured = np.array([disp["u"].ravel(), disp["v"].ravel()]).T * 1

                if mesh is not None:
                    return mesh, mesh.displacements_measured
            except IndexError:
                pass
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

import glob
from pathlib import Path
import os
def get_stacks_spheroid(input_path, output_path,
               pixel_size=None,
               time_delta=None,
               exist_overwrite_callback=None,
               load_existing=False) -> List[ResultSpheroid]:
    text = os.path.normpath(input_path)
    glob_string = text.replace("?", "*")
    files = natsorted(glob.glob(glob_string))

    output_base = glob_string
    while "*" in str(output_base):
        output_base = Path(output_base).parent

    regex_string = re.escape(text).replace(r"\*", "(.*)").replace(r"\?", ".*")

    data_dict = {}
    results = []
    for file in files:
        file = os.path.normpath(file)
        match = re.match(regex_string, file).groups()
        reconstructed_file = regex_string
        for element in match:
            reconstructed_file = reconstructed_file.replace("(.*)", element, 1)
        reconstructed_file = reconstructed_file.replace(".*", "*")
        reconstructed_file = re.sub(r'\\(.)', r'\1', reconstructed_file)

        if reconstructed_file not in data_dict:
            output = Path(output_path) / os.path.relpath(reconstructed_file.replace("*", "{t}"), output_base)
            output = output.parent / (output.stem + ".saenopySpheroid")

            if output.exists():
                if exist_overwrite_callback is not None:
                    mode = exist_overwrite_callback(output)
                    if mode == 0:
                        break
                    if mode == "read":
                        data = ResultSpheroid.load(output)
                        data.is_read = True
                        data_dict[reconstructed_file] = data
                        results.append(data)
                        continue
                elif load_existing is True:
                    data = ResultSpheroid.load(output)
                    data.is_read = True
                    data_dict[reconstructed_file] = data
                    results.append(data)
                    continue

            data_dict[reconstructed_file] = ResultSpheroid(
                template=reconstructed_file,
                images=[],
                pixel_size=pixel_size,
                time_delta=time_delta,
                output=str(output),
            )
            results.append(data_dict[reconstructed_file])
        if not getattr(data_dict[reconstructed_file], "is_read", False):
            data_dict[reconstructed_file].images.append(file)

    return results
