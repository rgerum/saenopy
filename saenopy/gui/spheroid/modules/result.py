import io
from typing import List

import matplotlib.pyplot as plt
from saenopy.saveable import Saveable
import numpy as np
from tifffile import imread
from pyTFM.plotting import show_quiver
from saenopy.result_file import make_path_absolute
from natsort import natsorted
import re


class ResultSpheroid(Saveable):
    __save_parameters__ = ['template', 'images', 'output', 'pixel_size',
                           'thresh_segmentation', 'continuous_segmentation',
                           'custom_mask', 'n_min', 'n_max',

                           'piv_parameters', 'force_parameters',

                           'segmentations', 'displacements',
                            'res_data', 'res_angles',
                           '___save_name__', '___save_version__']
    ___save_name__ = "ResultSpheroid"
    ___save_version__ = "1.0"

    template: str = None
    images: list = None
    output: str = None

    pixel_size: float = None

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

    def __init__(self, template, images, output, **kwargs):
        self.template = template
        self.images = images
        self.output = str(output)
        if "res_dict" not in kwargs:
            kwargs["res_dict"] = {}
        self.res_dict = {}
        if "res_angles" not in kwargs:
            kwargs["res_angles"] = {}
        self.res_angles = {}

        super().__init__(**kwargs)
        print(self.pixel_size)

    def save(self, file_name=None):
        if file_name is None:
            file_name = self.output
        Path(self.output).parent.mkdir(exist_ok=True, parents=True)
        super().save(file_name)

    def get_absolute_path(self):
        return make_path_absolute(self.template, Path(self.output).parent)



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
               exist_overwrite_callback=None,
               load_existing=False) -> List[ResultSpheroid]:
    text = os.path.normpath(input_path)
    glob_string = text.replace("?", "*")
    # print("globbing", glob_string)
    files = natsorted(glob.glob(glob_string))

    output_base = glob_string
    while "*" in str(output_base):
        output_base = Path(output_base).parent

    regex_string = re.escape(text).replace(r"\*", "(.*)").replace(r"\?", ".*")

    data_dict = {}
    results = []
    for file in files:
        file = os.path.normpath(file)
        #print(file, regex_string)
        match = re.match(regex_string, file).groups()
        reconstructed_file = regex_string
        for element in match:
            reconstructed_file = reconstructed_file.replace("(.*)", element, 1)
        reconstructed_file = reconstructed_file.replace(".*", "*")
        reconstructed_file = re.sub(r'\\(.)', r'\1', reconstructed_file)

        if reconstructed_file not in data_dict:
            output = Path(output_path) / os.path.relpath(file, output_base)
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
                output=str(output),
            )
            results.append(data_dict[reconstructed_file])
        if not getattr(data_dict[reconstructed_file], "is_read", False):
            data_dict[reconstructed_file].images.append(file)

    return results
