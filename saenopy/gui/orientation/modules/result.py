import numpy as np
import io
from typing import List, TypedDict, Tuple, Union
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


class SegmentationParametersDict(TypedDict, total=False):
    thresh: float
    gauss1: float
    gauss2: float
    invert: bool
    mask_sigma: float
    mask_threshold: float
    mask_invert: bool


class OrientationParametersDict(TypedDict, total=False):
    sigma_tensor: float
    sigma_tensor_type: str
    edge: int
    max_dist: Union[float, None]
    ignore_cell_outline: bool
    sigma_first_blur: float
    angle_sections: int
    shell_width: float
    shell_width_type: str
    plotting: bool
    SaveNumpy: bool
    reference_x: Union[float, None]
    reference_y: Union[float, None]
    line_x1: Union[float, None]
    line_y1: Union[float, None]
    line_x2: Union[float, None]
    line_y2: Union[float, None]


class Segmentation(TypedDict):
    mask: np.ndarray
    radius: float
    centroid: Tuple[float, float]

class GlobalResult(TypedDict):
    pass

GlobalResult = TypedDict("GlobalResult", {
    'Mean Coherency': float,
    'Mean Coherency (weighted by intensity)': float,
    'Mean Angle': float,
    'Mean Angle (weighted by coherency)': float,
    'Mean Angle (weighted by intensity and coherency)': float,
    'Orientation': float,
    'Orientation (weighted by coherency)': float,
    'Orientation (weighted by intensity and coherency)': float,
}, total=False)

class DistanceResult(TypedDict):
    pass

DistanceResult = TypedDict('DistanceResult', {
    'Shell_mid (px)': float,
    'Shell_mid (µm)': float,
    'Shell_start (px)': float,
    'Shell_end (px)': float,
    'Point alignment (shell)': float,
    'Point angular deviation (shell)': float,
    'Point coherency (shell)': float,
    'Valid pixels': int,
    'Coherency pixels': int,
    'Intensity (accumulated)': float,
    'Intensity (individual)': float,
    'Intensity Norm (individual)': float,
    'Intensity Norm (accumulated)': float,
    'Angle (accumulated)': float,
    'Angle (individual)': float,
    'Angle (individual-weightInt)': float,
    'Orientation (accumulated)': float,
    'Orientation (individual)': float,
    'Orientation (individual-weightInt)': float,
    'Intensity disttocenter (accumulated)': float,
    'Intensity disttocenter (individual)': float,
    'Intensity Norm disttocenter (individual)': float,
    'Intensity Norm disttocenter (accumulated)': float,
    'Angle disttocenter (accumulated)': float,
    'Angle disttocenter (individual)': float,
    'Orientation disttocenter (accumulated)': float,
    'Orientation disttocenter (individual)': float,
}, total=False)  # Cell and point profiles contain different sets of columns.


class ResultOrientation(Saveable):
    __save_parameters__ = ['analysis_mode', 'validity_mask', 'angle_to_y_map', 'axis_histogram', 'image_cell', 'image_fiber', 'pixel_size', 'output',
                           'segmentation_parameters', 'orientation_parameters',
                           'segmentation',
                           'orientation_map',
                           'coherence_map',
                           'angle_to_x_map',
                           'orientation_vectors',
                           'shape',
                           "results_total",
                           "results_distance",
                           '___save_name__', '___save_version__']
    ___save_name__ = "ResultOrientation"
    ___save_version__ = "1.1"

    # Missing fields in version 1.0 files are filled explicitly in from_dict.
    analysis_mode: str = "full"
    validity_mask: np.ndarray = None
    angle_to_y_map: np.ndarray = None
    axis_histogram: np.ndarray = None

    image_cell: str = None
    image_fiber: str = None
    output: str = None
    pixel_size: float = None

    segmentation_parameters: SegmentationParametersDict = None
    orientation_parameters: OrientationParametersDict = None

    segmentation: Segmentation = None

    orientation_map: np.ndarray = None
    coherence_map: np.ndarray = None
    angle_to_x_map: np.ndarray = None
    orientation_vectors: np.ndarray = None

    results_total: List[GlobalResult] = None
    results_distance: List[DistanceResult] = None

    state: bool = False

    shape: Tuple[int, int] = None

    def __init__(self, output, image_cell, image_fiber, pixel_size, **kwargs):
        self.image_cell = image_cell
        self.image_fiber = image_fiber
        self.pixel_size = pixel_size
        self.output = output

        super().__init__(**kwargs)

    @classmethod
    def from_dict(cls, data_dict):
        data_dict = data_dict.copy()
        # Older RGB previews stored (height, width, channels), although the
        # result's shape describes spatial dimensions only.
        shape = data_dict.get("shape")
        if isinstance(shape, (tuple, list, np.ndarray)) and np.ndim(shape) == 1 and len(shape) == 3:
            data_dict["shape"] = tuple(shape[:2])
        for name, default in (("analysis_mode", "full"), ("validity_mask", None),
                              ("angle_to_y_map", None), ("axis_histogram", None)):
            data_dict.setdefault(name, default)
        if data_dict.get("___save_version__") == "1.0":
            data_dict["___save_version__"] = "1.1"
        return super().from_dict(data_dict)

    def save(self, file_name=None):
        if file_name is None:
            file_name = self.output
        Path(file_name).parent.mkdir(exist_ok=True, parents=True)
        super().save(file_name)

    def on_load(self, filename: str):
        self.output = str(Path(filename))

    def get_absolute_path_cell(self):
        return None if self.image_cell is None else make_path_absolute(self.image_cell, Path(self.output).parent)

    def get_absolute_path_fiber(self):
        return make_path_absolute(self.image_fiber, Path(self.output).parent)

    @property
    def uses_mask(self):
        return self.analysis_mode == "axis_orientation_masked" or (
            self.analysis_mode in ("point_orientation", "line_orientation") and self.image_cell is not None)

    @property
    def mask_source(self):
        """Infer the mask source from saved paths, including older result files."""
        if not self.uses_mask:
            return None if self.analysis_mode == "full" else "whole_image"
        fiber = os.path.normcase(os.path.abspath(self.get_absolute_path_fiber()))
        cell = self.get_absolute_path_cell()
        return "same_image" if cell is not None and os.path.normcase(os.path.abspath(cell)) == fiber else "separate_image"

    def get_image(self, index, corrected=True):
        try:
            if index == 0 and self.image_cell is not None:
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
            self.shape = im.shape[:2]
        return im

    def get_data_structure(self):
        if self.shape is None:
            self.get_image(0)
        data = {
            "dimensions": 2,
            "z_slices_count": 1,
            "im_shape": [self.shape[0], self.shape[1], 1],
            "time_point_count": 1,
            "has_reference": False,
            "voxel_size": [self.pixel_size, self.pixel_size, 1],
            "time_delta": None,
            "channels": ["cells", "fibers"],
            "maps": {
                "orientation_cell_center": {
                    "measure": "orientation",
                    "unit": "",
                    "name": "orientation_cell_center",
                    "colormap": "coolwarm",
                    "lim": [-1, 1],
                },
                "orientation_x_axis": {
                    "measure": "angle",
                    "unit": "",
                    "name": "orientation_x_axis",
                    "colormap": "gist_rainbow",
                    "lim": [-90, 90],
                },
                "coherence": {
                    "measure": "normalized_strength",
                    "unit": "",
                    "name": "coherence",
                    "colormap": "turbo",
                    "lim": [0, 1],
                },
            },
        }

        data["maps"]["orientation_y_axis"] = dict(data["maps"]["orientation_x_axis"], name="orientation_y_axis")
        if self.analysis_mode != "full":
            data["maps"].pop("orientation_cell_center")
            for axis in ("x", "y"):
                data["maps"][f"alignment_{axis}_axis"] = {
                    "measure": "orientation", "unit": "", "name": f"alignment_{axis}_axis",
                    "colormap": "coolwarm", "lim": [-1, 1]}
        if self.analysis_mode == "point_orientation":
            data["maps"]["orientation_point"] = {
                "measure": "orientation", "unit": "", "name": "orientation_point",
                "colormap": "coolwarm", "lim": [-1, 1]}
        if self.analysis_mode == "line_orientation":
            data["maps"]["orientation_line"] = {
                "measure": "orientation", "unit": "", "name": "orientation_line",
                "colormap": "coolwarm", "lim": [-1, 1]}
            data["maps"]["angle_line"] = dict(data["maps"]["orientation_x_axis"], name="angle_line")
        if self.image_cell is None:
            data["channels"] = ["fibers"]
        elif self.uses_mask:
            data["channels"] = ["mask", "fibers"]
        return data

    def get_image_data(self, time_point, channel="default", use_reference=False):
        if channel in ("cells", "mask"):
            im = self.get_image(0)
        elif channel == "fibers":
            im = self.get_image(1)
        else:
            raise ValueError("unknown channel")
        if self.orientation_parameters is not None:
            edge = self.orientation_parameters["edge"]
            if edge:
                im = im[edge:-edge, edge:-edge]
        if len(im.shape) == 2:
            return im[:, :, None, None]
        return im[:, :, :, None]

    def get_map_data(self, name):
        if name == "orientation_line":
            return self.orientation_map if self.analysis_mode == "line_orientation" else None
        if name == "angle_line":
            if self.analysis_mode != "line_orientation" or self.angle_to_x_map is None:
                return None
            angle = (self.results_total or [{}])[0].get("Reference line angle (deg)")
            if angle is None:
                return None
            # Derive from the saved reference, never from a pending new line.
            values = (self.angle_to_x_map - angle + 90) % 180 - 90
            return np.where(self.validity_mask, values, np.nan) if self.validity_mask is not None else values
        if name == "orientation_point":
            return self.orientation_map if self.analysis_mode == "point_orientation" else None
        if name == "orientation_cell_center":
            return self.orientation_map if self.analysis_mode == "full" else None
        if name == "orientation_x_axis":
            values = self.angle_to_x_map
        elif name == "orientation_y_axis":
            values = self.angle_to_y_map
            if values is None and self.angle_to_x_map is not None:
                values = (self.angle_to_x_map + 180) % 180 - 90
        elif name == "coherence":
            values = self.coherence_map
        elif name in ("alignment_x_axis", "alignment_y_axis"):
            if self.angle_to_x_map is None:
                return None
            values = 1 - np.abs(self.angle_to_x_map) / 45
            if name == "alignment_y_axis":
                values = -values
        else:
            return None
        if values is not None and self.analysis_mode in ("axis_orientation_masked", "point_orientation", "line_orientation") and self.validity_mask is not None:
            values = np.where(self.validity_mask, values, np.nan)
        return values


def get_orientation_files(output_path, fiber_list_string, cell_list_string, pixel_size,
                    exist_overwrite_callback=None,
                    load_existing=False, analysis_mode="full"):
    if analysis_mode not in ("full", "axis_orientation_only", "axis_orientation_masked", "point_orientation", "line_orientation"):
        raise ValueError("Unknown orientation analysis mode")
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError("pixel_size must be positive and finite")
    output_base = Path(fiber_list_string).parent
    while "*" in str(output_base):
        output_base = Path(output_base).parent

    fiber_list_string = sorted(glob.glob(str(fiber_list_string)))
    output_path = str(output_path)
    if analysis_mode == "axis_orientation_only" or (analysis_mode in ("point_orientation", "line_orientation") and cell_list_string is None):
        cell_list_string = [None] * len(fiber_list_string)
    else:
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
            image_cell=None if im1 is None else str(im1),
            analysis_mode=analysis_mode,
            pixel_size=float(pixel_size),
        )
        data.save()
        results.append(data)

    return results
