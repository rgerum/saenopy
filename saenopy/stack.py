import re
from pathlib import Path

import natsort
import numpy as np
import pandas as pd

import tifffile
import imageio

from saenopy.saveable import Saveable


class Stack(Saveable):
    __save_parameters__ = ['template', 'voxel_size', 'crop', '_shape',
                           'image_filenames', 'channels', # 'leica_file',
                           'packed_files']
    parent = None

    template: str = None
    voxel_size: tuple = None
    crop: dict = None

    _shape = None

    image_filenames: list = None
    leica_file = None
    channels: list = None

    packed_files: list = None

    def __init__(self, template: str, voxel_size: tuple, crop: dict = None, **kwargs):
        # reconstruct the stack savable
        super().__init__(template=template, voxel_size=voxel_size, crop=crop, **kwargs)
        # if the stack has not been initialized
        if self.image_filenames is None:
            # check if the template is a leica file
            match = re.match(r"(.*)\{f\:(\d*)\}\{c\:(\d*)\}(?:\{t\:(\d*)\})?.lif", template)
            if match:
                from saenopy.gui.common.lif_reader import LifFile
                self.leica_filename, self.leica_folder, self.leica_channel, self.leica_time = match.groups()
                if self.leica_time is None:  # pragma: no cover
                    self.leica_time = 0
                else:
                    self.leica_time = int(self.leica_time)
                self.leica_channel = int(self.leica_channel)
                self.leica_file = LifFile(self.leica_filename + ".lif").get_image(self.leica_folder)
                self.channels = [str(self.leica_channel)]
                for i in range(0, self.leica_file.channels):
                    if i != self.leica_channel:
                        self.channels.append(str(i))
            # or a tiff file
            else:
                self.image_filenames, self.channels = template_to_array(template, crop)

    def paths_relative(self, parent):
        self.parent = parent

        def normalize_path(template, output):
            template = str(Path(template).absolute())
            output = str(Path(output).absolute())
            # relative and optionally go up to two folders up
            try:
                template = Path(template).relative_to(output)
            except ValueError:
                try:
                    template = Path("..") / Path(template).relative_to(Path(output).parent)
                except ValueError:
                    try:
                        template = Path("..") / ".." / Path(template).relative_to(Path(output).parent.parent)
                    except ValueError:
                        pass
            return str(template)

        def process(image):
            if isinstance(image, list):
                return [process(i) for i in image]
            return normalize_path(image, Path(self.parent.output).parent)

        self.template = process(self.template)
        #self.parent.template = process(self.parent.template)

        if self.image_filenames is not None:
            self.image_filenames = process(self.image_filenames)
        else:
            self.leica_filename = process(self.leica_filename)

    def paths_absolute(self):
        def normalize_path(template, output):
            if not Path(template).is_absolute():
                return str(Path(output).absolute() / template)
            return str(Path(template).absolute())

        def process(image):
            if isinstance(image, list):
                return [process(i) for i in image]
            return normalize_path(image, Path(self.parent.output).parent)

        self.template = process(self.template)
        if self.image_filenames is not None:
            self.image_filenames = process(self.image_filenames)
        else:
            self.leica_filename = process(self.leica_filename)

    def pack_files(self):
        images = np.array(self.image_filenames)[:, :]
        images = np.asarray(load_image_files_to_nparray(images, self.crop, self.parent)).T

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
            filename = self.image_filenames[0][0]
            if not Path(filename).is_absolute() and self.parent is not None:
                filename = Path(self.parent.output).parent / filename
            im = read_tiff(filename)
            if self.crop is not None and "x" in self.crop:
                im = im[:, slice(*self.crop["x"])]
            if self.crop is not None and "y" in self.crop:
                im = im[slice(*self.crop["y"])]
            self._shape = tuple(list(im.shape[:2]) + list(np.array(self.image_filenames).shape))
        return self._shape

    def get_image(self, z, channel):
        return self[:, :, :, z, channel].squeeze()

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
            images = images.transpose(1, 2, 0)[:, :, None, :]
            if isinstance(index[3], int):
                images = images[:, :, :, 0]
            return images[index[0], index[1], index[2]]
        if self.packed_files is None:
            images = np.array(self.image_filenames)[index[3], index[4]]
            images = np.asarray(load_image_files_to_nparray(images, self.crop, self.parent)).T
        else:
            images = self.packed_files[:, :, :, index[4], index[3]]
        images = np.swapaxes(images, 0, 2)
        return images[index[0], index[1], index[2]]

    def __array__(self) -> np.ndarray:
        return self[:, :, :, :, 0]


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


def load_image_files_to_nparray(image_filenames, crop=None, parent=None):
    if isinstance(image_filenames, str):
        # make relative paths relative to the .saenopy file
        if not Path(image_filenames).is_absolute() and parent is not None:
            image_filenames = str(Path(parent.output).absolute().parent / image_filenames)
        im = read_tiff(image_filenames)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if crop is not None and "x" in crop:
            im = im[:, slice(*crop["x"])]
        if crop is not None and "y" in crop:
            im = im[slice(*crop["y"])]
        return im
    else:
        return [load_image_files_to_nparray(i, crop, parent) for i in image_filenames]


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
    return im


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
                group["template"] = template_name + "[" + str(page) + "]"
            file_list.append(group)
        except ValueError:
            with tifffile.TiffReader(file) as tif:
                group["template"] = template_name + "[" + str(page) + "]"
                for i in range(len(tif.pages)):
                    group["filename"] = file+f"[{i}]"
                    group[page] = i
                    file_list.append(group.copy())
    return pd.DataFrame(file_list), output_base
