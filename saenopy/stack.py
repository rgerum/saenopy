import glob as glob
import re
from pathlib import Path

import imageio
import natsort
import numpy as np
import pandas as pd
import tifffile
from skimage import io

from saenopy.saveable import Saveable


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
        print("stack", template)
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
                from saenopy.gui.common.lif_reader import LifFile
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
            im = read_tiff(self.image_filenames[0][0])
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
        return get_stack(self.images)


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


def load_image_files_to_nparray(image_filenames, crop=None):
    if isinstance(image_filenames, str):
        im = read_tiff(image_filenames)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if crop is not None and "x" in crop:
            im = im[:, slice(*crop["x"])]
        if crop is not None and "y" in crop:
            im = im[slice(*crop["y"])]
        return im
    else:
        return [load_image_files_to_nparray(i, crop) for i in image_filenames]


def read_tiff(image_filenames):
    if re.match(r".*\.tiff?(\[.*\])?$", str(image_filenames)):
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


def get_stack(filename):
    if isinstance(filename, str):
        images = glob.glob(filename)
    else:
        images = list(filename)
    im = io.imread(images[0], as_gray=True)
    stack = np.zeros((im.shape[0], im.shape[1], len(images)), dtype=im.dtype)
    for i, im in enumerate(images):
        stack[:, :, i] = io.imread(im, as_gray=True)
    return stack


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
