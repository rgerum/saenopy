import readlif.reader
import numpy as np
import mmap

def _recursive_image_find(self, tree, return_list=None, path=""):
    """Creates list of images by parsing the XML header recursively"""

    if return_list is None:
        return_list = []

    children = tree.findall("./Children/Element")
    if len(children) < 1:  # Fix for 'first round'
        children = tree.findall("./Element")
    for item in children:
        folder_name = item.attrib["Name"]
        if path == "":
            appended_path = folder_name
        else:
            appended_path = path + "/" + folder_name
        has_sub_children = len(item.findall("./Children/Element")) > 0
        is_image = (
                len(item.findall("./Data/Image/ImageDescription/Dimensions")) > 0
        )

        if has_sub_children:
            self._recursive_image_find(item, return_list, appended_path)

        elif is_image:
            # If additional XML data extraction is needed, add it here.
            # Get number of frames (time points)
            try:
                print(item.findall(
                    "./Data/Image/ImageDescription/"
                    "Dimensions/"
                    "DimensionDescription"
                ))
                dim_t = np.prod([int(p.attrib["NumberOfElements"]) for p in item.findall(
                    "./Data/Image/ImageDescription/"
                    "Dimensions/"
                    "DimensionDescription"
                ) if p.attrib["DimID"] not in ["1", "2", "3"]])
            except AttributeError:
                dim_t = 1

            # Don't need a try / except block, all images have x and y
            dim_x = int(item.find(
                "./Data/Image/ImageDescription/"
                "Dimensions/"
                "DimensionDescription"
                '[@DimID="1"]'
            ).attrib["NumberOfElements"])
            dim_y = int(item.find(
                "./Data/Image/ImageDescription/"
                "Dimensions/"
                "DimensionDescription"
                '[@DimID="2"]'
            ).attrib["NumberOfElements"])
            # Try to get z-dimension
            try:
                dim_z = int(item.find(
                    "./Data/Image/ImageDescription/"
                    "Dimensions/"
                    "DimensionDescription"
                    '[@DimID="3"]'
                ).attrib["NumberOfElements"])
            except AttributeError:
                dim_z = 1

            # Determine number of channels
            channel_list = item.findall(
                "./Data/Image/ImageDescription/Channels/ChannelDescription"
            )

            n_channels = int(len(channel_list))

            # Iterate over each channel, get the resolution
            bit_depth = tuple([int(c.attrib["Resolution"]) for
                               c in channel_list])

            # Find the scale of the image. All images have x and y,
            # only some have z and t.
            # It is plausible that 'Length' is not defined - use try/except.
            try:
                len_x = item.find(
                    "./Data/Image/ImageDescription/"
                    "Dimensions/"
                    "DimensionDescription"
                    '[@DimID="1"]'
                ).attrib["Length"]  # Returns len in meters
                scale_x = (int(dim_x) - 1) / (float(len_x) * 10 ** 6)
            except (AttributeError, ZeroDivisionError):
                scale_x = None

            try:
                len_y = item.find(
                    "./Data/Image/ImageDescription/"
                    "Dimensions/"
                    "DimensionDescription"
                    '[@DimID="2"]'
                ).attrib["Length"]  # Returns len in meters
                scale_y = (int(dim_y) - 1) / (float(len_y) * 10 ** 6)
            except (AttributeError, ZeroDivisionError):
                scale_y = None

            # Try to get z-dimension
            try:
                len_z = item.find(
                    "./Data/Image/ImageDescription/"
                    "Dimensions/"
                    "DimensionDescription"
                    '[@DimID="3"]'
                ).attrib["Length"]  # Returns len in meters
                scale_z = int(dim_z) / (float(len_z) * 10 ** 6)
            except (AttributeError, ZeroDivisionError):
                scale_z = None

            try:
                len_t = item.find(
                    "./Data/Image/ImageDescription/"
                    "Dimensions/"
                    "DimensionDescription"
                    '[@DimID="4"]'
                ).attrib["Length"]  # Returns len in meters
                scale_t = int(dim_t) / float(len_t)
            except (AttributeError, ZeroDivisionError):
                scale_t = None

            data_dict = {
                "dims": (dim_x, dim_y, dim_z, dim_t),
                "path": str(path + "/"),
                "name": item.attrib["Name"],
                "channels": n_channels,
                "scale": (scale_x, scale_y, scale_z, scale_t),
                "bit_depth": bit_depth
            }

            return_list.append(data_dict)

    return return_list


def __item__(self, slices):
    # nt x nz x channels x height x width (x 3)
    if np.all(np.array(self.bit_depth) <= 8):
        dtype = np.uint8
    elif np.all((np.array(self.bit_depth) > 8) & (np.array(self.bit_depth) <= 16)):
        dtype = np.uint16
    elif np.all((np.array(self.bit_depth) > 16) & (np.array(self.bit_depth) <= 32)):
        dtype = np.uint32

    with open(self.filename, 'rb') as f, mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
        data = np.ndarray(buffer=mm, dtype=dtype, offset=self.offsets[0], strides=1, shape=(self.offsets[1],))
        data = data.reshape(self.nt, self.nz, self.channels, self.dims[0], self.dims[1])
        data = data[slices].copy()
    return data


readlif.reader.LifImage.__getitem__ = __item__
readlif.reader.LifFile._recursive_image_find = _recursive_image_find