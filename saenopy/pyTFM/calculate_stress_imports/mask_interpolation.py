import numpy as np
from scipy.ndimage import binary_closing
from skimage.morphology import (
    remove_small_holes,
    remove_small_objects,
)


def mask_interpolation(mask, dims, min_cell_size=100, dtype=bool):
    #
    # some pre clean up of the mask
    mask = remove_small_holes(mask.astype(bool), min_cell_size)
    mask = remove_small_objects(mask.astype(bool), 1000)  # removing other small bits
    # note: remove_small_objects labels automatically if mask is bool
    coords = np.array(np.where(mask)).astype(float)  # coordinates of all points
    interpol_factors = np.array([dims[0] / mask.shape[0], dims[1] / mask.shape[1]])
    coords[0] = coords[0] * interpol_factors[0]  # interpolating x coordinates
    coords[1] = coords[1] * interpol_factors[1]  # interpolating xy coordinates
    coords = np.round(coords).astype(int)

    coords[0, coords[0] >= dims[0]] = (
        dims[0] - 1
    )  # fixing issue when interpolated object is just at the image border
    coords[1, coords[1] >= dims[1]] = dims[1] - 1

    mask_int = np.zeros(dims)
    mask_int[coords[0], coords[1]] = 1
    mask_int = mask_int.astype(int)
    # filling gaps if we interpolate upwards
    if dims[0] * dims[1] >= mask.shape[0] * mask.shape[1]:
        iter = int(
            np.ceil(np.max([mask.shape[0] / dims[0], mask.shape[0] / dims[0]])) * 5
        )  # times 5 is safety factor
        mask_int = binary_closing(mask_int, iterations=10)
        print(iter)
    return mask_int.astype(bool)
