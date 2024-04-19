import numpy as np
from saenopy.pyTFM.TFM_tractions import TFM_tractions


def calculate_forces(u, v, pixel_size, shape, h, young, sigma):
    ps1 = pixel_size  # pixel size of the image of the beads
    # dimensions of the image of the beads
    im1_shape = shape
    ps2 = ps1 * np.mean(
        np.array(im1_shape) / np.array(u.shape)
    )  # pixel size of the deformation field
    tx, ty = TFM_tractions(
        u, v, pixel_size1=ps1, pixel_size2=ps2, h=h, young=young, sigma=sigma
    )
    return tx, ty
