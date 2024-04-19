import numpy as np
from scipy.ndimage import binary_fill_holes

from saenopy.pyTFM.TFM_functions import strain_energy_points, contractility
from saenopy.pyTFM.calculate_stress_imports.mask_interpolation import mask_interpolation


def calculate_strain_energy(mask, pixel_size, shape, u, v, tx, ty):
    mask = binary_fill_holes(
        mask == 1
    )  # the mask should be a single patch without holes
    # changing the masks dimensions to fit to the deformation and traction fields
    mask = mask_interpolation(mask, dims=u.shape)
    ps1 = pixel_size  # pixel size of the image of the beads
    # dimensions of the image of the beads
    ps2 = ps1 * np.mean(
        np.array(shape) / np.array(u.shape)
    )  # pixel size of the deformation field
    # strain energy:
    # first we calculate a map of strain energy
    energy_points = strain_energy_points(u, v, tx, ty, ps1, ps2)  # J/pixel

    # then we sum all energy points in the area defined by mask
    strain_energy = np.sum(energy_points[mask])  # 2.14*10**-13 J
    # contractility
    contractile_force, proj_x, proj_y, center = contractility(
        tx, ty, ps2, mask
    )  # 2.03*10**-6 N

    return {
        "contractility": contractile_force,
        "area Traction Area": np.sum(mask) * ((pixel_size * 10**-6) ** 2),
        "strain energy": strain_energy,
        "center of object": center,
    }
