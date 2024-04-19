import numpy as np
import openpiv.filters

from openpiv.pyprocess import extended_search_area_piv
import openpiv.scaling
import openpiv.tools
import openpiv.validation


def calculate_deformation(im1, im2, window_size=64, overlap=32, std_factor=20):
    """
    Calculation of deformation field using particle image velocimetry (PIV). Recommendations: window_size
    should be about 6 time the size of bead. overlap should be no less than half of the window_size.
    Std_factor should be kept as high as possible. Make sure to check for to many exclusions caused by this factor
    e.g. by looking at the mask_std. Side note: returns -v because original v is negative if compared to coordinates
    of images (y-axis is inverted).

    :param im1: after image
    :param im2: before image
    :param window_size: integer, size of interrogation windows for PIV
    :param overlap: integer, overlap of interrogation windows for PIV
    :param std_factor: Filtering extreme outliers beyond
                       mean(deformation) + std_factor * standard deviation (deformation)
    :return:u,v deformation in x and y direction in pixel of the before and after image
            x,y positions of the deformation field in coordinates of the after and before image
            mask, mask_std  mask of filtered values by signal-to-noise filtering (piv internal) and filtering for
            extreme outliers
    """
    # accepting either path to file or image data directly
    if isinstance(im1, str):
        frame_a = np.array(openpiv.tools.imread(im1), dtype="int32")
    elif isinstance(im1, np.ndarray):
        frame_a = im1
    else:
        raise ValueError
    if isinstance(im2, str):
        frame_b = np.array(openpiv.tools.imread(im2), dtype="int32")
    elif isinstance(im2, np.ndarray):
        frame_b = im2
    else:
        raise ValueError

    u, v, sig2noise = extended_search_area_piv(
        frame_a,
        frame_b,
        window_size=window_size,
        overlap=overlap,
        dt=1,
        subpixel_method="gaussian",
        search_area_size=window_size,
        sig2noise_method="peak2peak",
    )

    u, v, mask = openpiv.validation.sig2noise_val(u, v, sig2noise, threshold=1.05)

    def_abs = np.sqrt(u**2 + v**2)
    m = np.nanmean(def_abs)
    std = np.nanstd(def_abs)

    threshold = std * std_factor + m
    mask_std = def_abs > threshold
    u[mask_std] = np.nan
    v[mask_std] = np.nan

    u, v = openpiv.filters.replace_outliers(
        u, v, method="localmean", max_iter=10, kernel_size=2
    )
    return u, -v, mask, mask_std  # return -v because of image inverted axis
