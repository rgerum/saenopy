import numpy as np
import openpiv.filters

from openpiv.pyprocess import extended_search_area_piv
import openpiv.scaling
import openpiv.tools
import openpiv.validation


def strain_energy_points(u, v, tx, ty, pixelsize1, pixelsize2):
    pixelsize1 *= 10 ** -6
    pixelsize2 *= 10 ** -6  # conversion to m
    # u is given in pixels/minutes where a pixel is from the original image (pixelsize1)
    # tx is given in forces/pixels**2 where a pixel is from the deformation/traction field (pixelsize2)
    energy_points = 0.5 * (pixelsize2 ** 2) * (tx * u * pixelsize1 + ty * v * pixelsize1)
    # value of a background point
    bg = np.percentile(energy_points, 20)
    energy_points -= bg
    return energy_points


def get_xy_for_quiver(u):
    """
    accessory function to calculate grid for plt.quiver. Size of the array will correspond to input u.
    :param u:any array,
    :return:
    """
    xs = np.zeros(np.shape(u))
    for i in range(np.shape(u)[0]):
        xs[i, :] = np.arange(0, np.shape(u)[1], 1)
    ys = np.zeros(np.shape(u))
    for j in range(np.shape(u)[1]):  # is inverted in other skript
        ys[:, j] = np.arange(0, np.shape(u)[0], 1)
    return xs, ys


def contractillity(tx, ty, pixelsize, mask):

    """
    Calculation of contractile force and force epicenter.Contractile force is the sum of all projection of traction
    forces (in N) towards the force epicenter. The force epicenter is the point that maximizes the contractile force.
    :param tx: traction forces in x direction in Pa
    :param ty: traction forces in y direction in Pa
    :param pixelsize: pixelsize of the traction field
    :param mask: mask of which values to use for calculation
    :return: contractile_force,contractile force in N
             proj_x, projection of traction forces towards the force epicenter, x component
             proj_y, projection of traction forces towards the force epicenter, y component
             center, coordinates of the force epicenter
    """

    mask = mask.astype(bool)
    tx_filter = np.zeros(np.shape(tx))
    tx_filter[mask] = tx[mask]

    ty_filter = np.zeros(np.shape(ty))
    ty_filter[mask] = ty[mask]

    tract_abs = np.sqrt(tx_filter ** 2 + ty_filter ** 2)

    area = (pixelsize * (10 ** -6)) ** 2  # in meter
    fx = tx_filter * area  # calculating forces (in Newton) by multiplying with area
    fy = ty_filter * area

    x, y = get_xy_for_quiver(tx)
    bx = np.sum(x * (tract_abs ** 2) + fx * (tx_filter * fx + ty_filter * fy))
    by = np.sum(y * (tract_abs ** 2) + fy * (tx_filter * fx + ty_filter * fy))

    axx = np.sum(tract_abs ** 2 + fx ** 2)
    axy = np.sum(fx * fy)
    ayy = np.sum(tract_abs ** 2 + fy ** 2)
    # ayx=np.sum(tx*ty)

    A = np.array([[axx, axy], [axy, ayy]])
    b = np.array([bx, by]).T

    # solve equation system:
    # center*[bx,by]=[[axx,axy],
    #                [axy,ayy]]
    # given by A*[[1/bx],
    #             [1/by]
    center = np.matmul(np.linalg.inv(A), b)

    # vector projection to origin

    dist_x = center[0] - x
    dist_y = center[1] - y
    dist_abs = np.sqrt(dist_y ** 2 + dist_x ** 2)
    proj_abs = (fx * dist_x + fy * dist_y) / dist_abs
    contractile_force = np.nansum(proj_abs)

    # project_vectors
    proj_x = proj_abs * dist_x / dist_abs
    proj_y = proj_abs * dist_y / dist_abs

    return contractile_force, proj_x, proj_y, center  # unit of contractile force is N


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

    u, v, sig2noise = extended_search_area_piv(frame_a, frame_b, window_size=window_size,
                                               overlap=overlap,
                                               dt=1, subpixel_method="gaussian",
                                               search_area_size=window_size,
                                               sig2noise_method='peak2peak')

    u, v, mask = openpiv.validation.sig2noise_val(u, v, sig2noise, threshold=1.05)

    def_abs = np.sqrt(u ** 2 + v ** 2)
    m = np.nanmean(def_abs)
    std = np.nanstd(def_abs)

    threshold = std * std_factor + m
    mask_std = def_abs > threshold
    u[mask_std] = np.nan
    v[mask_std] = np.nan

    u, v = openpiv.filters.replace_outliers(u, v, method='localmean', max_iter=10, kernel_size=2)
    return u, -v, mask, mask_std  # return -v because of image inverted axis
