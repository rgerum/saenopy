import numpy as np


def strain_energy_points(u, v, tx, ty, pixel_size1, pixel_size2):
    pixel_size1 *= 10 ** -6
    pixel_size2 *= 10 ** -6  # conversion to m
    # u is given in pixels/minutes where a pixel is from the original image (pixel_size1)
    # tx is given in forces/pixels**2 where a pixel is from the deformation/traction field (pixel_size2)
    energy_points = 0.5 * (pixel_size2 ** 2) * (tx * u * pixel_size1 + ty * v * pixel_size1)
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


def contractility(tx, ty, pixel_size, mask):
    """
    Calculation of contractile force and force epicenter.Contractile force is the sum of all projection of traction
    forces (in N) towards the force epicenter. The force epicenter is the point that maximizes the contractile force.
    :param tx: traction forces in x direction in Pa
    :param ty: traction forces in y direction in Pa
    :param pixel_size: pixel size of the traction field
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

    tract_abs = np.sqrt(tx_filter**2 + ty_filter**2)

    area = (pixel_size * (10 ** -6)) ** 2  # in meter
    fx = tx_filter * area  # calculating forces (in Newton) by multiplying with area
    fy = ty_filter * area

    x, y = get_xy_for_quiver(tx)
    bx = np.sum(x * (tract_abs**2) + fx * (tx_filter * fx + ty_filter * fy))
    by = np.sum(y * (tract_abs**2) + fy * (tx_filter * fx + ty_filter * fy))

    axx = np.sum(tract_abs**2 + fx**2)
    axy = np.sum(fx * fy)
    ayy = np.sum(tract_abs**2 + fy**2)
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
    dist_abs = np.sqrt(dist_y**2 + dist_x**2)
    proj_abs = (fx * dist_x + fy * dist_y) / dist_abs
    contractile_force = np.nansum(proj_abs)

    # project_vectors
    proj_x = proj_abs * dist_x / dist_abs
    proj_y = proj_abs * dist_y / dist_abs

    return contractile_force, proj_x, proj_y, center  # unit of contractile force is N
