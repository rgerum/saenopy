import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.interpolate import splev
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_erosion


def exclude_by_key(d, ex_list):
    ex_dict = {key: values for key, values in d.items() if key not in ex_list}
    return ex_dict


def normal_vectors_from_splines(u, tck):
    """
    computes the normal vector from a spline representation of a curve at the points defined by parameters u.
    The normal vector is the derivative, with [dx/du,dy/du]-->[dy/du,-dx/du].
    (Derivative is parallel vector to spline curve)
    :param u: parameters defining on which points to evaluate the splines
    :param tck: spline
    :return: n_vectors, list of normal vectors corresponding to the points defined by u
    """

    n_vectors = np.array(
        splev(u, tck, der=1)
    ).T  # evaluation of the spline derivative, only at the points inx and y
    # (returns result as x,y value array
    n_vectors[:, [1, 0]] = n_vectors[
        :, [0, 1]
    ]  # normal vectors by switching x andy and tacking negative x values
    n_vectors[:, 1] *= -1
    n_vectors = n_vectors / np.linalg.norm(n_vectors, axis=1)[:, None]  # normalizing
    return n_vectors


def line_tension(
    lines_splines, line_lengths, stress_tensor, pixel_length, interpol_factor=1
):
    """
    function to perform interpolation on lines, to get new x,y coordinates, calculate the normal vectors on these points
    and calculate the stress vector and the norm of the stress vectors, across the lines at the interpolated points.
    Also performs interpolation of the stress tensor

    :param lines_splines: dictionary with lines_id:spline interpolation of the line from scipy.interpolate.splprep.
    Can be used with scipy.interpolate.splev to interpolate new points and also get the derivative in these points.
    :param stress_tensor: complete cauchy stress tensor
    :param pixel_length: length of unit of the x y coordinates in µm
    :param interpol_factor: Defines the number of points for interpolation with n= len(original line)*interpol_factor
    :return:lines_interpol: dictionary containing new points, normal vectors, stress vectors and the
    norm of the stress vectors  from the interpolation
            min_v,max_v: maximal and minimal values of the stress_vector norm. This is used to get a uniform
            color bar when plotting later.

    Parameters
    ----------
    line_lengths
    """

    # interpolating the stress vector:
    pixx = np.linspace(
        0, 1, stress_tensor.shape[1]
    )  # coordinate space on which to perform the interpolation
    pixy = np.linspace(0, 1, stress_tensor.shape[0])

    # using 2-dimensional interpolation on each component
    # this returns a continuous function f(x,y)=z, that can be evaluated at any point x,y
    sig_xx_inter = RectBivariateSpline(pixx, pixy, stress_tensor[:, :, 0, 0].T)
    sig_yx_inter = RectBivariateSpline(pixx, pixy, stress_tensor[:, :, 1, 0].T)
    sig_yy_inter = RectBivariateSpline(pixx, pixy, stress_tensor[:, :, 1, 1].T)

    # factors to translate from the interpolation range back to actual coordinates of the 2D-array
    inter_ranges = (stress_tensor.shape[1], stress_tensor.shape[0])

    min_v = 0
    max_v = 0
    lines_interpol = {}

    for i in lines_splines.keys():
        # getting more points:
        tck = lines_splines[i]
        len_u = line_lengths[i] * interpol_factor  # number of points to be interpolated
        u_new = np.linspace(0, 1, len_u)  # new points in the interpolation range
        x_new, y_new = splev(u_new, tck, der=0)  # interpolating new x,y points
        p_new = np.vstack([x_new, y_new]).T  # convenient form for new points

        n_vecs = normal_vectors_from_splines(
            u_new, tck
        )  # new normal_vectors, using the derivative at interpolation points

        # traction vectors using interpolation of the stress tensor
        t_vecs, t_norm, t_vecs_n, t_vecs_shear = (
            stress_vector_from_tensor_interpolation(
                p_new, n_vecs, sig_xx_inter, sig_yx_inter, sig_yy_inter, inter_ranges
            )
        )

        # conversion to N/m (height not included)
        t_vecs, t_norm, t_vecs_n, t_vecs_shear = [
            x / (pixel_length * 10**-6)
            for x in [t_vecs, t_norm, t_vecs_n, t_vecs_shear]
        ]

        # updating minimal value to find global minimum eventually
        min_v = np.min([min_v, np.nanmin(t_norm)])
        max_v = np.max([max_v, np.nanmax(t_norm)])

        lines_interpol[i] = {
            "points_new": p_new,  # array of new points from interpolation
            "t_vecs": t_vecs,  # stress vectors at the new points
            "t_norm": t_norm,  # norm of stress vectors at the new points
            "n_vecs": n_vecs,  # normal vectors at the new points
            "t_normal": t_vecs_n,  # normal component of stress vectors at the new points
            "t_shear": t_vecs_shear,
        }  # shear component of stress vectors at the new points

    return lines_interpol, min_v, max_v


def stress_vector_from_tensor_interpolation(
    ps, n_vecs, sig_xx_inter, sig_yx_inter, sig_yy_inter, inter_ranges
):
    """
    calculates the stress vector with t=sigma*n ; sigma: stress tensor, n: normal vector to the cut over which the
    stress tensor is calculated. This function uses 2-interpolation functions for the individual stress components,
    to calculate the stress vector at any point x,y.

    :param ps: list of xy(!) coordinates of points. (reversed orientation as compared to some other points lists)
    :param n_vecs: list of normal vectors at each point x,y (usually the result from spline interpolation)
    :param sig_xx_inter: 2d interpolation function of the xx-normal stress components generated from
    scipy.interpolation.interp2d . The function has been generated on the intervals [0,1](x) and [0,1](y)
    corresponding to the actual pixel indices [0,inter_ranges[0]] and[0,inter_ranges[1]]. (see below)
    :param sig_yx_inter: see above
    :param sig_yy_inter: see above
    :param inter_ranges: real pixel range referring to the interpolation interval.
    :return: t_vecs: list of stress vectors at x,y coordinates of the input x,y values
            t_norm: norm of these vectors
    """

    u = copy.deepcopy(
        ps
    )  # points represented on the (0,1) interval used for the interpolation function.
    u[:, 0] = u[:, 0] / inter_ranges[0]
    u[:, 1] = u[:, 1] / inter_ranges[1]
    t_vecs = []
    for i in range(len(u)):
        # stress vectors according to cauchy theorem
        t_vec = [
            sig_xx_inter(u[i][0], u[i][1]) * n_vecs[i][0]
            + sig_yx_inter(u[i][0], u[i][1]) * n_vecs[i][1],
            sig_yx_inter(u[i][0], u[i][1]) * n_vecs[i][0]
            + sig_yy_inter(u[i][0], u[i][1]) * n_vecs[i][1],
        ]
        t_vecs.append(t_vec)

    t_vecs = np.array(t_vecs).squeeze()  # removing unused array dimension
    t_norm = np.linalg.norm(t_vecs, axis=1)  # calculating norm of the stress vector
    t_vec_n = np.abs(
        np.sum(t_vecs * n_vecs, axis=1)
    )  # length of the normal component of the line tension
    t_vec_shear = np.sqrt(t_norm**2 - t_vec_n**2)  # shear component of the line tension
    return t_vecs, t_norm, t_vec_n, t_vec_shear


def calculate_stress_tensor(s_nodes, nodes, dims=None):
    if not dims:
        stress_tensor = np.zeros(
            (int(np.sqrt(len(s_nodes))), int(np.sqrt(len(s_nodes))), 2, 2)
        )
    else:
        stress_tensor = np.zeros((dims[0], dims[1], 2, 2))
    stress_tensor[nodes[:, 2].astype(int), nodes[:, 1].astype(int), 0, 0] = s_nodes[
        :, 0
    ]  # sigma_x
    stress_tensor[nodes[:, 2].astype(int), nodes[:, 1].astype(int), 1, 1] = s_nodes[
        :, 1
    ]  # sigma_y
    stress_tensor[nodes[:, 2].astype(int), nodes[:, 1].astype(int), 1, 0] = s_nodes[
        :, 2
    ]  # sigma_yx
    stress_tensor[nodes[:, 2].astype(int), nodes[:, 1].astype(int), 0, 1] = s_nodes[
        :, 2
    ]  # sigma_xy
    return stress_tensor


def coefficient_of_variation(mask, x, border_pad=0):
    # leave some space away from the edge
    mask_cp = copy.deepcopy(mask)
    if border_pad > 0:
        mask_cp = binary_erosion(mask_cp, iterations=border_pad)
    return np.nanstd(x[mask_cp]) / np.abs(
        np.nanmean(x[mask_cp])
    )  # absolute value of mean is an alternative definition


def all_stress_measures(st, px_size=1):
    sig_x = st[:, :, 0, 0]  # normal stress in x direction
    sig_y = st[:, :, 1, 1]  # normal stress in x direction
    tau_xy = st[:, :, 0, 1]  # shear stress

    # principal (normal stresses)
    sigma_max = (sig_x + sig_y) / 2 + np.sqrt(((sig_x - sig_y) / 2) ** 2 + tau_xy**2)
    sigma_min = (sig_x + sig_y) / 2 - np.sqrt(((sig_x - sig_y) / 2) ** 2 + tau_xy**2)
    sigma_max_abs = np.maximum(np.abs(sigma_max), np.abs(sigma_min))
    # maximum shear stress
    tau_max = np.sqrt(((sig_x - sig_y) / 2) ** 2 + tau_xy**2)
    # angle of maximal principal stress
    phi_n = np.arctan(2 * tau_xy / (sig_x - sig_y)) / 2
    # angel of maximal shear stress
    phi_shear = np.arctan(-(sig_x - sig_y) / (2 * tau_xy)) / 2
    # side note:  (phi_n-phi_shear) = pi/4 should always hold
    sigma_mean = (sigma_max + sigma_min) / 2  # mean normal stress

    return (
        sigma_max / px_size,
        sigma_min / px_size,
        sigma_max_abs / px_size,
        tau_max / px_size,
        phi_n,
        phi_shear,
        sigma_mean / px_size,
    )


def reorder_vectors_inward(
    borders, lines_interpol, cell_id, line_ids, plot_n_vecs=False, mask_boundaries=None
):
    """
    reorientation of normal and traction force vectors, so that the normal vectors of one cell all point inwards.
    :param borders:
    :param lines_interpol:
    :param cell_id:
    :param line_ids:
    :param plot_n_vecs:
    :param mask_boundaries:
    :return:
    """
    cell_area = borders.cells_area[cell_id]
    n_vectors = {
        line_id: lines_interpol[line_id]["n_vecs"] for line_id in line_ids
    }  # extracting relevant normal vectors
    t_vectors = {
        line_id: lines_interpol[line_id]["t_vecs"] for line_id in line_ids
    }  # extracting relevant normal vectors
    points_dict = {
        line_id: lines_interpol[line_id]["points_new"] for line_id in line_ids
    }  # extracting relevant normal vectors

    for l_id in line_ids:
        nps1 = np.round(points_dict[l_id] + n_vectors[l_id]).astype(
            int
        )  # predict points original orientation
        nps2 = np.round(points_dict[l_id] - n_vectors[l_id]).astype(
            int
        )  # reversed orientation
        s1 = np.sum(cell_area[nps1[:, 1], nps1[:, 0]])
        s2 = np.sum(cell_area[nps2[:, 1], nps2[:, 0]])
        change_orientation = s1 < s2
        n_vectors[l_id] *= -(
            change_orientation * 2 - 1
        )  # changes orientation inwards or keep it
        t_vectors[l_id] *= -(change_orientation * 2 - 1)  # change t vector accordingly

    # confirmation of correct vector orientation
    if plot_n_vecs:
        plt.figure()
        if isinstance(mask_boundaries, np.ndarray):
            plt.imshow(mask_boundaries, cmap="jet")
        # plotting all points with line id as label
        for length, ps in points_dict.items():
            plt.plot(ps[:, 0], ps[:, 1])  # plotting all points
        for points, n_vecs in zip(points_dict.values(), n_vectors.values()):
            for n_vec, p in zip(n_vecs, points):
                plt.arrow(p[0], p[1], n_vec[0], n_vec[1], head_width=0.15)
    return n_vectors, t_vectors


def normal_and_shear(t_vecs, n_vecs):
    tn = np.sum(t_vecs * n_vecs, axis=1)  # dot product of t vec and n vec
    # remaining component by triangle equation (Pythagorean theorem)
    ts = np.sqrt(np.sum(t_vecs * t_vecs, axis=1) - tn**2)
    return tn, ts


def mean_stress_vector_norm(
    lines_interpolation,
    borders,
    exclude_colony_edge=True,
    norm_level="points",
    vtype="t_norm",
):
    """
    average norm of the stress vector.First some vectors are averaged. Averaging can be eft out (level="points"),
    for each line(level="lines") or for each cell (level="cells")

    Taking the norm can happen on the level of single points,
    lines, or cells.
    :param lines_interpolation: Fine interpolation of points, and stresses for each line segments
    :param borders: Cells_and_Lines object
    :param exclude_colony_edge: set weather to exclude the edges of the colony from calculations. You should do this, as
    these edges have no physical meaning.
    :param norm_level: where to take the norm. Values are ["points","lines","cells"]
    :param vtype: using full traction vector (with x and y components), the norm (length) of the traction vector, normal
    or shear components  "t_vecs", "t_norm","tn","ts"
    :return:
    """
    all_values = []
    if norm_level == "cells":
        # only makes sense if normal vectors are orientated all in the same manner
        for (
            cell_id,
            line_ids,
        ) in borders.cells_lines.items():  # only uses lines associated to cells
            # orientation correction:
            n_vectors, t_vectors = reorder_vectors_inward(
                borders, lines_interpolation, cell_id, line_ids, plot_n_vecs=False
            )
            # optional exclude borders at the colony edge
            exclude_cond = (
                borders.edge_lines
                if exclude_colony_edge
                else list(lines_interpolation.keys())
            )
            # excluding some lines (or none) and joining to single array
            t_vectors = np.vstack(
                list(exclude_by_key(t_vectors, exclude_cond).values())
            )
            n_vectors = np.vstack(
                list(exclude_by_key(n_vectors, exclude_cond).values())
            )
            # calculating normal and shear components of traction vector after reorientation
            tns, tss = normal_and_shear(t_vectors, n_vectors)
            single_cell_force = None
            if vtype == "t_vecs":
                single_cell_force = np.mean(t_vectors, axis=0)
            if vtype == "t_norm":
                single_cell_force = np.mean(np.linalg.norm(t_vectors, axis=1))
            if vtype == "t_normal":
                single_cell_force = np.mean(tns, axis=0)
            if vtype == "t_shear":
                single_cell_force = np.mean(tss, axis=0)
            all_values.append(single_cell_force)
        all_values = np.array(all_values)

    # optional excluding borders at the colony edge
    if exclude_colony_edge:
        lines_interpolation = exclude_by_key(lines_interpolation, borders.edge_lines)
    if norm_level == "lines":  # mean over a line
        all_values = np.vstack(
            [
                np.mean(sub_dict[vtype], axis=0)
                for sub_dict in lines_interpolation.values()
            ]
        )
    if norm_level == "points":  # each point individually
        all_values = np.concatenate(
            [sub_dict[vtype] for sub_dict in lines_interpolation.values()]
        )

    # returning the norm of the mean t_vector
    all_values = np.linalg.norm(all_values, axis=1) if vtype == "t_vecs" else all_values
    mean = np.mean(all_values, axis=0)  #
    std = np.std(all_values, axis=0)

    return all_values, mean, std
