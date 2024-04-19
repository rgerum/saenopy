import numpy as np
import copy

import solidspy
import solidspy.assemutil as ass
import solidspy.postprocesor as pos
import solidspy.solutil as sol
from saenopy.pyTFM.calculate_stress_imports.stress_functions import (
    calculate_stress_tensor,
)
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from skimage.measure import regionprops


def cut_mask_from_edge(mask, cut_factor, warn_flag=False, fill=True):
    sum_mask1 = np.sum(mask)
    dims = mask.shape
    inds = [
        int(dims[0] * cut_factor),
        int(dims[0] - (dims[0] * cut_factor)),
        int(dims[1] * cut_factor),
        int(dims[1] - (dims[1] * cut_factor)),
    ]
    if fill:  # filling to the original shape
        mask_cut = copy.deepcopy(mask)
        mask_cut[: inds[0], :] = 0
        mask_cut[inds[1] :, :] = 0
        mask_cut[:, : inds[2]] = 0
        mask_cut[:, inds[3] :] = 0
    else:  # new array with new shape
        mask_cut = np.zeros((inds[1] - inds[0], inds[3] - inds[2]))
        mask_cut = mask[inds[0] : inds[1], inds[2] : inds[3]]

    sum_mask2 = np.sum(mask_cut)
    warn = (
        "mask was cut close to image edge"
        if (sum_mask2 < sum_mask1 and warn_flag)
        else ""
    )
    return mask_cut, warn


def FEM_simulation(nodes, elements, loads, mats, mask_area, verbose=False, **kwargs):
    from packaging import version

    if version.parse(solidspy.__version__) > version.parse("1.1.0"):
        cond = nodes[:, -2:]
        nodes = nodes[:, :-2]
    else:
        cond = nodes

    try:
        DME, IBC, neq = ass.DME(cond, elements)  # boundary conditions asembly??
    except ValueError:
        cond = nodes[:, -2:]
        nodes = nodes[:, :-2]
        DME, IBC, neq = ass.DME(cond, elements)  # boundary conditions asembly??

    print("Number of elements: {}".format(elements.shape[0]))
    print("Number of equations: {}".format(neq))

    # System assembly
    KG = ass.assembler(elements, mats, nodes, neq, DME, sparse=True)
    if isinstance(KG, tuple) or version.parse(solidspy.__version__) > version.parse(
        "1.1.0"
    ):
        KG = KG[0]
    RHSG = ass.loadasem(loads, IBC, neq)

    if (
        np.sum(IBC == -1) < 3
    ):  # 1 or zero fixed nodes/ pure neumann-boundary-condition system needs further constraints
        # System solution with custom conditions
        # solver with constraints to zero translation and zero rotation
        UG_sol, rx = custom_solver(KG, RHSG, mask_area, nodes, IBC, verbose=verbose)

    # System solution with default solver
    else:
        UG_sol = sol.static_sol(KG, RHSG)  # automatically detect sparce matrix
        if not (np.allclose(KG.dot(UG_sol) / KG.max(), RHSG / KG.max())):
            print("The system is not in equilibrium!")

    # average shear and normal stress on the colony area
    UC = pos.complete_disp(IBC, nodes, UG_sol)  # uc are x and y displacements
    E_nodes, S_nodes = pos.strain_nodes(
        nodes, elements, mats, UC
    )  # stresses and strains
    stress_tensor = calculate_stress_tensor(
        S_nodes, nodes, dims=mask_area.shape
    )  # assembling the stress tensor
    return UG_sol, stress_tensor


def grid_setup(mask_area, f_x, f_y, E=1, sigma=0.5, edge_factor=0):
    """
    setup of nodes, elements, loads and mats(elastic material properties) lists for solids pys finite elements analysis. Every pixel of
    the provided mask is used as a node. Values from f_x,f_y at these pixels are used as loads. Mats is just
    [E, sigma].
    :param mask_area:
    :param f_x:
    :param f_y:
    :param E:
    :param sigma:
    :return:
    """

    coords = np.array(
        np.where(mask_area)
    )  # retrieving all coordinates from the  points  in the mask

    # setting up nodes list:[node_id,x_coordinate,y_coordinate,fixation_y,fixation_x]
    nodes = np.zeros((coords.shape[1], 5))
    nodes[:, 0] = np.arange(coords.shape[1])
    nodes[:, 1] = coords[1]  # x coordinate
    nodes[:, 2] = coords[0]  # y coordinate

    # creating an 2D array, with the node id of each pixel. Non assigned pixel is -1.
    ids = np.zeros(mask_area.shape).T - 1
    ids[coords[0], coords[1]] = np.arange(
        coords.shape[1], dtype=int
    )  # filling with node ids

    # fix all nodes that are exactly at the edge of the image (minus any regions close to the image edge that are
    # supposed to be ignored)in the movement direction perpendicular to the edge
    ids_cut, w = cut_mask_from_edge(ids, edge_factor, "", fill=False)
    edge_nodes_horizontal = np.hstack([ids_cut[:, 0], ids_cut[:, -1]]).astype(
        int
    )  # upper and lower image edge
    edge_nodes_vertical = np.hstack([ids_cut[0, :], ids_cut[-1, :]]).astype(
        int
    )  # left and right image edge
    edge_nodes_horizontal = edge_nodes_horizontal[edge_nodes_horizontal >= 0]
    edge_nodes_vertical = edge_nodes_vertical[edge_nodes_vertical >= 0]
    nodes[edge_nodes_vertical, 3] = -1  # fixed in x direction
    nodes[edge_nodes_horizontal, 4] = -1  # fixed in y direction
    nodes = nodes.astype(int)

    # setting up elements list:[ele_id,element type,reference_to_material_properties,node1,node2,node3,node4]
    # nodes must be list in counter clockwise oder for solidspy reasons
    elements = np.zeros((coords.shape[1], 7))

    # list the square(node,node left,node left down, node down) for each node. These are all possible square shaped
    # elements, with the correct orientation

    sqr = [
        (coords[0], coords[1] - 1),
        (coords[0] - 1, coords[1] - 1),
        (coords[0] - 1, coords[1]),
        (coords[0], coords[1]),
    ]
    # this produce negative indices, when at the edge of the mask
    # filtering these values
    filter = (
        np.sum(np.array([(s[0] < 0) + (s[1] < 0) for s in sqr]), axis=0) > 0
    )  # logical to find any square with negative coordinates
    sqr = [(s[0][~filter], s[1][~filter]) for s in sqr]  # applying filter
    elements = elements[
        ~filter
    ]  # shortening length of elements list according to the same filter

    # enter node ids in elements, needs counterclockwise arrangement
    # check by calling pyTFM.functions_for_cell_colonie.plot_grid(nodes,elements,inverted_axis=False,symbol_size=4,arrows=True,image=0)
    elements[:, 6] = ids[sqr[3]]
    elements[:, 5] = ids[sqr[2]]
    elements[:, 4] = ids[sqr[1]]
    elements[:, 3] = ids[sqr[0]]

    # cleaning up elements with nodes outside of cell area
    elements = np.delete(elements, np.where(elements == -1)[0], 0)
    # setting element id and attributing material and element type properties
    elements[:, 0] = np.arange(len(elements))  # id
    elements[:, 1] = 1  # element type/geometry (squares)
    elements[:, 2] = 0  # elastic properties reference
    elements = elements.astype(int)

    # setting up forces
    loads = np.zeros((len(nodes), 3))
    loads[:, 0] = np.arange(len(nodes))
    loads[:, 1] = f_x[coords[0], coords[1]]
    loads[:, 2] = f_y[coords[0], coords[1]]
    # loads=loads[not edge_nodes,:] ## check if this works/ is necessary

    mats = np.array([[E, sigma]])  # material properties: youngsmodulus, poisson ratio
    return nodes, elements, loads, mats


def prepare_forces(tx, ty, ps, mask):
    f_x = tx * ((ps * (10**-6)) ** 2)  # point force for each node from traction forces
    f_y = ty * ((ps * (10**-6)) ** 2)
    f_x[~mask] = np.nan  # setting all values outside of mask area to zero
    f_y[~mask] = np.nan
    f_x_c1 = f_x - np.nanmean(
        f_x
    )  # normalizing traction force to sum up to zero (no displacement)
    f_y_c1 = f_y - np.nanmean(f_y)
    f_x_c2, f_y_c2, p = correct_torque(f_x_c1, f_y_c1, mask)
    return f_x_c2, f_y_c2


def correct_torque(fx, fy, mask_area):
    com = regionprops(mask_area.astype(int))[0].centroid  # finding center of mass
    com = (com[1], com[0])  # as x y coordinate

    c_x, c_y = np.meshgrid(
        range(fx.shape[1]), range(fx.shape[0])
    )  # arrays with all x and y coordinates
    r = np.zeros((fx.shape[0], fx.shape[1], 2))  # array with all positional vectors
    r[:, :, 0] = c_x  # note maybe its also enough to chose any point as refernece point
    r[:, :, 1] = c_y
    r = r - np.array(com)

    f = np.zeros(
        (fx.shape[0], fx.shape[1], 2), dtype="float64"
    )  # array with all force vectors
    f[:, :, 0] = fx
    f[:, :, 1] = fy
    q = np.zeros(
        (fx.shape[0], fx.shape[1], 2), dtype="float64"
    )  # rotated positional vectors

    def get_torque_angle(p):
        q[:, :, 0] = +np.cos(p) * (f[:, :, 0]) - np.sin(p) * (
            f[:, :, 1]
        )  # what's the mathematics behind this??
        q[:, :, 1] = +np.sin(p) * (f[:, :, 0]) + np.cos(p) * (f[:, :, 1])
        torque = np.abs(
            np.nansum(np.cross(r, q, axisa=2, axisb=2))
        )  # using nan sum to only look at force values in mask
        return torque.astype("float64")

    # plotting torque angle relation ship
    # ps=np.arange(-np.pi/2,np.pi/2,0.01)
    # torques=[get_torque_angle(p)*1000 for p in ps]
    # plt.figure()
    # ticks=np.arange(-np.pi/2,np.pi/2+np.pi/6,np.pi/6)
    # tick_labels=[r"$-\frac{\pi}{2}$",r"$-\frac{\pi}{3}$",r"$-\frac{\pi}{6}$",r"$0$",r"$\frac{\pi}{6}$",r"$\frac{\pi}{3}$",r"$\frac{\pi}{2}$"]
    # plt.xticks(ticks,tick_labels,fontsize=25)
    # plt.yticks(fontsize=15)
    # plt.plot(ps,torques,linewidth=6)
    # plt.gca().spines['bottom'].set_color('black')
    # plt.gca().spines['left'].set_color('black')
    # plt.gca().tick_params(axis='x', colors='black')
    # plt.gca().tick_params(axis='y', colors='black')
    # plt.savefig("/home/user/Desktop/results/thesis/figures/torque_angle.png")

    pstart = 0
    # bounds = ([-np.pi], [np.pi])
    ## just use normal gradient descent??
    eps = np.finfo(float).eps  # minimum machine tolerance, for most exact calculation
    # trust region algorithm,
    # there seems to be a bug when using very small tolerances close to the machine precision limit (eps)
    # in rare cases there is an error. see also https://github.com/scipy/scipy/issues/11572
    try:
        p = least_squares(
            fun=get_torque_angle,
            x0=pstart,
            method="lm",
            max_nfev=100000000,
            xtol=eps,
            ftol=eps,
            gtol=eps,
            args=(),
        )["x"]
    except KeyError:
        eps *= 5
        p = least_squares(
            fun=get_torque_angle,
            x0=pstart,
            method="lm",
            max_nfev=100000000,
            xtol=eps,
            ftol=eps,
            gtol=eps,
            args=(),
        )["x"]

    q[:, :, 0] = +np.cos(p) * (f[:, :, 0]) - np.sin(p) * (
        f[:, :, 1]
    )  # corrected forces
    q[:, :, 1] = +np.sin(p) * (f[:, :, 0]) + np.cos(p) * (f[:, :, 1])

    return q[:, :, 0], q[:, :, 1], p  # returns corrected forces and rotation angle


def find_eq_position(nodes, IBC, neq):
    # based on solidspy.assemutil.loadasem

    nloads = IBC.shape[0]
    RHSG = np.zeros((neq, 2))
    x_points = np.zeros((neq)).astype(
        bool
    )  # mask showing which point has x deformation
    y_points = np.zeros((neq)).astype(
        bool
    )  # mask showing which point has y deformation
    for i in range(nloads):
        il = int(nodes[i, 0])  # index of the node
        ilx = IBC[il, 0]  # indices in RHSG or fixed nodes, if -1
        ily = IBC[il, 1]
        if ilx != -1:
            RHSG[ilx] = nodes[i, [1, 2]]  # x,y position/ not the orientation
            x_points[ilx] = [True]
        if ily != -1:
            RHSG[ily] = nodes[i, [1, 2]]
            y_points[ily] = [True]

    return RHSG.astype(int), x_points, y_points


def custom_solver(mat, rhs, mask_area, nodes, IBC, verbose=False):
    # IBC is "internal boundary condition" contains information about which nodes are fixed and
    # where the unfixed nodes can be found in the rhs vector

    """Solve a static problem [mat]{u_sol} = {rhs}

    Parameters
    ----------
    mat : array
        Array with the system of equations. It can be stored in
        dense or sparse scheme.
    rhs : array
        Array with right-hand-side of the system of equations.

    Returns
    -------
    u_sol : array
        Solution of the system of equations.

    Raises
    ------
    """

    len_disp = mat.shape[1]  # length of the  displacement vector
    zero_disp_x = np.zeros(len_disp)
    zero_disp_y = np.zeros(len_disp)
    zero_torque = np.zeros(len_disp)

    com = regionprops(mask_area.astype(int))[0].centroid  # finding center of mass
    com = (com[1], com[0])  # as x y coordinate

    c_x, c_y = np.meshgrid(
        range(mask_area.shape[1]), range(mask_area.shape[0])
    )  # arrays with all x and y coordinates
    r = np.zeros(
        (mask_area.shape[0], mask_area.shape[1], 2)
    )  # array with all positional vectors
    r[:, :, 0] = (
        c_x  # Note: maybe its also enough to chose any point as reference point
    )
    r[:, :, 1] = c_y
    # solidspy function that is used to construct the loads vector (rhs)
    nodes_xy_ordered, x_points, y_points = find_eq_position(nodes, IBC, len_disp)
    r = r[
        nodes_xy_ordered[:, 1], nodes_xy_ordered[:, 0], :
    ]  # ordering r in the same order as rhs
    r = r - np.array(com)

    zero_disp_x[x_points] = 1
    zero_disp_y[y_points] = 1

    # torque=sum(r1*f2-r2*f1)   # TDOD: this is actually zero rotation
    zero_torque[x_points] = r[x_points, 1]  # -r2 factor
    zero_torque[y_points] = -r[y_points, 0]  # r1 factor
    add_matrix = np.vstack([zero_disp_x, zero_disp_y, zero_torque])
    # adding zero conditions for force vector and torque
    rhs = np.append(rhs, np.zeros(3))

    if type(mat) is csr_matrix:
        import scipy.sparse

        # convert additional conditions to sparse matrix
        mat = scipy.sparse.vstack([mat, csr_matrix(add_matrix)], format="csr")
        u_sol, error = np.array(
            lsqr(
                mat,
                rhs,
                atol=10**-12,
                btol=10**-12,
                iter_lim=200000,
                show=verbose,
                conlim=10**12,
            ),
            dtype=object
        )[
            [0, 3]
        ]  # sparse least squares solver
    elif type(mat) is np.ndarray:
        # adding to matrix
        mat = np.append(mat, add_matrix, axis=0)

        u_sol, error = np.array(np.linalg.lstsq(mat, rhs))[[0, 1]]
    else:
        raise TypeError("Matrix should be numpy array or csr_matrix.")

    return u_sol, error
