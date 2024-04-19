import numpy as np
import copy

from collections import defaultdict
from collections import Counter
from contextlib import suppress
from itertools import chain

from saenopy.pyTFM.calculate_stress_imports.stress_functions import (
    normal_vectors_from_splines,
)
from .graph_theory_for_cell_boundaries import (
    mask_to_graph,
    FindingBorderError,
    find_lines_simple,
    remove_endpoints_wrapper,
    graph_to_mask,
    identify_line_segments,
    find_dead_end_lines,
    find_exact_line_endpoints,
)
from saenopy.pyTFM.utilities_TFM import join_dictionary
from scipy.interpolate import splprep, splev
from scipy.ndimage import binary_fill_holes
from skimage.morphology import (
    skeletonize,
    remove_small_objects,
    label,
    binary_dilation,
)

from .mask_interpolation import mask_interpolation


def spline_interpolation(line, points, endpoints=None):
    """
    function takes points from a line and uses spline interpolation to create a smooth representation.

    :param line: list point ids that define a line. Must be incorrect order
    :param points: List of all points. This list is where the point ids correspond to. yx order
    :param endpoints: additional pair of endpoints in yx order, to include in the spline interpolation
    :return: tck, array defining the spline, with knot position, parameter values and order
            points_new, new position of the points on the line according to spline interpolation in x y coordinates !!!
    """

    x = points[line, 1]  # list x and y coordinate for points on the line
    y = points[line, 0]

    if endpoints:  # adding endpoints if provided
        # splprep throws error if two points are exactly identical
        # for example for the "dead-end-lines" the endpoint is identical to the first and last point and thus shouldnt
        # be added here
        if not (endpoints[0][1] == x[0] and endpoints[0][0] == y[0]):
            x = np.concatenate([[endpoints[0][1]], x], axis=0)
            y = np.concatenate([[endpoints[0][0]], y], axis=0)
        if not (endpoints[1][1] == x[-1] and endpoints[1][0] == y[-1]):
            x = np.concatenate([x, [endpoints[1][1]]], axis=0)
            y = np.concatenate([y, [endpoints[1][0]]], axis=0)

    k = (
        len(x) - 1 if len(x) <= 3 else 3
    )  # addapt spline order, according to number of points
    tck, u = splprep([x, y], s=10, k=k)  # parametric spline interpolation
    # fits essentially function: [x,y] =f(t) , t(paramter) is default np.linspace(0,1,len(x))
    # tck is array with x_position of knot, y_position of knot, parameters of the plne, order of the spline
    # ui is paramter given to x,y points, in this case default (values from 0 to 1)
    # s is smoothing factor(?)
    # k is order of the spline, cubic is fine

    if endpoints:  # forcing splines to through endpoints
        tck[1][0][0] = endpoints[0][1]
        tck[1][0][-1] = endpoints[1][1]
        tck[1][1][0] = endpoints[0][0]
        tck[1][1][-1] = endpoints[1][0]
    # new points from spline interpolation (thes points will be used for normal stress vector calculation
    points_new = np.array(splev(u, tck, der=0)).T
    # in xy orientation
    # points_new = np.round(points_new).astype(int)  ## could also use exact values and the interpolate the stress tensor, but thats probably to complicated
    return tck, u, points_new  # points new in xy orientation


def arrange_lines_from_endpoints(cells_lines, lines_endpoints_com):
    """
    rearranging the order of lines in the line_id list for one one cell. The endpoints for all lines for ne cell are
    extracted and then puzzled together. Each endpoint must occure in two lines.

    :param cells_lines: dictionary cell_id:[line_ids]
    :param lines_endpoints_com: dictionary line_id:[endpoint1,endpoint2] , each endoinpt is an array with x and y coordinates
    :return: cells_lines_new: updated cells_lines dictionary

    """
    cells_lines_new = {}

    for cell_id, line_ids in cells_lines.items():
        # extracting relevant endpoints
        local_endpoints = {
            line_id: lines_endpoints_com[line_id] for line_id in line_ids
        }
        # rearranging endpoints into an suitabel array: axis0: lines axis 1:[endpoint1, endpoint2], axis3: x,y coordinates
        eps = np.array(
            [np.array([value[0], value[1]]) for value in local_endpoints.values()]
        )

        new_line_ids = []  # newly_arranged line_ids
        p_ind1 = 0  # start ids
        p_ind2 = 0
        # iterating through eps, until all endpoints have been visited
        while not np.isnan(eps).all():
            point = copy.deepcopy(eps[p_ind1, p_ind2])  # extracting an endpoint
            eps[p_ind1, p_ind2] = np.nan  # remove the point from array
            # find second occurrence, by taking the norm of the diffrence between the current point and all other points
            # this should be zero
            np_ind1, np_ind2 = np.array(
                np.where(np.linalg.norm(eps - point, axis=2) == 0)
            ).T[0]
            new_line_ids.append(line_ids[np_ind1])  # note corresponding line_ids
            eps[np_ind1, np_ind2] = np.nan  # remove this point from array
            p_ind1, p_ind2 = np_ind1, np.abs(
                np_ind2 - 1
            )  # change to the other end of the line

        cells_lines_new[cell_id] = new_line_ids  # update dictionary

    return cells_lines_new


def find_edge_lines(cells_lines):
    """
    Finding all lines (cell borders) at the edge of a cell colony. Simply checks if
    a line is associated to only one cell.
    :param cells_lines: dictionary with cell_id:[associated line_ids]
    :return: edge_lines: lsit of line ids at the edge of the cell colony
    """
    all_lines = np.array(
        list(chain.from_iterable(cells_lines.values()))
    )  # unpacking all line ids
    counts = Counter(all_lines)  # counting occurences
    edge_lines = [
        line for line, count in counts.items() if count == 1
    ]  # select if line_id was only associated to one cell
    return edge_lines


def center_of_mass_cells(cells_points, points):
    """
    calulating the "center of mass" of a cell using only the points at the edge of the cell
    :param cells_points:
    :param points:
    :return:
    """
    cells_com = {}
    for cell_id, hull_points in cells_points.items():
        cells_com[cell_id] = np.mean(points[hull_points], axis=0)

    return cells_com


def remove_circular_line(
    allLines_points, lines_endpoints_com, lines_points, lines_endpoints
):
    """
    finds lines that are circular by checking if the first and second endpoint are identical. The lines are
    deleted from all input dictionaries
    :param lines_endpoints_com:
    :param lines_points:
    :param lines_endpoints:
    :return:
    """
    # finding all lines where first and second endpoint is identical
    circular = [
        l_id
        for l_id, endpoints in lines_endpoints_com.items()
        if np.linalg.norm(endpoints[0] - endpoints[1]) == 0
    ]
    # print(circular)
    # clearing these lines from the input dictionaries
    for l_id in circular:
        del lines_endpoints_com[l_id]
        del lines_points[l_id]
        del lines_endpoints[l_id]
        del allLines_points[l_id]


def interpolate_cell_area(cells_area, shape):
    cells_area_interpol = {}
    for cell_id, areas in cells_area.items():
        cells_area_interpol[cell_id] = mask_interpolation(areas, shape, min_cell_size=0)
    return cells_area_interpol


def identify_cells(mask_area, mask_boundaries, points):
    """
    function to identify cells. Each cell is a dictionary entry with a list of ids, referring to
    points.
    :param mask_area:
    :param mask_boundaries:
    :param points:
    :return:
    """

    cells = {}  # dictionary containg a list of point idsthat sourround each cell
    cells_area = (
        {}
    )  # dictionary containg a all pixels belonging to that cell as boolean aray
    # points_to_flatt array map:
    # labeling each cell
    m = mask_area.astype(int) - mask_boundaries.astype(int)
    ml = label(m, connectivity=1)
    # creating a list of point coordinates corresponding to a flattend array
    # this will allow easier identification of the id of a point
    points_fl = (points[:, 0]) + mask_area.shape[0] * (points[:, 1])
    sort_ids = np.argsort(points_fl)  # sorting will allow np.searchsorted function;
    points_fl = points_fl[sort_ids]  #

    for i, l in enumerate(
        np.unique(ml)[1:]
    ):  # getting each cell border by binary dilation of one pixel; iterating over each cell
        m_part = (ml == l).astype(bool)  # extracting a cell area
        edge = np.logical_and(
            binary_dilation(m_part), ~m_part
        )  # getting the boundary of a cell
        ps = np.array(np.where(edge)).T  # finding coordinates
        ps_fl = (ps[:, 0]) + mask_area.shape[0] * (
            ps[:, 1]
        )  # convert coordinates to the one of a flat array
        p_ids = sort_ids[
            np.searchsorted(points_fl, ps_fl)
        ]  # find indices, where i would need to insert,supposed to be the fastest way
        #  and read index from unsorted list
        cells[i] = p_ids  # save to dictionary
        cells_area[i] = m_part

    ## vizualization
    # creating a(random lsit of hex colors)
    # colors = []
    # for i in range(len(cells.items())):
    #    colors.append('#%06X' % np.random.randint(0, 0xFFFFFF))
    # plt.figure()
    # plt.imshow(mask_boundaries)
    # offset = 0.01
    # for cell, ps in cells.items():
    #    for p in ps:
    #        plt.plot(points[p][1] + offset * cell, points[p][0] + offset * cell, "o", color=colors[cell])

    return cells, cells_area


def interpolation_single_point(point, shape_target, shape_origin):
    # is also works with 2d arrays of shape(n,2)
    interpol_factors = np.array(
        [shape_target[0] / shape_origin[0], shape_target[1] / shape_origin[1]]
    )
    point_interp = point * interpol_factors
    return point_interp


def interpolate_points_dict(points_dict, shape_target, shape_orgin):
    points_dict_interp = {}
    for p_id, coords in points_dict.items():
        points_dict_interp[p_id] = (
            interpolation_single_point(coords[0], shape_target, shape_orgin),
            interpolation_single_point(coords[1], shape_target, shape_orgin),
        )
    return points_dict_interp


class Cells_and_Lines:
    # container for cells and lines, assignment of points with them, and assignment with each other
    def __init__(self, mask_boundaries, shape, graph, points):

        # masks, graph and points including dead-end lines // this distinction is mostly due to historic reasons
        self.mask_boundaries_wp = mask_boundaries
        self.inter_shape = shape
        # graph as a dictionary with key = point id, values: ids of neighbouring points
        # any point id is the index in the points array (contains coordinate of these points
        self.graph_wp = graph
        self.points_wp = points
        self.graph, self.points, removed = remove_endpoints_wrapper(
            self.graph_wp, self.points_wp
        )
        # masks, graph and points excluding dead-end lines
        self.mask_boundaries = graph_to_mask(
            self.graph, self.points, mask_boundaries.shape
        )  # rebuilding the mask

        # interpolate points to the size of the future FEM_grid
        self.points_interpol = interpolation_single_point(
            self.points, self.inter_shape, self.mask_boundaries.shape
        )
        # interpolation factors used in the fun
        # self.inerpol_factors=np.array([self.inter_shape[0] /self.mask_boundaries.shape[0], self.inter_shape[1] / self.mask_boundaries.shape[1]])
        # points as dictionary with key=points id, values: points coordinates
        self.points_dict = {i: self.points[i] for i in range(len(self.points))}

        # lines as a dictionary with key=line id, values: ids of containing points (in correct order)
        self.lines_points = identify_line_segments(self.graph, self.points_interpol)

        # cells as a dictionary with key=cell id, values: ids of containing points (not ordered)
        self.cells_points, self.cells_area = identify_cells(
            self.mask_boundaries, binary_fill_holes(self.mask_boundaries), self.points
        )

        # interpolate the area of individual cells to the size of deformation
        self.cells_area_interpol = interpolate_cell_area(
            self.cells_area, self.inter_shape
        )

        # self.points_lines = invert_dictionary(self.lines_points) # point_id:line_id
        self.max_line_id = np.max(list(self.lines_points.keys()))
        self.de_lines_points, self.max_line_id = find_dead_end_lines(
            self.graph_wp, list(self.graph.keys()), self.max_line_id
        )
        self.de_endpoints = {
            key: (self.points[value[0]], self.points[value[-1]])
            for key, value in self.de_lines_points.items()
        }  # using exact endpoints for the dead end lines
        self.allLines_points = join_dictionary(self.lines_points, self.de_lines_points)

        # dictionary with endpoints, needed to completely fill the gaps between all cell_lines
        self.lines_endpoints_com, self.lines_endpoints = find_exact_line_endpoints(
            self.lines_points, self.points, self.graph
        )

        # self.simple_line_plotting(self.allLines_points, subset=np.inf)
        # for p in self.lines_endpoints_com.values():
        #    plt.plot(p[0][1],p[0][0],"o")
        #    plt.plot(p[1][1], p[1][0], "o")

        # removing all lines that are predicted to be circular. Mostly a problem for very short lines
        remove_circular_line(
            self.allLines_points,
            self.lines_endpoints_com,
            self.lines_points,
            self.lines_endpoints,
        )

        # center of mass of cells, calculated only from the hull points
        self.cells_com = center_of_mass_cells(self.cells_points, self.points)
        # dictionary to associate cells with correct lines, key is cell id, value is line_id
        self.cells_lines = defaultdict(list)
        # dictionary to associate cells with correct lines, key is line id, value is cell

        self.lines_cells = defaultdict(list)
        for l_id, l in self.lines_points.items():
            for c_id, c in self.cells_points.items():
                if l[int(len(l) / 2)] in c:
                    self.cells_lines[c_id].append(l_id)
                    self.lines_cells[l_id].append(c_id)
        # using the new endpoints to arrange lines in the correct way
        self.cells_lines = arrange_lines_from_endpoints(
            self.cells_lines, self.lines_endpoints_com
        )

        # adding dead end endpoints only now to avoid complications when identifying cells
        self.de_endpoints = {
            key: (self.points[value[0]], self.points[value[-1]])
            for key, value in self.de_lines_points.items()
        }
        self.lines_endpoints_com = join_dictionary(
            self.lines_endpoints_com, self.de_endpoints
        )
        self.lines_endpoints_interpol = interpolate_points_dict(
            self.lines_endpoints_com, self.inter_shape, self.mask_boundaries.shape
        )

        # list of ids
        self.point_ids = list(self.points_dict.keys())
        self.cell_ids = list(self.cells_points.keys())
        self.line_ids = list(self.allLines_points.keys())
        # list of all lines at the edge of the cell colony
        self.edge_lines = find_edge_lines(self.cells_lines)
        # list of dead end lines
        self.dead_end_lines = list(self.de_lines_points.keys())
        # list of central boundary (non-dead-end, non-edge lines)
        self.central_lines = [
            line_id
            for line_id in self.allLines_points.keys()
            if line_id not in self.edge_lines and line_id not in self.dead_end_lines
        ]
        self.n_cells = len(self.cell_ids)
        # list with original line lengths--> later used for interpolation
        self.line_lengths = {
            key: len(value) for key, value in self.allLines_points.items()
        }

        # dictionary containing the spline representation of the points as a parametric function
        # [x,y]=f(u). u is always np.linspace(0,1,"number of points in the line). Use scipy.interpolate.splev
        # to evaluate at other points
        self.lines_splines = defaultdict(list)

        # dictionary with the normal vectors for each line according to the spline interpolation. Contains list
        # of the normal vectors , position of these points is listed in lines_spline_points
        # interpolation !!! positions are given in xy order !!!
        self.lines_n_vectors = defaultdict(list)

        # dictionary with example points (where a normal vector originates) from spline interpolation
        self.lines_spline_points = defaultdict(list)

        for line_id, line_ps in self.allLines_points.items():
            endpoints = self.lines_endpoints_interpol[line_id]
            k = (
                len(line_ps) + 2 - 1 if len(line_ps) <= 3 else 3
            )  # adapt spline order, according to number of points
            # spline order must be below number of points, so choose 2 if lne has one point + 2 endpoints
            tck, u, points_new = spline_interpolation(
                line_ps, self.points_interpol, endpoints=endpoints
            )  # spline interpolation
            self.lines_splines[line_id] = tck  # saving the spline object
            # saving a few points and n vectors for easy representations/ debugging,
            # these points will not be used further
            n_vectors = normal_vectors_from_splines(
                u, tck
            )  # calculating normal vectors as a list
            self.lines_n_vectors[line_id] = n_vectors
            self.lines_spline_points[line_id] = points_new

    def filter_small_de_line(self, min_length):
        for l_id in copy.deepcopy(
            self.dead_end_lines
        ):  # does not filter small line segments around cells -
            if self.line_lengths[l_id] < min_length:
                with suppress(AttributeError):
                    self.allLines_points.pop(l_id, None)
                with suppress(AttributeError):
                    self.lines_endpoints_com.pop(l_id, None)
                with suppress(AttributeError):
                    self.lines_endpoints_interpol.pop(l_id, None)
                with suppress(AttributeError):
                    self.lines_splines.pop(l_id, None)
                with suppress(AttributeError):
                    self.lines_n_vectors.pop(l_id, None)
                with suppress(AttributeError):
                    self.lines_spline_points.pop(l_id, None)
                with suppress(AttributeError):
                    self.line_lengths.pop(l_id, None)
                with suppress(AttributeError):
                    self.de_endpoints.pop(l_id, None)
                with suppress(AttributeError):
                    self.de_lines_points.pop(l_id, None)

                with suppress(ValueError, AttributeError):
                    self.line_ids.remove(l_id)  # list
                with suppress(ValueError, AttributeError):
                    self.dead_end_lines.remove(l_id)


class Cells_and_Lines2(Cells_and_Lines):
    # much simplified version. Doesn't find cells and doesn't guarantee to find long line segments.
    # Can't identify lines at the edge of a colony

    def __init__(self, mask_boundaries, shape, graph, points):
        self.mask_boundaries = mask_boundaries
        self.inter_shape = shape
        # graph as a dictionary with key=point id, values: ids of neighbouring points
        # any point id is the index in the points array (contains coordinate of these points
        self.graph = graph
        self.points = points
        # finding line segments
        self.lines_points = find_lines_simple(self.graph)
        self.points_interpol = interpolation_single_point(
            self.points, self.inter_shape, self.mask_boundaries.shape
        )

        self.lines_splines = defaultdict(list)
        self.lines_n_vectors = defaultdict(list)
        self.lines_spline_points = defaultdict(list)

        # spline interpolation
        for line_id, line_ps in self.lines_points.items():
            # spline order must be below number of points, so choose 2 if lne has one point + 2 endpoints
            tck, u, points_new = spline_interpolation(
                line_ps, self.points_interpol
            )  # spline interpolation
            self.lines_splines[line_id] = tck  # saving the spline object
            # saving a few points and n vectors for easy representations/ debugging,
            # these points will not be used further
            n_vectors = normal_vectors_from_splines(
                u, tck
            )  # calculating normal vectors as a list
            self.lines_n_vectors[line_id] = n_vectors
            self.lines_spline_points[line_id] = points_new

        # categories from other lines object
        self.cell_ids = []
        self.line_ids = list(self.lines_points.keys())
        self.edge_lines = []
        self.dead_end_lines = []
        self.central_lines = self.line_ids
        self.line_lengths = {
            key: len(value) for key, value in self.lines_points.items()
        }

        # very rough estimate of cell number
        label_mask, self.n_cells = label(
            ~mask_boundaries, connectivity=1, return_num=True
        )


def find_borders(mask, shape, raise_error=True, type="colony", min_length=0):
    # maybe reintroduce small cell filter
    # removing small bits
    mask = remove_small_objects(mask.astype(bool), 1000).astype(bool)
    # generating borders
    mask_boundaries = skeletonize(mask.astype(int))
    # converting mask to graph object
    graph, points = mask_to_graph(mask_boundaries)
    # finding dead ends: cell borders which don't connect to other cell borders at one end:
    # this is use full to get a clean structure of cell borders, which is later used to identifying the number and area of cells
    # applying remove endpoints multiple times to deal with forking dead ends

    try:
        if type == "colony":
            c_l = Cells_and_Lines(mask_boundaries, shape, graph, points)
            c_l.filter_small_de_line(min_length)
        elif type == "cell layer":
            c_l = Cells_and_Lines2(mask_boundaries, shape, graph, points)
            c_l.filter_small_de_line(min_length)
        else:
            raise ValueError("unknown type")
    except (RecursionError, FindingBorderError, IndexError, KeyError) as e:
        print("original error: ", e)
        if raise_error:
            raise FindingBorderError
        else:
            return None

    # c_l.cut_to_FEM_grid(mask_int) # this should never be necessary
    # c_l.vizualize_lines_and_cells(sample_factor=0.2,plot_n_vectors=True)
    # c_l.vizualize_splines(sample_factor=4,subset=200000)
    # c_l.simple_line_plotting(c_l.lines_points)
    ## vizualization of spline interpolation with new line enpoints
    return c_l
