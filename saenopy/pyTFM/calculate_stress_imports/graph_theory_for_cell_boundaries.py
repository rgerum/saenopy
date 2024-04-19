# handling cell boundaries and masks by applying graph theory
import copy
from collections import defaultdict
from collections import namedtuple
from itertools import chain

import numpy as np
from scipy.spatial import cKDTree

inf = float("inf")
Edge = namedtuple("Edge", "start, end, cost")


class FindingBorderError(Exception):
    pass


def graph_to_mask(graph, points, dims):
    m = np.zeros(dims)
    ps = np.array(
        [y for x in list(graph.values()) for y in x]
    )  # flattening list of point ids
    ps_coord = points[ps]  # getting coordinates
    m[ps_coord[:, 0], ps_coord[:, 1]] = 1  # writing points
    return m


def remove_endpoints_wrapper(graph, points):
    graph_cp = copy.deepcopy(graph)
    eps_id = find_endpoints(graph_cp)  # keys of endpoints, this fast and efficient
    removed = []
    for ep in eps_id:  # removing the endpoint and its connections from the graph
        remove_endpoints(graph_cp, ep, removed)
    return graph_cp, points, removed


def check_connectivity(graph, ep):
    """
    checking if removing a node from a graph changes the connectivity of the neighbouring nodes.
    in other words: are the neighbouring nodes still connected to one another even if I remove the original node.
    The neighbouring points must be connected via maximally one other node. (more would represent ana actual hole in the
    skeletonized mask)
    This classifies loose ends and points that can be removed.
    :return:
    """
    # this works for points 1,2 or neighbours
    # identifying an endpoint by checking if removing this point changes the connectivity of its neighbouring nodes
    # i.e. if I break a connection between two points by removing ep
    l1_ps = graph[ep]  # first layer of points
    # check if this causes a line break
    l2_ps = [
        [pi for pi in graph[p] if pi != ep] for p in l1_ps
    ]  # second layer of points if original point was removed
    # third layer of points // don't need to go deeper due to properties of skeletonize
    # also adding points form second layer
    l3_ps = [
        np.unique(list(chain.from_iterable([graph[p] for p in l2_p] + [l2_p])))
        for l2_p in l2_ps
    ]
    # check if all points in l1_ps are connected even if ep is removed
    connectivity = all([all([p in sub_group for p in l1_ps]) for sub_group in l3_ps])
    # check for connection between points in layer 1--> no connection means
    # that removing ep introduces a line break
    return connectivity


def remove_endpoints(graph, ep, removed=None):
    """
    recursive function to remove dead ends in a graph starting from point ep. Ep has one neighbour.
    Function stops if it hits a point with 3 neighbours or the removal of a point would cause the appearance of two more
    loose lines.
    :param graph: graph as a dictionary
    :param ep: start point
    :param removed:
    :return:
    """
    if removed is None:
        removed = []

    connectivity = check_connectivity(graph, ep)
    if connectivity:
        nps = graph[ep]
        remove_point_from_graph(
            graph, ep
        )  # removes the point and all connections form the graph
        removed.append(ep)
    else:
        return
    for p in nps:  # iterating through further points
        remove_endpoints(graph, p, removed)

    return


def remove_point_from_graph(graph, point):
    """
    removes a point and all connections to this point from a graph
    :param graph:
    :param point:
    :return:
    """
    nps = graph[point]  # neighbouring points/nodes
    graph.pop(point)  # removing the node of the graph
    for p in nps:  # removing all connections to this node
        graph[p].remove(point)


def find_endpoints(graph):
    """
    identifies "loose ends":
     goes through all points and checks if removing them would introduce a line break.
    this is just as fast as checking for the number  of neighbours and then checking the distance of these neighbours
    :param graph:
    :return:
    """
    eps = [ep for ep in graph.keys() if check_connectivity(graph, ep)]
    return np.array(eps)


def find_dead_end_lines(graph, non_dead_end_points, max_id):
    """
    finds dead end line segments from their start to the point where they hit a none dead end line.
    The point in the none dead edn line is included
    """

    eps_id = find_endpoints(graph)  # keys of endpoints, this fast and efficient
    dead_end_lines = {}
    for ep in eps_id:
        lps = find_path(graph, ep, non_dead_end_points, path=[])
        non_dead_end_points.extend(
            lps
        )  # adding points in the newly discovered line segments to termination points
        if len(lps) > 3:  # filtering single points and very small bits
            max_id += 1
            dead_end_lines[max_id] = lps
    return dead_end_lines, max_id


def find_lines_simple(graph):
    # find all endpoints
    graph_cp = copy.deepcopy(graph)
    lines_points = {}
    i = 0
    while len(graph_cp.keys()) > 0:
        # first endpoint, if no endpoint the first point
        new_endpoint = next(
            (x for x in iter(graph_cp.keys()) if len(graph_cp[x]) == 1),
            next(iter(graph_cp.keys())),
        )
        line = find_path_to_endpoint(
            graph_cp, new_endpoint, path=[], first=True
        )  # only explores one direction
        for p in line:
            remove_point_from_graph(graph_cp, p)
        if len(line) > 2:
            lines_points[i] = line
            i += 1
        if i > 10000:
            raise FindingBorderError(
                "found more than 100000 lines; something went wrong"
            )
    return lines_points


def find_path(graph, start, end, path=None):
    """
    recursive function
    finds a path (not necessarily the shortest one) through a graph from start to an end node (not necessarily the
    closest one).

    :param graph: dict, graph
    :param start: int, start point, must be a key in the graph
    :param end: list, list of endpoints. when any endpoint is reach the path search is stopped
    :param path: list, all nodes visited on the way from start to the first endpoint
    :return:
    """
    if path is None:
        path = []
    path = path + [start]
    if start in end:
        return path
    if start not in graph.keys():
        return None
    for node in graph[start]:
        if node not in path:
            new_path = find_path(graph, node, end, path)
            if new_path:
                return new_path
    return None  # only partial path


def find_path_to_endpoint(graph, start, path=None, first=False):
    """
    recursive function
    finds a path to a (not specific) point with only one neighbour

    :param graph: dict, graph
    :param start: int, start point, must be a key in the graph
    :param path: list, all nodes visited on the way from start to the first endpoint
    :param first:
    :return:
    """
    if path is None:
        path = []
    path = path + [start]
    if (
        len(graph[start]) < 2 and not first
    ):  # stop if we reached a point with only one neighbour
        return path
    if start not in graph.keys():
        return None
    for node in graph[start]:
        if node not in path:
            new_path = find_path_to_endpoint(graph, node, path)
            if new_path:
                return new_path
    return path  # only partial path


def find_line_segment_recursive(graph, start, path=None, left_right=0):
    """
    ---> would sometimes cause stack overflow/recursion error
    recursive function
    finds path from a point going from the first or second neighbour until
    it reaches an intersection (point with three neighbours)
    :param graph: graph as a dictionary
    :param start: start point
    :param path: path as list of nodes
    :param left_right: define which neighbour from start to explore (0 or 1
    :return:
    """
    if path is None:
        path = []
    if (
        len(graph[start]) > 2
    ):  # stop if intersection (point with 3 neighbours is reached
        return path  # returns the function before next recursion
    # otherwise there is some overlap

    path = path + [start]
    new_ps = np.array(graph[start])  # next points
    if len(path) == 1:
        new_p = new_ps[left_right]  # just choose one point
    else:
        new_p = new_ps[new_ps != path[-2]][
            0
        ]  # next point that wasn't the previous point
    # recursive function
    new_path = find_line_segment_recursive(graph, new_p, path)  # next step

    if new_path:
        return new_path  # return if recursion is completed


def find_line_segment(graph, start, path=None, left_right=0):
    """
    ---> would sometimes cause stack overflow/recursion error
    recursive function
    finds path from a point going from the first or second neighbour until
    it reaches an intersection (point with three neighbours)
    :param graph: graph as a dictionary
    :param start: start point
    :param path: path as list of nodes
    :param left_right: define which neighbour from start to explore (0 or 1
    :return:
    """

    # first point
    path = [start]
    new_ps = graph[start]
    new_p = new_ps[left_right]  # just choose one point to the left or right
    # break of if we already hit another intersection
    if len(graph[new_p]) > 2:
        return path
    else:
        path.append(new_p)

    while True:  # stop if intersection (point with 3 neighbours) is reached
        new_ps = graph[new_p]
        new_p = [p for p in new_ps if p not in path]
        new_p = new_p[0]
        # check if the point has more (or less) then 2 neighbours.
        # should be more than 2 to indicate intersection
        if len(graph[new_p]) != 2:
            break
        else:
            path.append(new_p)
    return path


def mask_to_graph(mask, d=np.sqrt(2)):
    """
    converts a binary mask to a  graph (dictionary of neighbours)
    Neighbours are identified by cKDTree method
    :param mask:
    :param d: maximal allowed distance
    :return:
    """
    graph = defaultdict(list)
    points = np.array(np.where(mask)).T
    point_tree = cKDTree(points)  # look up table for nearest neighbours ??
    for i, p in enumerate(points):
        neighbours = point_tree.query_ball_point(p, d)
        neighbours.remove(i)  # removing the point itself from list of its neighbours
        graph[i].extend(neighbours)
    return graph, points


def identify_line_segments(graph, points):  #
    """
    function to identify all line segments (representing individual cell boundaries). Segments are returned as a
    dictionary with an id as key and a list of points (referring to the points array) that are included in the line.
    The points are in correct order already
    :param graph:
    :param points:
    :return: dictionary with ordered points  in the line
    """

    lines_dict = {}
    n = 0  # counter in while loop
    all_points = list(graph.keys())  # all points in the graph
    intersect_ps = [
        key for key, values in graph.items() if len(values) > 2
    ]  # finding intersection points
    if len(intersect_ps) == 0:
        raise FindingBorderError("Can't identify internal cell borders.")
    remaining = set(all_points) - set(intersect_ps)  # remaining point ids
    while (
        len(remaining) > 0
    ):  # stop when all points are assigned to intersect or line segment
        start = next(iter(remaining))  # first point of remaining points
        # finding a line segment
        line_seg = list(reversed(find_line_segment(graph, start=start, left_right=0)))[
            :-1
        ] + find_line_segment(graph, start=start, left_right=1)

        remaining -= set(line_seg)  # updating remaining list

        # avoid single point lines, which are not useful (cant really get normal vectors from them)
        if len(line_seg) > 1:
            lines_dict[n] = line_seg
            n = n + 1

        if n > 20000:  # expectation if loop should get suck
            raise FindingBorderError(
                "found more than 20000 cell borders; something went wrong"
            )

    # plot to confirm correct lines
    # plt.figure()
    # plt.imshow(graph_to_mask(graph,points,mask_boundaries.shape))
    # for seg_ps in lines_dict.values():
    #    for i, p in enumerate(seg_ps):
    #        plt.plot(points[p, 1], points[p, 0], "o")
    #        plt.text(points[p, 1], points[p, 0], str(i))

    return lines_dict


def find_neighbor_lines(
    graph,
    start_ps,
    other_endpoint,
    own_points,
    end_points,
    visited=None,
    neighbours=None,
):
    """
    recursive function to find neighbouring line. Explores the graph around the endpoint of a line. Notes the id of
    the line if it hits another line. Doesn't explore any points beyond the endpoints of lines.
    it reaches an intersection (point with three neighbours)
    :param graph: graph as a dictionary
    :param start_ps: start point as a list with len == 1
    :param other_endpoint:
    :param own_points: all points in own line
    :param end_points: # list of all endpoints
    :param visited: list of visited nodes. Is filled during recursion
    :param neighbours: id of neighbouring lines

    :return: visited: list of visited nodes
    :return: neighbours
    """
    if neighbours is None:
        neighbours = []
    if visited is None:
        visited = []
    visited = visited + start_ps  # update visited list  ## start must already be a list

    next_ps = [graph[s] for s in start_ps]  # next points
    next_ps = [p for ps in next_ps for p in ps]  # flatten
    next_ps = list(np.unique(next_ps))  # removing duplication

    # avoid connecting the two endpoints on own line directly
    # only true in first "iteration layer"
    if other_endpoint in next_ps and len(visited) == 1:
        next_ps.remove(other_endpoint)

    # remove if point is in visited list or in own line
    for p in copy.deepcopy(
        next_ps
    ):  # change in the list while iterating is not a nice idea-->
        if p in visited or p in own_points:
            next_ps.remove(p)

    # extract if point can be found in other line
    for p in copy.deepcopy(
        next_ps
    ):  # change in the list while iterating is not a nice idea--> make a copy
        if p in end_points:
            next_ps.remove(p)
            neighbours.append(p)

    # use other points for next iteration layer:
    if len(next_ps) == 0:  # stop recursion if no more next points are left
        return visited, neighbours

    visited, neighbours = find_neighbor_lines(
        graph,
        next_ps,
        other_endpoint,
        own_points,
        end_points,
        visited=visited,
        neighbours=neighbours,
    )
    # return when iteration is finished
    if visited:
        return visited, neighbours


def find_exact_line_endpoints(lines_points, points, graph):
    """
    function to find the exact meeting points of lines.
    First find the next closes points on neighbouring lines by exploring the graph. Then calculates a
    new endpoint as center of mass of these neighbouring points. Results are stored in a separate dictionary to be
    used in spline interpolation
    :param lines_points: dictionary with line_id: list of all points in correct order
    :param points: array of point coordinates
    :param graph: dictionary with connectivity of points
    :return: lines_endpoints_com: dictionary with the line_id:[new endpoint at start, new_endpoint at end]
    """

    end_points = [
        [ps[0], ps[-1]] for ps in lines_points.values()
    ]  # all end points in lines
    end_points = [p for ps in end_points for p in ps]

    # finding all neighbouring edpoints for one endpoint of a line
    lines_endpoints = {}
    for line, l_points in lines_points.items():
        # points on the line without both endpoints,
        # otherwise the algorithm can connect two endpoints on the same line
        l_points_core = l_points[1:-1]
        end1 = l_points[0]
        end2 = l_points[-1]
        v, neighbours1 = find_neighbor_lines(
            graph, [end1], end2, l_points_core, end_points, visited=[], neighbours=[]
        )
        v, neighbours2 = find_neighbor_lines(
            graph, [end2], end1, l_points_core, end_points, visited=[], neighbours=[]
        )
        lines_endpoints[line] = (
            neighbours1 + [end1],
            neighbours2 + [end2],
        )  # also adding own endpoints here
        # note adding endpoints after find_neighbour_lines is easiest
    # calculate new endpoints:
    # just center of mass of the endpoints
    # write to new dictionary and use this in splines calculation, without any points in between
    lines_endpoints_com = {}
    for line, endpoints in lines_endpoints.items():
        com1 = np.mean(points[np.array(endpoints[0])], axis=0)
        com2 = np.mean(points[np.array(endpoints[1])], axis=0)
        # line from new end points to old endpoints:
        lines_endpoints_com[line] = com1, com2

    return lines_endpoints_com, lines_endpoints
