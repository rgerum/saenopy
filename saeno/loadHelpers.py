import os
import numpy as np

from .multigridHelper import makeBoxmeshCoords, makeBoxmeshTets, setActiveFields


def load(filename, *args, **kwargs):
    file2 = filename[:-4] + ".npy"
    if not os.path.exists(file2) or os.path.getmtime(filename) > os.path.getmtime(file2):
        print("Load changed file", filename)
        data = np.loadtxt(filename, *args, **kwargs)
        np.save(file2, data)
    else:
        data = np.load(file2)
    return data


def makeBoxmesh(mesh, CFG):
    mesh.currentgrain = 1

    nx = CFG["BM_N"]
    dx = CFG["BM_GRAIN"]

    rin = CFG["BM_RIN"]
    mulout = CFG["BM_MULOUT"]
    rout = nx * dx * 0.5

    if rout < rin:
        print("WARNING in makeBoxmesh: Mesh BM_RIN should be smaller than BM_MULOUT*BM_GRAIN*0.5")

    print("coords")
    mesh.setNodes(makeBoxmeshCoords(dx, nx, rin, mulout))
    print("tets")
    mesh.setTetrahedra(makeBoxmeshTets(nx, mesh.currentgrain))
    print("var")
    mesh.var = setActiveFields(nx, mesh.currentgrain, True)
    print("done")


def loadMeshCoords(fcoordsname):
    """
    Load the nodes. Each line represents a node and has 3 float entries for the x, y, and z coordinates of the
    node.
    """

    # load the node file
    data = load(fcoordsname, dtype=float)

    # check the data
    assert data.shape[1] == 3, "coordinates in " + fcoordsname + " need to have 3 columns for the XYZ"
    print("%s read (%d entries)" % (fcoordsname, data.shape[0]))

    return data


def loadMeshTets(ftetsname):
    """
    Load the tetrahedrons. Each line represents a tetrahedron. Each line has 4 integer values representing the node
    indices.
    """
    # load the data
    data = load(ftetsname, dtype=int)

    # check the data
    assert data.shape[1] == 4, "node indices in " + ftetsname + " need to have 4 columns, the indices of the nodes of the 4 corners fo the tetrahedron"
    print("%s read (%d entries)" % (ftetsname, data.shape[0]))

    # the loaded data are the node indices but they start with 1 instead of 0 therefore "-1"
    return data - 1


def loadBeams(self, fbeamsname):
    return np.loadtxt(fbeamsname)


def loadBoundaryConditions(dbcondsname, N_c=None):
    """
    Loads a boundary condition file "bcond.dat".

    It has 4 values in each line.
    If the last value is 1, the other 3 define a force on a variable node
    If the last value is 0, the other 3 define a displacement on a fixed node
    """
    # load the data in the file
    data = np.loadtxt(dbcondsname)
    assert data.shape[1] == 4, "the boundary conditions need 4 columns"
    if N_c is not None:
        assert data.shape[0] == N_c, "the boundary conditions need to have the same count as the number of nodes"
    print("%s read (%d x %d entries)" % (dbcondsname, data.shape[0], data.shape[1]))

    # the last column is a bool whether the node is fixed or not
    var = data[:, 3] > 0.5
    # if it is fixed, the other coordinates define the displacement
    U = np.zeros((var.shape[0], 3))*np.nan
    U[~var] = data[~var, :3]
    # if it is variable, the given vector is the force on the node
    f_ext = np.zeros((var.shape[0], 3))*np.nan
    f_ext[var] = data[var, :3]

    # update the connections (as they only contain non-fixed nodes)
    return var, U, f_ext


def loadConfiguration(Uname, N_c=None):
    """
    Load the displacements for the nodes. The file has to have 3 columns for the displacement in XYZ and one
    line for each node.
    """
    data = np.loadtxt(Uname)
    assert data.shape[1] == 3, "the displacement file needs to have 3 columnds"
    if N_c is not None:
        assert data.shape[0] == N_c, "there needs to be a displacement for each node"
    print("%s read (%d entries)" % (Uname, data.shape[0]))

    # store the displacement
    return data