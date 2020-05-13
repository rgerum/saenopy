Mesh
====

Saenopy uses only thetrahedral meshes. The mesh is defined by the N nodes and by the connectivity of the nodes, i.e.
by M thetrahedra.

The N nodes are defined by an Nx3 array that contains the position of each node in the three spatial dimensions.

Nodes can for example be loaded from a .txt file structured like this::

    0. 0. 0.
    0. 1. 0.
    1. 1. 0.
    1. 0. 0.
    0. 0. 1.
    0. 1. 1.
    1. 1. 1.
    1. 0. 1.

It can be loaded with the following commands (see :py:meth:`~.Solver.setNodes`):

>>> import numpy as np
>>> from saenopy import Solver
>>> M = Solver()
>>> M.setNodes(np.loadtxt("nodes.txt"))

The connectivity is defined by M thetraedra which reference the nodes.
Therefore, it is a Mx4 array of integers ranging from 0 to N.

The connectivity can be loaded from a .txt file structured like this::

    0 1 3 5
    1 2 3 5
    0 5 3 4
    4 5 3 7
    5 2 3 6
    3 5 6 7

It can be loaded with the following commands (see :py:meth:`~.Solver.setTetrahedra`):

>>> M.setTetrahedra(np.loadtxt("connectivity.txt"))



