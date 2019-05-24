#!/usr/bin/env python
# coding: utf-8

from saeno.FiniteBodyForces import FiniteBodyForces
    
# initialize the object
M = FiniteBodyForces()

from saeno.buildEpsilon import buildEpsilon

# provide a material model
epsilon = buildEpsilon(1645, 0.0008, 0.0075, 0.033)
M.setMaterialModel(epsilon)

import numpy as np

# define the coordinates of the nodes of the mesh
# the array has to have the shape N_v x 3
R = np.array([[0., 0., 0.],  # 0
              [0., 1., 0.],  # 1
              [1., 1., 0.],  # 2
              [1., 0., 0.],  # 3
              [0., 0., 1.],  # 4
              [1., 0., 1.],  # 5
              [1., 1., 1.],  # 6
              [0., 1., 1.]]) # 7

# define the tetrahedra of the mesh
# the array has to have the shape N_t x 4
# every entry is an index referencing a verces in R (indices start with 0)
T = np.array([[0, 1, 7, 2],
              [0, 2, 5, 3],
              [0, 4, 5, 7],
              [2, 5, 6, 7],
              [0, 7, 5, 2]])

# define if the nodes are "variable", e.g. allowed to be moved by the solver
# a boolean matrix with shape N_v
#                   0      1      2      3      4      5      6      7
var = np.array([ True,  True, False, False,  True, False, False,  True])

# the initial displacements of the nodes
# if the node is fixed (e.g. not variable) than this displacement will be fixed
# during the solving
U = np.array([[ 0.  ,  0.  ,  0.  ],  # 0
              [ 0.  ,  0.  ,  0.  ],  # 1
              [-0.25,  0.  ,  0.  ],  # 2
              [-0.25,  0.  ,  0.  ],  # 3
              [ 0.  ,  0.  ,  0.  ],  # 4
              [-0.25,  0.  ,  0.  ],  # 5
              [-0.25,  0.  ,  0.  ],  # 6
              [ 0.  ,  0.  ,  0.  ]]) # 7

# provide the node data
M.setNodes(R, var, U)
# and the tetrahedron data
M.setTetrahedra(T)

# relax the mesh and move the "varible" nodes
M.relax()

# store the forces of the nodes
M.storeF("F.dat")
# store the positions and the displacements
M.storeRAndU("R.dat", "U.dat")
# store the center of each tetrahedron and a combined list with energies and volumina of the tetrahedrons
M.storeEandV("RR.dat", "EV.dat")

