#!/usr/bin/env python
# coding: utf-8

from saeno.FiniteBodyForces import FiniteBodyForces
    
# initialize the object
M = FiniteBodyForces()

from saeno.materials import SemiAffineFiberMaterial

# provide a material model
material = SemiAffineFiberMaterial(1645, 0.0008, 0.0075, 0.033)
M.setMaterialModel(material)

import numpy as np

# define the coordinates of the nodes of the mesh
# the array has to have the shape N_n x 3
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
# a boolean matrix with shape N_n
#                   0      1      2      3      4      5      6      7
var = np.array([False, False,  True,  True, False,  True,  True, False])

# the initial displacements of the nodes
# if the node is fixed (e.g. not variable) than this displacement will be fixed
# during the solving
U = np.array([[ 0.  ,  0.  ,  0.  ],  # 0
              [ 0.  ,  0.  ,  0.  ],  # 1
              [ 0.  ,  0.  ,  0.  ],  # 2
              [ 0.  ,  0.  ,  0.  ],  # 3
              [ 0.  ,  0.  ,  0.  ],  # 4
              [ 0.  ,  0.  ,  0.  ],  # 5
              [ 0.  ,  0.  ,  0.  ],  # 6
              [ 0.  ,  0.  ,  0.  ]]) # 7

# for the variable nodes, we can specify the target force.
# this is the force that the material applies after solving onto the nodes
# therefore for a pull to the right (positive x-direction) we have to provide
# a target force to the left (negative x-direction)
F_ext = np.array([[ 0. ,  0.  ,  0.  ],  # 0
                  [ 0. ,  0.  ,  0.  ],  # 1
                  [-2.5,  0.  ,  0.  ],  # 2
                  [-2.5,  0.  ,  0.  ],  # 3
                  [ 0. ,  0.  ,  0.  ],  # 4
                  [-2.5,  0.  ,  0.  ],  # 5
                  [-2.5,  0.  ,  0.  ],  # 6
                  [ 0. ,  0.  ,  0.  ]]) # 7

# provide the node data
M.setNodes(R, var, U, F_ext)
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

# the resulting displacement
np.set_printoptions(precision=3, suppress=True)
M.U

# the resulting forces on the nodes
M.f_glo

# the new position of the nodes
M.R+M.U

# visualize the meshes
M.plotMesh(use_displacement=False)
M.plotMesh(use_displacement=True)

