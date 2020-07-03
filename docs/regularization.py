#!/usr/bin/env python
# coding: utf-8


from saenopy import Solver
    
# initialize the object
M = Solver()

from saenopy.materials import SemiAffineFiberMaterial

# provide a material model
material = SemiAffineFiberMaterial(1645, 0.0008, 1.0075, 0.033)
M.setMaterialModel(material)

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

# provide the node data
M.setNodes(R)
# and the tetrahedron data
M.setTetrahedra(T)

# the displacements of the nodes which shall be fitted
# during the solving
U = np.array([[0   , 0, 0],  # 0
              [0   , 0, 0],  # 1
              [0.01, 0, 0],  # 2
              [0.01, 0, 0],  # 3
              [0   , 0, 0],  # 4
              [0.01, 0, 0],  # 5
              [0.01, 0, 0],  # 6
              [0   , 0, 0]]) # 7

# hand the displacements over to the class instance
M.setTargetDisplacements(U)

# call the regularisation
M.solve_regularized(stepper=0.1, alpha=0.001);


M.viewMesh(50, 1)

