#!/usr/bin/env python
# coding: utf-8



from saenopy import Solver
import saenopy
# initialize the object
M = Solver()

from saenopy.materials import SemiAffineFiberMaterial

# provide a material model
material = SemiAffineFiberMaterial(1645, 0.0008, 1.0075, 0.033)
M.setMaterialModel(material)

import numpy as np

# define the coordinates of the nodes of the mesh
# the array has to have the shape N_n x 3
R = np.array([[0., 0., 0.],  # 0
              [0., 1., 0.],  # 1
              [1., 1., 0.],  # 2
              [1., 0., 0.],  # 3
              [0., 0., 1.],  # 4
              [0., 1., 1.],  # 5
              [1., 1., 1.],  # 6
              [1., 0., 1.]]) # 7

# define the concetivity of the mesh (only tetrahedra are allowed)
# the array has to have the shape N_t x 4
# every entry is an index referencing a node in R (indices start with 0)
T = np.array([[0, 1, 3, 5],
              [1, 2, 3, 5],
              [0, 5, 3, 4],
              [4, 5, 3, 7],
              [5, 2, 3, 6],
              [3, 5, 6, 7]])

# provide the node data
M.setNodes(R)
# the tetrahedron data
M.setTetrahedra(T)

# the displacement boundary conditions of the nodes
# if a displacement boundary condition is given, the node will be fixed
U = np.array([[  0.  ,   0.  ,   0.  ],  # 0
              [  0.  ,   0.  ,   0.  ],  # 1
              [np.nan, np.nan, np.nan],  # 2
              [np.nan, np.nan, np.nan],  # 3
              [  0.  ,   0.  ,   0.  ],  # 4
              [  0.  ,   0.  ,   0.  ],  # 5
              [np.nan, np.nan, np.nan],  # 6
              [np.nan, np.nan, np.nan]]) # 7

# the force boundary conditions of the nodes
# if a target force boundary condition is given, the node will be free
# this is the force that the material applies after solving onto the nodes
# therefore for a pull to the right (positive x-direction) we have to provide
# a target force to the left (negative x-direction)
F_ext = np.array([[np.nan, np.nan, np.nan],  # 0
                  [np.nan, np.nan, np.nan],  # 1
                  [-2.5  ,  0.   ,  0.   ],  # 2
                  [-2.5  ,  0.   ,  0.   ],  # 3
                  [np.nan, np.nan, np.nan],  # 4
                  [np.nan, np.nan, np.nan],  # 5
                  [-2.5  ,  0.   ,  0.   ],  # 6
                  [-2.5  ,  0.   ,  0.   ]]) # 7

# and the boundary condition
M.setBoundaryCondition(U, F_ext)

# relax the mesh and move the "varible" nodes
M.solve_boundarycondition();


# visualize the meshes
M.viewMesh(50, 0.1)

