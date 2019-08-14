#!/usr/bin/env python
# coding: utf-8




from saeno import FiniteBodyForces
import saeno
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
              [0., 1., 1.],  # 5
              [1., 1., 1.],  # 6
              [1., 0., 1.]]) # 7

# define the tetrahedra of the mesh
# the array has to have the shape N_t x 4
# every entry is an index referencing a verces in R (indices start with 0)
T = np.array([[0, 1, 5, 2],
              [0, 2, 7, 3],
              [0, 4, 7, 5],
              [2, 7, 6, 5],
              [0, 5, 7, 2]])
T = np.array([[0, 1, 3, 5],
             [1, 2, 3, 5],
             [0, 5, 3, 4],
             [4, 5, 3, 7],
             [5, 2, 3, 6],
             [3, 5, 6, 7]])

# the initial displacements of the nodes
# if the node is fixed (e.g. not variable) than this displacement will be fixed
# during the solving
U = np.array([[  0.  ,   0.  ,   0.  ],  # 0
              [  0.  ,   0.  ,   0.  ],  # 1
              [np.nan, np.nan, np.nan],  # 2
              [np.nan, np.nan, np.nan],  # 3
              [  0.  ,   0.  ,   0.  ],  # 4
              [  0.  ,   0.  ,   0.  ],  # 5
              [np.nan, np.nan, np.nan],  # 6
              [np.nan, np.nan, np.nan]]) # 7

# for the variable nodes, we can specify the target force.
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


# provide the node data
M.setNodes(R)
# the tetrahedron data
M.setTetrahedra(T)
# and the boundary condition
M.setBoundaryCondition(U, F_ext)

U2 = np.zeros_like(M.R)*np.nan
F_ext2 = np.zeros_like(M.R)
U2[M.R[:, 0] < 0.001] = 0
U2[M.R[:, 0] > 1-0.001] = 0
U2[M.R[:, 0] > 1-0.001, 0] = 0.1
#F_ext2[np.isnan(U2) & np.isnan(F_ext2)] = 0
M.setBoundaryCondition(U2, F_ext2)

for i in range(1):
    T, R, U = M.T, M.R, M.U
    R = np.hstack((R, U))
    T2, R2 = saeno.multigridHelper.subdivideTetrahedra(T, R)
    U = R2[:, 3:6]
    R2 = R2[:, :3]
    M.setNodes(R2)
    M.setTetrahedra(T2)
    M.U = U
print(saeno.multigridHelper.getTetrahedraVolumnes(M.T, M.R))
#M.viewMesh(10, 1)

print(M.T.shape)
print(M.var.shape)
M._check_relax_ready()
M._prepare_temporary_quantities()
M._updateGloFAndK()
print(M.E.shape, np.sum(M.E), M.E_glo)


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
M.f
np.mean(M.f[M.R[:, 0] < 0.5], axis=0)

np.mean(M.f[M.R[:, 0] > 0.5], axis=0)

# the new position of the nodes
M.R+M.U

np.nansum(M.f-M.f_target)

np.nansum(M.U-M.U_fixed)

np.sum(np.linalg.norm(M.U, axis=1))

# visualize the meshes
M.viewMesh(50, 0.1)

M._check_relax_ready()

# the displacements of the nodes which shall be fitted
# during the solving
U = np.array([[0   , 0, 0],  # 0
              [0   , 0, 0],  # 1
              [0.1, 0, 0],  # 2
              [0.1, 0, 0],  # 3
              [0   , 0, 0],  # 4
              [0.1, 0, 0],  # 5
              [0.1, 0, 0],  # 6
              [0   , 0, 0]]) # 7
M.U = U

M._prepare_temporary_quantities()
M._updateGloFAndK()
M.f

# visualize the meshes
M.viewMesh(50, 0.1)



