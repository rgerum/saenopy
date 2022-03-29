import os
import sys
import time

import numpy as np
import scipy.sparse as ssp

from numba import jit, njit
from typing import Union

from saenopy.multigridHelper import getLinesTetrahedra, getLinesTetrahedra2
from saenopy.buildBeams import buildBeams
from saenopy.materials import Material, SemiAffineFiberMaterial
from saenopy.conjugateGradient import cg
from saenopy.loadHelpers import Saveable
from typing import List
from pathlib import Path
import natsort
from saenopy.getDeformations import getStack, Stack, format_glob

class Mesh(Saveable):
    __save_parameters__ = ['R', 'T', 'node_vars']
    R = None  # the 3D positions of the vertices, dimension: N_c x 3
    T = None  # the tetrahedra' 4 corner vertices (defined by index), dimensions: N_T x 4

    node_vars = None

    def __init__(self, R=None, T=None, node_vars=None):
        if R is not None:
            self.setNodes(R)
        if T is not None:
            self.setTetrahedra(T)
        self.node_vars = {}
        if node_vars is not None:
            for name, data in node_vars.items():
                self.setNodeVar(name, data)

    def setNodes(self, data: np.ndarray):
        """
        Provide mesh coordinates.

        Parameters
        ----------
        data : ndarray
            The coordinates of the vertices. Dimensions Nx3
        """
        # check the input
        data = np.asarray(data)
        assert len(data.shape) == 2, "Mesh node data needs to be Nx3."
        assert data.shape[1] == 3, "Mesh vertices need to have 3 spacial coordinate."

        # store the loaded node coordinates
        self.R = data.astype(np.float64)


    def setTetrahedra(self, data: np.ndarray):
        """
        Provide mesh connectivity. Nodes have to be connected by tetrahedra. Each tetraherdon consts of the indices of
        the 4 vertices which it connects.

        Parameters
        ----------
        data : ndarray
            The node indices of the 4 corners. Dimensions Nx4
        """
        # check the input
        data = np.asarray(data)
        assert len(data.shape) == 2, "Mesh tetrahedra needs to be Nx4."
        assert data.shape[1] == 4, "Mesh tetrahedra need to have 4 corners."
        assert 0 <= data.min(), "Mesh tetrahedron node indices are not allowed to be negative."
        assert data.max() < self.R.shape[0], "Mesh tetrahedron node indices cannot be bigger than the number of vertices."

        # store the tetrahedron data (needs to be int indices)
        self.T = data.astype(np.int)

    def setNodeVar(self, name, data):
        data = np.asarray(data)
        assert len(data.shape) == 1 or len(data.shape) == 2, "Node var needs to be scalar or vectorial"
        if len(data.shape) == 2:
            assert data.shape[1] == 3, "Vectrial node var needs to have 3 dimensionts"
        assert data.shape[0] == self.R.shape[0], "Node var has to have the same number of nodes than the mesh"
        self.node_vars[name] = data

    def getNodeVar(self, name):
        return self.node_vars[name]
    
    def hasNodeVar(self, name):
        return name in self.node_vars


class Solver(Saveable):
    __save_parameters__ = ["R", "T", "U", "f", "U_fixed", "U_target", "U_target_mask", "f_target", "E_glo", "var", "regularisation_results",
                      "reg_mask", "regularisation_parameters", "material_model"]

    R = None  # the 3D positions of the vertices, dimension: N_c x 3
    T = None  # the tetrahedra' 4 corner vertices (defined by index), dimensions: N_T x 4
    E = None  # the energy stored in each tetrahedron, dimensions: N_T
    V = None  # the volume of each tetrahedron, dimensions: N_T
    var = None  # a bool if a node is movable

    Phi = None  # the shape tensor of each tetrahedron, dimensions: N_T x 4 x 3
    Phi_valid = False
    U = None  # the displacements of each node, dimensions: N_c x 3
    U_fixed = None
    U_target = None
    U_target_mask = None
    reg_mask = None

    f = None  # the global forces on each node, dimensions: N_c x 3
    f_target = None  # the external forces on each node, dimensions: N_c x 3
    K_glo = None  # the global stiffness tensor, dimensions: N_c x N_c x 3 x 3

    Laplace = None

    E_glo = 0  # the global energy

    # a list of all vertices are connected via a tetrahedron, stored as pairs: dimensions: N_connections x 2
    connections = None
    connections_valid = False

    N_T = 0  # the number of tetrahedra
    N_c = 0  # the number of vertices

    s = None  # the beams, dimensions N_b x 3
    N_b = 0  # the number of beams

    material_model: SemiAffineFiberMaterial = None  # the function specifying the material model
    material_parameters = None

    verbose = False

    preprocessing = None
    '''
    preprocessing = [
        "load_stack": ["1_*.tif", "2_*.tif", voxel_sixe],
        "3D_piv": [overlap, windowsize, signoise, drifcorrection],
        "iterpolate_mesh": [],
    ]
    '''

    def setNodes(self, data: np.ndarray):
        """
        Provide mesh coordinates.

        Parameters
        ----------
        data : ndarray
            The coordinates of the vertices. Dimensions Nx3
        """
        # check the input
        data = np.asarray(data)
        assert len(data.shape) == 2, "Mesh node data needs to be Nx3."
        assert data.shape[1] == 3, "Mesh vertices need to have 3 spacial coordinate."

        # store the loaded node coordinates
        self.R = data.astype(np.float64)
        # schedule to recalculate the shape tensors
        self.Phi_valid = False

        # store the number of vertices
        self.N_c = data.shape[0]

        self.var = np.ones(self.N_c, dtype=np.bool)
        self.U = np.zeros((self.N_c, 3))
        self.f = np.zeros((self.N_c, 3))
        self.f_target = np.zeros((self.N_c, 3))

    def setBoundaryCondition(self, displacements: np.ndarray = None, forces: np.ndarray = None):
        """
        Provide the boundary condition for the mesh, to be used with :py:meth:`~.Solver.solve_nonregularized`.

        Parameters
        ----------
        displacements : ndarray, optional
            If the displacement of a node is not nan, it is treated as a Dirichlet boundary condition and the
            displacement of this node is kept fixed during solving. Dimensions Nx3
        forces : ndarray, optional
            If the force of a node is not nan, it is treated as a von Neumann boundary condition and the solver tries to
            match the force on the node with the here given force. Dimensions Nx3
        """

        # initialize 0 displacement for each node
        if displacements is None:
            self.U = np.zeros((self.N_c, 3))
        else:
            displacements = np.asarray(displacements, dtype=np.float64)
            assert displacements.shape == (self.N_c, 3)
            self.var = np.any(np.isnan(displacements), axis=1)
            self.U_fixed = displacements
            self.U[~self.var] = displacements[~self.var]

        # initialize global and external forces
        if forces is None:
            self.f_target = np.zeros((self.N_c, 3))
        else:
            self._setExternalForces(forces)
            # if no displacements where given, take the variable nodes from the nans in the force list
            if displacements is None:
                self.var = ~np.any(np.isnan(forces), axis=1)
            # if not, check if the the fixed displacements have no force
            elif np.all(np.isnan(self.f_target[~self.var])) is False:
                print("WARNING: Forces for fixed vertices were specified. These boundary conditions cannot be"
                      "fulfilled", file=sys.stderr)

    def setInitialDisplacements(self, displacements: np.ndarray):
        """
        Provide initial displacements of the nodes. For fixed nodes these displacements are ignored.

        Parameters
        ----------
        displacements : ndarray
            The list of displacements. Dimensions Nx3
        """
        # check the input
        displacements = np.asarray(displacements)
        assert displacements.shape == (self.N_c, 3)
        self.U[self.var] = displacements[self.var].astype(np.float64)

    def _setExternalForces(self, forces: np.ndarray):
        """
        Provide external forces that act on the vertices. The forces can also be set with
        :py:meth:`~.Solver.setNodes` directly with the vertices.

        Parameters
        ----------
        forces : ndarray
            The list of forces. Dimensions Nx3
        """
        # check the input
        forces = np.asarray(forces)
        assert forces.shape == (self.N_c, 3)
        self.f_target = forces.astype(np.float64)

    def setTetrahedra(self, data: np.ndarray):
        """
        Provide mesh connectivity. Nodes have to be connected by tetrahedra. Each tetraherdon consts of the indices of
        the 4 vertices which it connects.

        Parameters
        ----------
        data : ndarray
            The node indices of the 4 corners. Dimensions Nx4
        """
        # check the input
        data = np.asarray(data)
        assert len(data.shape) == 2, "Mesh tetrahedra needs to be Nx4."
        assert data.shape[1] == 4, "Mesh tetrahedra need to have 4 corners."
        assert 0 <= data.min(), "Mesh tetrahedron node indices are not allowed to be negative."
        assert data.max() < self.N_c, "Mesh tetrahedron node indices cannot be bigger than the number of vertices."

        # store the tetrahedron data (needs to be int indices)
        self.T = data.astype(np.int)

        # the number of tetrahedra
        self.N_T = data.shape[0]

        # Phi is a 4x3 tensor for every tetrahedron
        self.Phi = np.zeros((self.N_T, 4, 3))

        # initialize the volume and energy of each tetrahedron
        self.V = np.zeros(self.N_T)
        self.E = np.zeros(self.N_T)

        # schedule to recalculate the shape tensors
        self.Phi_valid = False

        # schedule to recalculate the connections
        self.connections_valid = False

    def setMaterialModel(self, material: Material, generate_lookup=True):
        """
        Provides the material model.

        Parameters
        ----------
        material : :py:class:`~.materials.Material`
             The material, must be of a subclass of Material.
        """
        self.material_model = material
        if generate_lookup is True:
            self.material_model_look_up = self.material_model.generate_look_up_table()

    def setBeams(self, beams: Union[int, np.ndarray] = 300):
        """
        Sets the beams for the calculation over the whole solid angle.

        Parameters
        ----------
        beams : int, ndarray
            Either an integer which defines in how many beams to discretize the whole solid angle or an ndarray providing
            the beams, dimensions Nx3, default 300
        """
        if isinstance(beams, int):
            beams = buildBeams(beams)
        self.s = beams
        self.N_b = beams.shape[0]

    def _computeConnections(self):
        from scipy.sparse.sputils import get_index_dtype
        # calculate the indices for "update_f_glo"
        y, x = np.meshgrid(np.arange(3), self.T.ravel())
        self.force_distribute_coordinates = (x.ravel(), y.ravel())

        self.force_distribute_coordinates = tuple(self.force_distribute_coordinates[i].astype(dtype=get_index_dtype(maxval=max(self.force_distribute_coordinates[i].shape))) for i in range(2))

        # calculate the indices for "update_K_glo"
        @njit()
        def numba_get_pair_coordinates(T, var):
            stiffness_distribute_coordinates2 = []
            filter_in = np.zeros((T.shape[0], 4, 4, 3, 3)) == 1
            # iterate over all tetrahedra
            for t in range(T.shape[0]):
                #if t % 1000:
                #    print(t, T.shape[0])
                tet = T[t]
                # over all corners
                for t1 in range(4):
                    c1 = tet[t1]

                    if not var[c1]:
                        continue

                    filter_in[t, t1, :, :, :] = True

                    for t2 in range(4):
                        # get two vertices of the tetrahedron
                        c2 = tet[t2]

                        for i in range(3):
                            for j in range(3):
                                # add the connection to the set
                                stiffness_distribute_coordinates2.append((c1*3+i, c2*3+j))
            stiffness_distribute_coordinates2 = np.array(stiffness_distribute_coordinates2)
            return filter_in.ravel(), (stiffness_distribute_coordinates2[:, 0], stiffness_distribute_coordinates2[:, 1])

        self.filter_in, self.stiffness_distribute_coordinates2 = numba_get_pair_coordinates(self.T, self.var)
        self.stiffness_distribute_coordinates2 = np.array(self.stiffness_distribute_coordinates2, dtype=get_index_dtype(maxval=max(self.stiffness_distribute_coordinates2[0].shape)))

        # remember that for the current configuration the connections have been calculated
        self.connections_valid = True

    def _computePhi(self):
        """
        Calculate the shape tensors of the tetrahedra (see page 49)
        """
        # define the helper matrix chi
        Chi = np.zeros((4, 3))
        Chi[0, :] = [-1, -1, -1]
        Chi[1, :] = [1, 0, 0]
        Chi[2, :] = [0, 1, 0]
        Chi[3, :] = [0, 0, 1]

        # tetrahedron matrix B (linear map of the undeformed tetrahedron T onto the primitive tetrahedron P)
        B = self.R[self.T[:, 1:4]] - self.R[self.T[:, 0]][:, None, :]
        B = B.transpose(0, 2, 1)

        # calculate the volume of the tetrahedron
        self.V = np.abs(np.linalg.det(B)) / 6.0
        sum_zero = np.sum(self.V == 0)
        if np.sum(self.V == 0):
            print("WARNING: found %d elements with volume of 0. Removing those elements." % sum_zero)
            self.setTetrahedra(self.T[self.V != 0])
            return self._computePhi()

        # the shape tensor of the tetrahedron is defined as Chi * B^-1
        self.Phi = Chi @ np.linalg.inv(B)

        # remember that for the current configuration the shape tensors have been calculated
        self.Phi_valid = True

    """ relaxation """

    def _prepare_temporary_quantities(self):
        # test if one node of the tetrahedron is variable
        # only count the energy if not the whole tetrahedron is fixed
        self._countEnergy = np.any(self.var[self.T], axis=1)

        # and the shape tensor with the beam
        # s*_tmb = Phi_tmj * s_jb  (t in [0, N_T], i,j in {x,y,z}, m in {1,2,3,4}), b in [0, N_b])
        self._s_star = self.Phi @ self.s.T

        self._V_over_Nb = np.expand_dims(self.V, axis=1) / self.N_b

    def _updateGloFAndK(self):
        """
        Calculates the stiffness matrix K_ij, the force F_i and the energy E of each node.
        """
        t_start = time.time()
        batchsize = 1000

        self.E_glo = 0
        f_glo = np.zeros((self.N_T, 4, 3))
        K_glo = np.zeros((self.N_T, 4, 4, 3, 3))

        for i in range(int(np.ceil(self.T.shape[0]/batchsize))):
            if self.verbose:
                print("updating forces and stiffness matrix %d%%" % (i/int(np.ceil(self.T.shape[0]/batchsize))*100), end="\r")
            t = slice(i*batchsize, (i+1)*batchsize)

            s_bar = self._get_s_bar(t)

            epsilon_b, dEdsbar, dEdsbarbar = self._get_applied_epsilon(s_bar, self.material_model_look_up, self._V_over_Nb[t])

            self._update_energy(epsilon_b, t)
            self._update_f_glo(self._s_star[t], s_bar, dEdsbar, out=f_glo[t])
            self._update_K_glo(self._s_star[t], s_bar, dEdsbar, dEdsbarbar, out=K_glo[t])

        # store the global forces in self.f_glo
        # transform from N_T x 4 x 3 -> N_v x 3
        ssp.coo_matrix((f_glo.ravel(), self.force_distribute_coordinates), shape=self.f.shape).toarray(out=self.f)

        # store the stiffness matrix K in self.K_glo
        # transform from N_T x 4 x 4 x 3 x 3 -> N_v * 3 x N_v * 3
        self.K_glo = ssp.coo_matrix((K_glo.ravel()[self.filter_in], self.stiffness_distribute_coordinates2),
                                shape=(self.N_c*3, self.N_c*3)).tocsr()
        if self.verbose:
            print("updating forces and stiffness matrix finished %.2fs" % (time.time() - t_start))

    def getMaxTetStiffness(self):
        """
        Calculates the stiffness matrix K_ij, the force F_i and the energy E of each node.
        """
        t_start = time.time()
        batchsize = 10000

        tetrahedra_stiffness = np.zeros(self.T.shape[0])

        for i in range(int(np.ceil(self.T.shape[0]/batchsize))):
            if self.verbose:
                print("updating forces and stiffness matrix %d%%" % (i/int(np.ceil(self.T.shape[0]/batchsize))*100), end="\r")
            t = slice(i*batchsize, (i+1)*batchsize)

            s_bar = self._get_s_bar(t)

            s = np.linalg.norm(s_bar, axis=1)

            epsbarbar_b = self.material_model.stiffness(s - 1)

            tetrahedra_stiffness[t] = np.max(epsbarbar_b, axis=1)

        return tetrahedra_stiffness

    def _get_s_bar(self, t: np.ndarray):
        # get the displacements of all corners of the tetrahedron (N_Tx3x4)
        # u_tim  (t in [0, N_T], i in {x,y,z}, m in {1,2,3,4})
        # F is the linear map from T (the undeformed tetrahedron) to T' (the deformed tetrahedron)
        # F_tij = d_ij + u_tmi * Phi_tmj  (t in [0, N_T], i,j in {x,y,z}, m in {1,2,3,4})
        F = np.eye(3) + np.einsum("tmi,tmj->tij", self.U[self.T[t]], self.Phi[t])

        # multiply the F tensor with the beam
        # s'_tib = F_tij * s_jb  (t in [0, N_T], i,j in {x,y,z}, b in [0, N_b])
        s_bar = F @ self.s.T

        return s_bar

    @staticmethod
    #@njit()#nopython=True, cache=True)
    def _get_applied_epsilon(s_bar: np.ndarray, lookUpEpsilon: callable, _V_over_Nb: np.ndarray):
        # the "deformation" amount # p 54 equ 2 part in the parentheses
        # s_tb = |s'_tib|  (t in [0, N_T], i in {x,y,z}, b in [0, N_b])
        s = np.linalg.norm(s_bar, axis=1)

        epsilon_b, epsbar_b, epsbarbar_b = lookUpEpsilon(s - 1)

        #                eps'_tb    1
        # dEdsbar_tb = - ------- * --- * V_t
        #                 s_tb     N_b
        dEdsbar = - (epsbar_b / s) * _V_over_Nb

        #                  s_tb * eps''_tb - eps'_tb     1
        # dEdsbarbar_tb = --------------------------- * --- * V_t
        #                         s_tb**3               N_b
        dEdsbarbar = ((s * epsbarbar_b - epsbar_b) / (s ** 3)) * _V_over_Nb

        return epsilon_b, dEdsbar, dEdsbarbar

    def _update_energy(self, epsilon_b: np.ndarray, t: np.ndarray):
        # sum the energy of this tetrahedron
        # E_t = eps_tb * V_t
        self.E[t] = np.mean(epsilon_b, axis=1) * self.V[t]

        # only count the energy of the tetrahedron to the global energy if the tetrahedron has at least one
        # variable node
        self.E_glo += np.sum(self.E[t][self._countEnergy[t]])

    def _update_f_glo(self, s_star: np.ndarray, s_bar: np.ndarray, dEdsbar: np.ndarray, out: np.ndarray):
        # f_tmi = s*_tmb * s'_tib * dEds'_tb  (t in [0, N_T], i in {x,y,z}, m in {1,2,3,4}, b in [0, N_b])
        np.einsum("tmb,tib,tb->tmi", s_star, s_bar, dEdsbar, out=out)

    def _update_K_glo(self, s_star: np.ndarray, s_bar: np.ndarray, dEdsbar: np.ndarray, dEdsbarbar: np.ndarray, out: np.ndarray):
        #                              / |  |     \      / |  |     \                   / |    |     \
        #     ___             /  s'  w"| |s'| - 1 | - w' | |s'| - 1 |                w' | | s' | - 1 |             \
        # 1   \   *     *     |   b    \ | b|     /      \ | b|     /                   \ |  b |     /             |
        # -    > s   * s    * | ------------------------------------ * s' * s'  + ---------------------- * delta   |
        # N   /   bm    br    |                  |s'|³                  ib   lb             |s'  |              li |
        #  b  ---             \                  | b|                                       |  b |                 /
        #
        # (t in [0, N_T], i,l in {x,y,z}, m,r in {1,2,3,4}, b in [0, N_b])
        s_bar_s_bar = 0.5 * (np.einsum("tb,tib,tlb->tilb", dEdsbarbar, s_bar, s_bar)
                             - np.einsum("il,tb->tilb", np.eye(3), dEdsbar))

        np.einsum("tmb,trb,tilb->tmril", s_star, s_star, s_bar_s_bar, out=out, optimize=['einsum_path', (0, 1), (0, 1)])

    def _check_relax_ready(self):
        """
        Checks whether everything is loaded to start a relaxation process.
        """
        # check if we have nodes
        if self.R is None or self.R.shape[0] == 0:
            raise ValueError("No nodes have yet been set. Call setNodes first.")
        self.N_c = self.R.shape[0]

        # check if we have tetrahedra
        if self.T is None or self.T.shape[0] == 0:
            raise ValueError("No tetrahedra have yet been set. Call setTetrahedra first.")
        self.N_T = self.T.shape[0]
        self.E = np.zeros(self.N_T)

        # check if we have a material model
        if self.material_model is None:
            raise ValueError("No material model has been set. Call setMaterialModel first.")

        # if the beams have not been set yet, initialize them with the default configuration
        if self.s is None:
            self.setBeams()

        # if the shape tensors are not valid, calculate them
        if self.Phi_valid is False:
            self._computePhi()

        # if the connections are not valid, calculate them
        if self.connections_valid is False:
            self._computeConnections()

    def solve_boundarycondition(self, stepper: float = 0.066, i_max: int = 300, rel_conv_crit: float = 0.01, relrecname: str = None, verbose: bool = False, callback: callable = None):
        """
        Solve the displacement of the free nodes constraint to the boundary conditions.

        Parameters
        ----------
        stepper : float, optional
            How much of the displacement of each conjugate gradient step to apply. Default 0.066
        i_max : int, optional
            The maximal number of iterations for the relaxation. Default 300
        rel_conv_crit : float, optional
            If the relative standard deviation of the last 6 energy values is below this threshold, finish the iteration.
            Default 0.01
        relrecname : string, optional
            If a filename is provided, for every iteration the displacement of the conjugate gradient step, the global
            energy and the residuum are stored in this file.
        verbose : bool, optional
            If true print status during optimisation
        callback : callable, optional
            A function to call after each iteration (e.g. for a live plot of the convergence)
        """
        # set the verbosity level
        self.verbose = verbose

        # check if everything is prepared
        self._check_relax_ready()

        self._prepare_temporary_quantities()

        # update the forces and stiffness matrix
        self._updateGloFAndK()

        relrec = [[0, self.E_glo, np.sum(self.f[self.var] ** 2)]]

        start = time.time()
        # start the iteration
        for i in range(i_max):
            # move the displacements in the direction of the forces one step
            # but while moving the stiffness tensor is kept constant
            du = self._solve_CG(stepper)

            # update the forces on each tetrahedron and the global stiffness tensor
            self._updateGloFAndK()

            # sum all squared forces of non fixed nodes
            ff = np.sum((self.f[self.var] - self.f_target[self.var]) ** 2)
            #ff = np.sum(self.f[self.var] ** 2)

            # print and store status
            if self.verbose:
                print("Newton ", i, ": du=", du, "  Energy=", self.E_glo, "  Residuum=", ff)

            # log and store values (if a target file was provided)
            relrec.append([du, self.E_glo, ff])
            if relrecname is not None:
                np.savetxt(relrecname, relrec)
            if callback is not None:
                callback(self, relrec)

            # if we have passed 6 iterations calculate average and std
            if i > 6:
                # calculate the average energy over the last 6 iterations
                last_Es = np.array(relrec)[:-6:-1, 1]
                Emean = np.mean(last_Es)
                Estd = np.std(last_Es)/np.sqrt(5)  # the original formula just had /N instead of /sqrt(N)

                # if the iterations converge, stop the iteration
                if Estd / Emean < rel_conv_crit:
                    break

        # print the elapsed time
        finish = time.time()
        if self.verbose:
            print("| time for relaxation was", finish - start)

        self.boundary_results = relrec

        return relrec

    def _solve_CG(self, stepper: float):
        """
        Solve the displacements from the current stiffness tensor using conjugate gradient.
        """
        # calculate the difference between the current forces on the nodes and the desired forces
        ff = self.f - self.f_target

        # ignore the force deviations on fixed nodes
        ff[~self.var, :] = 0

        # solve the conjugate gradient which solves the equation A x = b for x
        # where A is the stiffness matrix K_glo and b is the vector of the target forces
        uu = cg(self.K_glo, ff.ravel(), maxiter=3 * self.N_c, tol=0.00001, verbose=self.verbose).reshape(ff.shape)

        # add the new displacements to the stored displacements
        self.U[self.var] += uu[self.var] * stepper
        # sum the applied displacements
        du = np.sum(uu[self.var] ** 2) * stepper * stepper

        # return the total applied displacement
        return du

    """ regularization """

    def setTargetDisplacements(self, displacement: np.ndarray, reg_mask: np.ndarray=None):
        """
        Provide the displacements that should be fitted by the regularization.

        Parameters
        ----------
        displacement : ndarray
            If the displacement of a node is not nan, it is
            The displacements for each node. Dimensions N x 3
        """
        displacement = np.asarray(displacement)
        assert displacement.shape == (self.N_c, 3)
        self.U_target = displacement
        # only use displacements that are not nan
        self.U_target_mask = np.any(~np.isnan(displacement), axis=1)
        # regularisation mask
        if reg_mask is not None:
            assert reg_mask.shape == (self.N_c,), f"reg_mask should have the shape {(self.N_c,)} but has {reg_mask.shape}."
            assert reg_mask.dtype == bool, f"reg_mask should have the type bool but has {reg_mask.dtype}."
            self.reg_mask = reg_mask
        else:
            self.reg_mask = np.ones_like(displacement[:, 0]).astype(np.bool)

    def _updateLocalRegularizationWeigth(self, method: str):

        self.localweight[:] = 1

        Fvalues = np.linalg.norm(self.f, axis=1)
        #Fmedian = np.median(Fvalues[self.var])
        Fmedian = np.median(Fvalues[self.var & self.reg_mask])

        if method == "singlepoint":
            self.localweight[int(self.CFG["REG_FORCEPOINT"])] = 1.0e-10

        if method == "bisquare":
            k = 4.685

            index = Fvalues < k * Fmedian
            self.localweight[index * self.var] *= (1 - (Fvalues[index * self.var] / k / Fmedian) * (Fvalues[index * self.var] / k / Fmedian)) * (
                    1 - (Fvalues[index * self.var] / k / Fmedian) * (Fvalues[index * self.var] / k / Fmedian))
            self.localweight[~index * self.var] *= 1e-10

        if method == "cauchy":
            k = 2.385

            if Fmedian > 0:
                self.localweight[self.var] *= 1.0 / (1.0 + np.power((Fvalues / k / Fmedian), 2.0))
            else:
                self.localweight *= 1.0

        if method == "huber":
            k = 1.345

            index = (Fvalues > (k * Fmedian)) & self.var
            self.localweight[index] = k * Fmedian / Fvalues[index]

        if method == "L1":
            if Fmedian > 0:
                self.localweight[:] = 1 / Fvalues[:]
            else:
                self.localweight *= 1.0

        index = self.localweight < 1e-10
        self.localweight[index & self.var] = 1e-10

        self.localweight[~self.reg_mask] = 0

        counter = np.sum(1.0 - self.localweight[self.var])
        counterall = np.sum(self.var)

        if self.verbose:
            print("total weight: ", counter, "/", counterall)

    def _computeRegularizationAAndb(self, alpha: float):
        KA = self.K_glo.multiply(np.repeat(self.localweight * alpha, 3)[None, :])
        self.KAK = KA @ self.K_glo
        self.A = self.I + self.KAK

        self.b = (KA @ self.f.ravel()).reshape(self.f.shape)

        index = self.var & self.U_target_mask
        self.b[index] += self.U_target[index] - self.U[index]

    def _recordRegularizationStatus(self, relrec: list, alpha: float, relrecname: str = None):
        indices = self.var & self.U_target_mask
        btemp = self.U_target[indices] - self.U[indices]
        uuf2 = np.sum(btemp ** 2)
        suuf = np.sum(np.linalg.norm(btemp, axis=1))
        bcount = btemp.shape[0]

        u2 = np.sum(self.U[self.var]**2)

        f = np.zeros((self.N_c, 3))
        f[self.var] = self.f[self.var]

        ff = np.sum(np.sum(f**2, axis=1)*self.localweight*self.var)

        L = alpha*ff + uuf2

        if self.verbose:
            print("|u-uf|^2 =", uuf2, "\t\tperbead=", suuf/bcount)
            print("|w*f|^2  =", ff, "\t\t|u|^2 =", u2)
            print("L = |u-uf|^2 + lambda*|w*f|^2 = ", L)

        relrec.append((L, uuf2, ff))

        if relrecname is not None:
            np.savetxt(relrecname, relrec)

    def solve_regularized(self, stepper: float =0.33, solver_precision: float =1e-18, i_max: int = 100,
                          rel_conv_crit: float = 0.01, alpha: float = 3e9, method: str = "huber", relrecname: str = None,
                          verbose: bool = False, callback: callable = None):
        """
        Fit the provided displacements. Displacements can be provided with
        :py:meth:`~.Solver.setTargetDisplacements`.

        Parameters
        ----------
        stepper : float, optional
             How much of the displacement of each conjugate gradient step to apply. Default 0.033
        solver_precision : float, optional
            The tolerance for the conjugate gradient step. Will be multiplied by the number of nodes. Default 1e-18.
        i_max : int, optional
            The maximal number of iterations for the regularisation. Default 100
        rel_conv_crit :  float, optional
            If the relative standard deviation of the last 6 energy values is below this threshold, finish the iteration.
            Default 0.01
        alpha :  float, optional
            The regularisation parameter. How much to weight the suppression of forces against the fitting of the measured
            displacement. Default 3e9
        method :  string, optional
            The regularisation method to use:
                "huber"
                "bisquare"
                "cauchy"
                "singlepoint"
        relrecname : string, optional
            The filename where to store the output. Default is to not store the output, just to return it.
        verbose : bool, optional
            If true print status during optimisation
        callback : callable, optional
            A function to call after each iteration (e.g. for a live plot of the convergence)
        """
        self.regularisation_parameters = {
            "stepper": stepper,
            "solver_precision": solver_precision,
            "i_max": i_max,
            "rel_conv_crit": rel_conv_crit,
            "alpha": alpha,
            "method": method,
        }

        # set the verbosity level
        self.verbose = verbose

        self.I = ssp.lil_matrix((self.U_target_mask.shape[0] * 3, self.U_target_mask.shape[0] * 3))
        self.I.setdiag(np.repeat(self.U_target_mask, 3))

        # check if everything is prepared
        self._check_relax_ready()

        self._prepare_temporary_quantities()

        self.localweight = np.ones(self.N_c)

        # update the forces on each tetrahedron and the global stiffness tensor
        if self.verbose:
            print("going to update glo f and K")
        self._updateGloFAndK()

        # log and store values (if a target file was provided)
        relrec = []
        self.relrec = relrec
        if callback is not None:
            callback(self, relrec)
        self._recordRegularizationStatus(relrec, alpha, relrecname)

        if self.verbose:
            print("check before relax !")
        # start the iteration
        for i in range(i_max):
            # compute the weight matrix
            if method != "normal":
                self._updateLocalRegularizationWeigth(method)

            # compute A and b for the linear equation that solves the regularisation problem
            self._computeRegularizationAAndb(alpha)

            # get and apply the displacements that solve the regularisation term
            uu = self._solve_regularization_CG(stepper, solver_precision)

            # update the forces on each tetrahedron and the global stiffness tensor
            self._updateGloFAndK()

            if self.verbose:
                print("Round", i+1, " |du|=", uu)

            # log and store values (if a target file was provided)
            self._recordRegularizationStatus(relrec, alpha, relrecname)

            if callback is not None:
                callback(self, relrec)

            # if we have passed 6 iterations calculate average and std
            if i > 6:
                # calculate the average energy over the last 6 iterations
                last_Ls = np.array(relrec)[:-6:-1, 1]
                Lmean = np.mean(last_Ls)
                Lstd = np.std(last_Ls) / np.sqrt(5)  # the original formula just had /N instead of /sqrt(N)

                # if the iterations converge, stop the iteration
                if Lstd / Lmean < rel_conv_crit:
                    break

        self.regularisation_results = relrec

        return relrec

    def _solve_regularization_CG(self, stepper: float =0.33, solver_precision: float = 1e-18):
        """
        Solve the displacements from the current stiffness tensor using conjugate gradient.
        """

        # solve the conjugate gradient which solves the equation A x = b for x
        # where A is (I - KAK) (K: stiffness matrix, A: weight matrix) and b is (u_meas - u - KAf)
        uu = cg(self.A, self.b.flatten(), maxiter=25*int(pow(self.N_c, 0.33333)+0.5), tol=self.N_c * solver_precision).reshape((self.N_c, 3))

        # add the new displacements to the stored displacements
        self.U += uu * stepper
        # sum the applied displacements
        du = np.sum(uu ** 2) * stepper * stepper

        # return the total applied displacement
        return np.sqrt(du/self.N_c)

    """ helper methods """

    def smoothen(self):
        ddu = 0
        for c in range(self.N_c):
            if self.var[c]:
                A = self.K_glo[c][c]

                f = self.f[c]

                du = np.linalg.inv(A) * f

                self.U[c] += du

                ddu += np.linalg.norm(du)

    def computeStiffening(self, results):

        uu = self.U.copy()

        Ku = self._mulK(uu)

        kWithStiffening = np.sum(uu * Ku)
        k1 = self.CFG["K_0"]

        ds0 = self.CFG["D_0"]

        self.epsilon, self.epsbar, self.epsbarbar = SemiAffineFiberMaterial(k1, ds0, 0, 0, self.CFG)

        self._updateGloFAndK()

        uu = self.U.copy()

        Ku = self._mulK(uu)

        kWithoutStiffening = np.sum(uu, Ku)

        results["STIFFENING"] = kWithStiffening / kWithoutStiffening

        self.computeEpsilon()

    # def contractility(R, f):
    #     B = np.einsum("ni,ni,nj->j", f, f, R) - np.einsum("kj,ki,ki->j", f, R, f)

    #     A = np.sum(np.einsum("ij,kl,kl->kij", np.eye(3), f, f) - np.einsum("ki,kj->kij", f, f), axis=0)

    #     Rcms = np.linalg.inv(A) @ B

    #     RR = R - Rcms
    #     contractility = np.sum(np.einsum("ki,ki->k", RR, f) / np.linalg.norm(RR, axis=1))
    #     return contractility

    # def computeForceMoments(self, rmax):
    #     results = {}

    #     inner = np.linalg.norm(self.R, axis=1) < rmax
    #     f = self.f[inner]
    #     R = self.R[inner]

    #     fsum = np.sum(f, axis=0)

    #     # B1 += self.R[c] * np.sum(f**2)
    #     B1 = np.einsum("ni,ni,nj->j", f, f, R)
    #     # B2 += f * (self.R[c] @ f)
    #     B2 = np.einsum("ki,ki,kj->j", f, R, f)

    #     # A += I * np.sum(f**2) - np.outer(f, f)
    #     A = np.sum(np.einsum("ij,kl,kl->kij", np.eye(3), f, f) - np.einsum("ki,kj->kij", f, f), axis=0)

    #     B = B1 - B2

    #     Rcms = np.linalg.inv(A) @ B

    #     results["FSUM_X"] = fsum[0]
    #     results["FSUM_Y"] = fsum[1]
    #     results["FSUM_Z"] = fsum[2]
    #     results["FSUMABS"] = np.linalg.norm(fsum)

    #     results["CMS_X"] = Rcms[0]
    #     results["CMS_Y"] = Rcms[1]
    #     results["CMS_Z"] = Rcms[2]

    #     RR = R - Rcms
    #     contractility = np.sum(np.einsum("ki,ki->k", RR, f) / np.linalg.norm(RR, axis=1))

    #     results["CONTRACTILITY"] = contractility

    #     vecs = buildBeams(150)

    #     eR = RR / np.linalg.norm(RR, axis=1)[:, None]
    #     f = self.f[inner]

    #     # (eR @ vecs[b]) * (vecs[b] @ self.f_glo[c])
    #     ff = np.sum(np.einsum("ni,bi->nb", eR, vecs) * np.einsum("bi,ni->nb", vecs, f), axis=0)
    #     # (RR @ vecs[b]) * (vecs[b] @ self.f_glo[c])
    #     mm = np.sum(np.einsum("ni,bi->nb", RR, vecs) * np.einsum("bi,ni->nb", vecs, f), axis=0)

    #     bmax = np.argmax(mm)
    #     fmax = ff[bmax]
    #     mmax = mm[bmax]

    #     bmin = np.argmin(mm)
    #     fmin = ff[bmin]
    #     mmin = mm[bmin]

    #     vmid = np.cross(vecs[bmax], vecs[bmin])
    #     vmid = vmid / np.linalg.norm(vmid)

    #     # (eR @ vmid) * (vmid @ self.f_glo[c])
    #     fmid = np.sum(np.einsum("ni,i->n", eR, vmid) * np.einsum("i,ni->n", vmid, f), axis=0)
    #     # (RR @ vmid) * (vmid @ self.f_glo[c])
    #     mmid = np.sum(np.einsum("ni,i->n", RR, vmid) * np.einsum("i,ni->n", vmid, f), axis=0)

    #     results["FMAX"] = fmax
    #     results["MMAX"] = mmax
    #     results["VMAX_X"] = vecs[bmax][0]
    #     results["VMAX_Y"] = vecs[bmax][1]
    #     results["VMAX_Z"] = vecs[bmax][2]

    #     results["FMID"] = fmid
    #     results["MMID"] = mmid
    #     results["VMID_X"] = vmid[0]
    #     results["VMID_Y"] = vmid[1]
    #     results["VMID_Z"] = vmid[2]

    #     results["FMIN"] = fmin
    #     results["MMIN"] = mmin
    #     results["VMIN_X"] = vecs[bmin][0]
    #     results["VMIN_Y"] = vecs[bmin][1]
    #     results["VMIN_Z"] = vecs[bmin][2]

    #     results["E_GLO"] = self.E_glo

    #     results["POLARITY"] = fmax / contractility

    #     return results

    def getPolarity(self):

        inner = self.reg_mask
        f = self.f[inner]
        R = self.R[inner]

        fsum = np.sum(f, axis=0)

        # B1 += self.R[c] * np.sum(f**2)
        B1 = np.einsum("ni,ni,nj->j", f, f, R)
        # B2 += f * (self.R[c] @ f)
        B2 = np.einsum("ki,ki,kj->j", f, R, f)

        # A += I * np.sum(f**2) - np.outer(f, f)
        A = np.sum(np.einsum("ij,kl,kl->kij", np.eye(3), f, f) - np.einsum("ki,kj->kij", f, f), axis=0)

        B = B1 - B2

        try:
            Rcms = np.linalg.inv(A) @ B
        except np.linalg.LinAlgError:
            Rcms = np.array([0, 0, 0])

        RR = R - Rcms
        contractility = np.sum(np.einsum("ki,ki->k", RR, f) / np.linalg.norm(RR, axis=1))

        vecs = buildBeams(150)

        eR = RR / np.linalg.norm(RR, axis=1)[:, None]
        f = self.f[inner]

        # (eR @ vecs[b]) * (vecs[b] @ self.f_glo[c])
        ff = np.sum(np.einsum("ni,bi->nb", eR, vecs) * np.einsum("bi,ni->nb", vecs, f), axis=0)
        # (RR @ vecs[b]) * (vecs[b] @ self.f_glo[c])
        mm = np.sum(np.einsum("ni,bi->nb", RR, vecs) * np.einsum("bi,ni->nb", vecs, f), axis=0)

        bmax = np.argmax(mm)
        fmax = ff[bmax]

        return fmax / contractility

    def storePrincipalStressAndStiffness(self, sbname, sbminname, epkname):
        return # TODO
        sbrec = []
        sbminrec = []
        epkrec = []

        for tt in range(self.N_T):
            if tt % 100 == 0:
                print("computing principa stress and stiffness",
                      (np.floor((tt / (self.N_T + 0.0)) * 1000) + 0.0) / 10.0, "          ", end="\r")

            u_T = np.zeros((3, 4))
            for t in range(4):
                for i in range(3):
                    u_T[i][t] = self.U[self.T[tt][t]][i]

            FF = u_T @ self.Phi[tt]

            F = FF + np.eye(3)

            P = np.zeros((3, 3))
            K = np.zeros((3, 3, 3, 3))

            for b in range(self.N_b):
                s_bar = F @ self.s[b]

                deltal = abs(s_bar) - 1.0

                if deltal > dlbmax:
                    bmax = b
                    dlbmax = deltal

                if deltal < dlbmin:
                    bmin = b
                    dlbmin = deltal

                li = np.round((deltal - self.dlmin) / self.dlstep)

                if li > ((self.dlmax - self.dlmin) / self.dlstep):
                    li = ((self.dlmax - self.dlmin) / self.dlstep) - 1

                self.E += self.epsilon[li] / (self.N_b + 0.0)

                P += np.outer(self.s[b], s_bar) @ self.epsbar[li] * (1.0 / (deltal + 1.0) / (self.N_b + 0.0))

                dEdsbar = -1.0 * (self.epsbar[li] / (deltal + 1.0)) / (self.N_b + 0.0)

                dEdsbarbar = (((deltal + 1.0) * self.epsbarbar[li] - self.epsbar[li]) / (
                        (deltal + 1.0) * (deltal + 1.0) * (deltal + 1.0))) / (self.N_b + 0.0)

                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            for l in range(3):
                                K[i][j][k][l] += dEdsbarbar * self.s[b][i] * self.s[b][k] * s_bar[j] * s_bar[l]
                                if j == l:
                                    K[i][j][k][l] -= dEdsbar * self.s[b][i] * self.s[b][k]

            p = (P * self.s[bmax]) @ self.s[bmax]

            kk = 0

            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for l in range(3):
                            kk += K[i][j][k][l] * self.s[bmax][i] * self.s[bmax][j] * self.s[bmax][k] * self.s[bmax][l]

            sbrec.append(F * self.s[bmax])
            sbminrec.append(F * self.s[bmin])
            epkrec.append(np.array([self.E, p, kk]))

        np.savetxt(sbname, sbrec)
        print(sbname, "stored.")
        np.savetxt(sbminname, sbminrec)
        print(sbminname, "stored.")
        np.savetxt(epkname, epkrec)
        print(epkname, "stored.")

    def storeRAndU(self, Rname: str, Uname: str):
        Rrec = []
        Urec = []

        for c in range(self.N_c):
            Rrec.append(self.R[c])
            Urec.append(self.U[c])

        np.savetxt(Rname, Rrec)
        print(Rname, "stored.")
        np.savetxt(Uname, Urec)
        print(Uname, "stored.")

    def storeF(self, Fname: str):
        Frec = []

        for c in range(self.N_c):
            Frec.append(self.f[c])

        np.savetxt(Fname, Frec)
        print(Fname, "stored.")

    def storeFden(self, Fdenname: str):
        Vr = np.zeros(self.N_c)

        for tt in range(self.N_T):
            for t in range(4):
                Vr[self.T[tt][t]] += self.V[tt] * 0.25

        Frec = []
        for c in range(self.N_c):
            Frec.append(self.f[c] / Vr[c])

        np.savetxt(Fdenname, Frec)
        print(Fdenname, "stored.")

    def storeEandV(self, Rname: str, EVname: str):
        Rrec = []
        EVrec = []

        for t in range(self.N_T):
            N = np.mean(np.array([self.R[self.T[t][i]] for i in range(4)]), axis=0)

            Rrec.append(N)

            EVrec.append([self.E[t], self.V[t]])

        np.savetxt(Rname, Rrec)
        print(Rname, "stored.")

        np.savetxt(EVname, EVrec)
        print(EVname, "stored.")

    def plotMesh(self, use_displacement: bool = True, edge_color: str = None, alpha: float = 0.2):
        import mpl_toolkits.mplot3d as a3
        import matplotlib.pyplot as plt
        from matplotlib import _pylab_helpers

        if _pylab_helpers.Gcf.get_active() is None:
            axes = a3.Axes3D(plt.figure())
        else:
            axes = plt.gca()

        if use_displacement:
            points = self.R+self.U
            if edge_color is None:
                edge_color = "red"
        else:
            points = self.R
            if edge_color is None:
                edge_color = "blue"

        vts = points[self.T, :]
        helper = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        tri = a3.art3d.Line3DCollection(vts[:, helper].reshape(-1, 2, 3))
        tri.set_alpha(alpha)
        tri.set_edgecolor(edge_color)
        axes.add_collection3d(tri)
        axes.plot(points[:, 0], points[:, 1], points[:, 2], 'ko')
        axes.set_aspect('equal')

    def viewMesh(self, f1: float, f2: float):
        from .meshViewer import MeshViewer

        L = getLinesTetrahedra2(self.T)

        return MeshViewer(self.R, L, self.f, self.U, f1, f2)

    def getCenter(self, mode="force", border=None):
        f = self.f
        R = self.R
        U = self.U 
                     
        if self.reg_mask is not None:
            f = self.f * self.reg_mask[:, None]
        
        if mode.lower() == "deformation":
            # B1 += self.R[c] * np.sum(f**2)
            B1 = np.einsum("ni,ni,nj->j", U, U, R)
            # B2 += f * (self.R[c] @ f)
            B2 = np.einsum("ki,ki,kj->j", U, R, U)
            # A += I * np.sum(f**2) - np.outer(f, f)
            A = np.sum(np.einsum("ij,kl,kl->kij", np.eye(3), U, U) - np.einsum("ki,kj->kij", U, U), axis=0)
            B = B1 - B2
            #print (U,R)
            Rcms = np.linalg.inv(A) @ B
        
        if mode.lower() == "deformation_target":      
            # mode to calculate from PIV data directly 
            U = self.U_target     
            # remove nanas from 3D PIV here to be able to do the calculations
            U[np.isnan(U)] = 0 
 
            # B1 += self.R[c] * np.sum(f**2)
            B1 = np.einsum("ni,ni,nj->j", U, U, R)
            # B2 += f * (self.R[c] @ f)
            B2 = np.einsum("ki,ki,kj->j", U, R, U)
            # A += I * np.sum(f**2) - np.outer(f, f)
            A = np.sum(np.einsum("ij,kl,kl->kij", np.eye(3), U, U) - np.einsum("ki,kj->kij", U, U), axis=0)
            B = B1 - B2
            #print (U,R)
            Rcms = np.linalg.inv(A) @ B
        
        if mode.lower() == "force":    # calculate Force center 
            # B1 += self.R[c] * np.sum(f**2)
            B1 = np.einsum("ni,ni,nj->j", f, f, R)
            # B2 += f * (self.R[c] @ f)
            B2 = np.einsum("ki,ki,kj->j", f, R, f)
            # A += I * np.sum(f**2) - np.outer(f, f)
            A = np.sum(np.einsum("ij,kl,kl->kij", np.eye(3), f, f) - np.einsum("ki,kj->kij", f, f), axis=0)
            B = B1 - B2
            try:
                Rcms = np.linalg.inv(A) @ B
            except np.linalg.LinAlgError:
                Rcms = np.array([0, 0, 0])
            
        return Rcms
    
    def getParrallelAndPerpendicularForces(self):
        # Plot perpendicular forces only
        f = self.f
        R = self.R  
        if self.reg_mask is not None:
            f = self.f * self.reg_mask[:, None]                   
        # get center of force field
        Rcms = self.getCenter(mode="force")
        RR = R - Rcms
        RRnorm = RR/ np.linalg.norm(RR, axis=1)[:,None]
        fnorm = f / np.linalg.norm(f, axis=1)[:,None]
        f2 = np.sum(RRnorm * fnorm, axis=1)[:,None] * f
        f3 = np.cross(RRnorm, f)
        return f2, f3
    
    def force_ratio_outer_to_inner(self, width_outer = 5e-6):
        """
        Divides the Cubus in an outer and inner part, where the outer part
        is defined as everything closer than width_outer (default ist 14 um) 
        to the edge of the stack. 
        
        Then computes the ratio of the mean forces in the outer part divided by
        the mean forces within the inner part.
        
        This ratio can be used to distinguish Drift from pulling cells. 
        A pulling cell will have a lower ratio (meaning higher forces in the inner part) 
        whereas drift/rotation often shows high forces at the edges (outer part) and
        therfore has a higher value.  
        """
       
        f = self.f
        R = self.R
        # exclude counter forces
        if self.reg_mask is not None:
            f = self.f * self.reg_mask[:, None]  
        # mask the data points in the outer layer (closer to the outer border than
        # width_outer and for the inner shell the remaining inner volume   
        outer_layer = (((R[:,0])>(R[:,0].max()-width_outer)) | ((R[:,0]) < (R[:,0].min()+width_outer)))\
            | (((R[:,1])>(R[:,1].max()-width_outer)) | ((R[:,1]) < (R[:,1].min()+width_outer)))\
                | (((R[:,2])>(R[:,2].max()-width_outer)) | ((R[:,2]) < (R[:,2].min()+width_outer)))  
    
        # mask the regions
        f_inner = f[~outer_layer]
        f_outer = f[outer_layer]
     
        # compute the mean forces of each layer
        mean_abs_f_inner = np.nanmean(np.linalg.norm(f_inner,axis=1))
        mean_abs_f_outer = np.nanmean(np.linalg.norm(f_outer,axis=1))
        #print (mean_abs_f_inner,mean_abs_f_outer)
        ratio =  mean_abs_f_outer / mean_abs_f_inner
        #print (ratio)
        return ratio
        
    def ratio_deformation_z_component(self):
        """
        Calculate the sum of the absolute deformations for each direction. 
        Returns the fraction of the z-components. 
        Can be usefull to discard stacks with high drift in z-direction, for
        example due to fast stage movement & gel wobbling
        """
        try:
            # sum up the absolute deformations component wise 
            sumx = np.nansum(np.abs(self.U_target[:,0]))
            sumy = np.nansum(np.abs(self.U_target[:,1]))
            sumz = np.nansum(np.abs(self.U_target[:,2]))
            # calculate the fraction of the z-component    
            ratioz = sumz / (sumx+sumy+sumz)
        except:
            ratioz = np.nan
        return ratioz
    
    def getContractility(self, center_mode="force", r_max=None):
        f = self.f
        R = self.R

        if self.reg_mask is not None:
            f = self.f * self.reg_mask[:, None]
                  
        # get center of force field
        Rcms = self.getCenter(mode=center_mode)
        RR = R - Rcms
       
        #if r_max specified only use forces within this distance to cell for contractility
        if r_max:  
            inner = np.linalg.norm(RR, axis=1) < r_max
            f = f[inner]
            RR = RR[inner]  

        #mag = np.linalg.norm(f, axis=1)
        
        RRnorm = RR / np.linalg.norm(RR, axis=1)[:, None]
        contractility = np.nansum(np.einsum("ki,ki->k", RRnorm, f))
        return contractility
    
        
    def getContractilityIgnoreBorder(self, width_outer = 5e-6, center_mode="force" ):   
        """
        Computes the contractilty value while ignoring all force vectors that
        are located within at the borders (outer shell with width_outer).
        For Drift/Rotation we often find high forces here.
        """  
        f = self.f
        R = self.R
        # exclude counter forces
        if self.reg_mask is not None:
            f = self.f * self.reg_mask[:, None]  
        # mask the data points in the outer layer (closer to the outer border than
        # width_outer and for the inner shell the remaining inner volume   
        outer_layer = (((R[:,0])>(R[:,0].max()-width_outer)) | ((R[:,0]) < (R[:,0].min()+width_outer)))\
            | (((R[:,1])>(R[:,1].max()-width_outer)) | ((R[:,1]) < (R[:,1].min()+width_outer)))\
                | (((R[:,2])>(R[:,2].max()-width_outer)) | ((R[:,2]) < (R[:,2].min()+width_outer)))  
        # now ignore the forces in the border region
        f[outer_layer] = np.nan
        # get center of complete force field and set as origin
        Rcms_all = self.getCenter(mode=center_mode)
        RR = R - Rcms_all
        
        RRnorm = RR / np.linalg.norm(RR, axis=1)[:, None]
        contractility = np.nansum(np.einsum("ki,ki->k", RRnorm, f))
    
        return contractility  
    
    def getContractilityUnbiasedEpicenter(self,  r_max=None):   
        """
        Computes a contractilty value, that uses an unbiased center approach.
        Instsead of computing the center from all force vectors, which will bias the 
        the location (as a result of optimization), here we split the image stack into +x,-x halfspaces
        +y,-y halfspaces and +z,-z halfspaces around the intitially found center.
        Then we compute the center of the force vectors within this halfsphere and
        compute the contractility of the force components in the opposing halfspacce
        onto this center (e.g. -x halfspace values uses the center from +x halfsphere).
        We then add up the derived contractility values from -x and +x components (same for -+y and -+z) 
        and  compute the final unbiased contractilty as mean value of all 3 dimension.
         
        returns the unbiased contractility value
        """    
       
        ### first get the initial center of the whole force field withe 
        ## the biased Center method
        f = self.f
        R = self.R
        if self.reg_mask is not None:
            f = self.f * self.reg_mask[:, None]       
        # get center of complete force field and set as origin
        Rcms_all = self.getCenter(mode="Force")
        RR = R - Rcms_all
        #if r_max specified only use forces within this distance to cell for contractility
        if r_max:  
            inner = np.linalg.norm(RR, axis=1) < r_max
            f = f[inner]
            RR = RR[inner]  
        RRnorm = RR / np.linalg.norm(RR, axis=1)[:, None]
        contractility = np.nansum(np.einsum("ki,ki->k", RRnorm, f))
        #print ("Contractility (Biased): "+str(contractility))
        try:       
            # UNBIASED EPICENTER
            # create new solver object for subvolumes and mask only the forces in that area
            subvolume_minus_x,subvolume_plus_x = Solver(),Solver()  
            subvolume_minus_y,subvolume_plus_y =  Solver(),Solver()  
            subvolume_minus_z,subvolume_plus_z =  Solver(),Solver() 
            
            ### now fill all empty solvers with identical data frrom original solver
            subvolumes = [subvolume_minus_x,subvolume_plus_x,
                          subvolume_minus_y,subvolume_plus_y,
                          subvolume_minus_z,subvolume_plus_z]
        
            for sub in subvolumes:
                sub.f = self.f.copy()
                sub.R = self.R.copy()
                sub.reg_mask = self.reg_mask.copy()
            
            ### select only the forces in the certain subvolumes
            subvolume_minus_x.f[RR[:,0]>0] = 0       
            subvolume_plus_x.f[RR[:,0]<0] =  0       
            subvolume_minus_y.f[RR[:,1]>0] = 0        
            subvolume_plus_y.f[RR[:,1]<0] = 0         
            subvolume_minus_z.f[RR[:,2]>0] = 0        
            subvolume_plus_z.f[RR[:,2]<0] = 0         
            
            ### get center of all individual subvolumes
            center_subs = []
            # do same getCenter calculation for each individual subvolume
            for sub in subvolumes:
                if sub.reg_mask is not None:
                    f_sub = sub.f * sub.reg_mask[:, None]         
                # get center of complete force field and set as origin
                center_subs.append(sub.getCenter(mode="Force"))
            
            #### switch the order to compare +x cube with -x center; same for all coordinated
            center_subs_unbiased = [center_subs[1],center_subs[0],
                                    center_subs[3],center_subs[2],
                                    center_subs[5],center_subs[4]]
        
            
            ### now compute contractility for all subvolumes with the unbiased center
            subvalues = []
            # do same Contractility calculation for each subvolume with theopposing center
            for n,sub in enumerate(subvolumes):
                if sub.reg_mask is not None:
                    f_sub = sub.f * sub.reg_mask[:, None]            
                # get center of complete force field and set as origin
                RR_sub = R - center_subs_unbiased[n]
                #if r_max specified only use forces within this distance to cell for contractility
                if r_max:  
                    inner = np.linalg.norm(RR_sub, axis=1) < r_max
                    f_sub = f_sub[inner]
                    RR_sub = RR_sub[inner]  
                # contactility for each subvolume
                RRnorm_sub = RR_sub / np.linalg.norm(RR_sub, axis=1)[:, None]
                contractility = np.nansum(np.einsum("ki,ki->k", RRnorm_sub, f_sub ))   
                subvalues.append(contractility)
            ### now add up both x components, both y components and both z components    
            Contractility_components = [subvalues[0]+subvalues[1],
                                        subvalues[2]+subvalues[3],
                                        subvalues[4]+subvalues[5]]
            # print (Contractility_components)
            # Mean Vaule of all three dimension as final unbiased Contractility
            Contractility_unbiased_epicenter = np.nanmean(Contractility_components)
            #print ("Contractility (Unbiased): "+str(Contractility_unbiased_epicenter))
        ## if not enough data points to compute the center in the subvolumes we
        ## go back to the contractility of the whole stack
        ## think of np.nan here... ?
        except:
            Contractility_unbiased_epicenter = contractility
        
        
        return Contractility_unbiased_epicenter
    
    
    def max_deformation_projection(self): 
        # perform a maximal deformation z-projection to the deformation field
        # input is the solver object, output ist projection in the 
        # form [x,y,dx,dy] for a quiver plot
           
        # go through the z planes in the image and append x,y,dx,dy to a list
        slide = []
        for z in np.unique(self.R[:,2]):
            mask = (self.R[:,2] == z)
            x = (self.R[mask][:,0])
            y = (self.R[mask][:,1]) 
            dx = self.U_target[mask][:,0]
            dy = self.U_target[mask][:,1]
            slide.append ([x,y,dx,dy])
        slide = np.array(slide)
        # maximum deformation z projection of our deformatin fields
        proj_dx, proj_dy = [],[]
        for xycord in range(len(slide[0][0])) :   
            # find at which z index we have the largest deformation
            deformation_z = np.sqrt(slide[:,2,xycord]**2+slide[:,3,xycord]**2)
            index_max = np.where(deformation_z==np.nanmax(deformation_z))[0][0] 
            #append this value to list
            proj_dx.append(slide[:,2,xycord][index_max])
            proj_dy.append(slide[:,3,xycord][index_max])
        # our final projection , just take xy indici from the first slide (since these are equal for all slices anyway..)    
        projection =  [slide[0][0],slide[0][1],proj_dx,proj_dy]
        return projection
    
    
    
    def getContractilityDeformations(self, center_mode="deformation", r_max=None):
           u = self.U_target
           R = self.R

           # get center of force field
           Rcms = self.getCenter(mode=center_mode)
           RR = R - Rcms
          
           #if r_max specified only use forces within this distance to cell for contractility
           if r_max:  
               inner = np.linalg.norm(RR, axis=1) < r_max
               u = u[inner]
               RR = RR[inner]  
 
           RRnorm = RR / np.linalg.norm(RR, axis=1)[:, None]
           # * -1 since deformations are opposed to forces 
           contractility = -1 *  np.nansum(np.einsum("ki,ki->k", RRnorm, u))
           return contractility
    
    
    def getPerpendicularForces(self, center_mode="force", r_max=None):
         f = self.f
         R = self.R

         if self.reg_mask is not None:
             f = self.f * self.reg_mask[:, None]
        
         # get center of force field
         Rcms = self.getCenter(mode=center_mode)
         RR = R - Rcms
       
         #if r_max specified only use forces within this distance to cell for contractility
         if r_max:  
            inner = np.linalg.norm(RR, axis=1) < r_max
            f = f[inner]
            RR = RR[inner]  
        
         RRnorm = RR / np.linalg.norm(RR, axis=1)[:, None]
         anti_contractility = np.nansum(np.linalg.norm(np.cross(RRnorm, f), axis=1))
         return anti_contractility
     
    def getPerpendicularDeformations(self, center_mode="deformation", r_max=None):
         u = self.U_target
         R = self.R

         # get center of force field
         Rcms = self.getCenter(mode=center_mode)
         RR = R - Rcms
       
         #if r_max specified only use forces within this distance to cell for contractility
         if r_max:  
            inner = np.linalg.norm(RR, axis=1) < r_max
            u = u[inner]
            RR = RR[inner]  
        
         RRnorm = RR / np.linalg.norm(RR, axis=1)[:, None]
         anti_contractility = np.nansum(np.linalg.norm(np.cross(RRnorm, u), axis=1))
         return anti_contractility
     
    def getCentricity(self, center_mode = "force", r_max=None):
         # ration between forces towards cell center and perpendicular forces
         Centricity = self.getContractility(center_mode=center_mode, r_max=r_max) / self.getPerpendicularForces(center_mode=center_mode, r_max=r_max) 
         return Centricity
     
    def getCentricityDeformations(self, center_mode = "deformation", r_max=None):
         # ration between forces towards cell center and perpendicular forces
         Centricity = self.getContractilityDeformations(center_mode=center_mode, r_max=r_max) / self.getPerpendicularDeformations(center_mode=center_mode, r_max=r_max) 
         return Centricity

    
    def forces_to_excel(self, output_folder=None, name="results.xlsx", r_max=25e-6, center_mode = "force", width_outer = 5e-6):
        import pandas as pd
        # Evaluate Force statistics and save to excel file in outpoutfolder if given
        # initialize result dictionary
        results = { 'r_max':[], 
                    'center_mode':[], 
                    'Strain_Energy': [],
                    'Force_sum_abs': [], 
                    'Force_sum_abs_rmax': [], 
                    'Contractility': [],
                    'ContractilityIgnoreBorder': [],     
                    'Contractility_unbiased': [],
                    'force_ratio_outer_to_inner': [],
                    'ratio_deformation_z_component': [],                
                    'Contractility_rmax': [],     
                    'Force_perpendicular': [],
                    'Centricity_force': [], 
                    'Centricity_deformations': [],
                    'Centricity_deformations_rmax': [],
                    'Center_x': [], 'Center_y': [], 'Center_z': [],
                    'Median_Deformation': [], 'Maximal_Deformation': [], '99_Percentile_Deformation': [],
                    'Median_Force': [], 'Maximal_Force': [], '99_Percentile_Force': [], 
                    #'RMS_Deformation_per_node': [],'RMS_Deformation': [], 
                    'RMS_Deformation_normed': [],
                    'Contractility_deformations': [], 
                    #'Iterations': [], 
                  }
        
        
        # fill values
        inner = np.linalg.norm(self.R, axis=1) < r_max
        results["r_max"].append(r_max)   
        results["center_mode"].append(center_mode) 
        results["Strain_Energy"].append(self.E_glo)
        results["Force_sum_abs"].append(np.nansum(np.linalg.norm(self.f[self.reg_mask],axis=1)))             
        results["Force_sum_abs_rmax"].append(np.nansum(np.linalg.norm(self.f[self.reg_mask & inner],axis=1)))
        results["Contractility"].append(self.getContractility(center_mode=center_mode))
        results["ContractilityIgnoreBorder"].append(self.getContractilityIgnoreBorder(width_outer = width_outer, center_mode=center_mode))   
        results["Contractility_unbiased"].append(self.getContractilityUnbiasedEpicenter())
        results["force_ratio_outer_to_inner"].append(self.force_ratio_outer_to_inner(width_outer = width_outer))   
        results["ratio_deformation_z_component"].append(self.ratio_deformation_z_component())   
        results["Contractility_rmax"].append(self.getContractility(center_mode=center_mode,r_max=r_max))
        results["Force_perpendicular"].append(self.getPerpendicularForces(center_mode=center_mode))
        results["Centricity_force"].append(self.getCentricity(center_mode=center_mode))  
        results["Centricity_deformations"].append(self.getCentricityDeformations(center_mode=center_mode))
        results["Centricity_deformations_rmax"].append(self.getCentricityDeformations(r_max=r_max,center_mode=center_mode))        
        results["Contractility_deformations"].append(self.getContractilityDeformations(center_mode=center_mode))    
        results["Center_x"].append(self.getCenter(mode=center_mode)[0])
        results["Center_y"].append(self.getCenter(mode=center_mode)[1])
        results["Center_z"].append(self.getCenter(mode=center_mode)[2])
        results["Median_Deformation"].append(np.nanmedian(np.linalg.norm(self.U_target[self.reg_mask],axis=1)))     
        results["Maximal_Deformation"].append(np.nanmax(np.linalg.norm(self.U_target[self.reg_mask],axis=1)))       
        results["99_Percentile_Deformation"].append(np.nanpercentile(np.linalg.norm(self.U_target[self.reg_mask],axis=1),99))    
        results["Median_Force"].append(np.nanmedian(np.linalg.norm(self.f[self.reg_mask],axis=1)))        
        results["Maximal_Force"].append(np.nanmax(np.linalg.norm(self.f[self.reg_mask],axis=1)))         
        results["99_Percentile_Force"].append(np.nanpercentile(np.linalg.norm(self.f[self.reg_mask],axis=1),99))  
        # calculate RMS of deformations ; RMS normed to the amount of nodes; 
        # RMS normed (to the 99 percentile); RMS of only the 99 percentile (normed to the maximal of those)
        # rms = np.sqrt(np.nanmean((self.U[self.reg_mask]-self.U_target[self.reg_mask])**2))  
        # results["RMS_Deformation"].append( rms  ) 
        # results["RMS_Deformation_per_node"].append( rms / self.R.shape[0] )    
        rms_percentage = np.sqrt(np.nanmean((self.U[self.reg_mask] -  self.U_target[self.reg_mask] ) **2)) / \
                  np.nanpercentile(np.linalg.norm(self.U_target[self.reg_mask],axis=1),99)                                                           
        results["RMS_Deformation_normed"].append( rms_percentage )  
        

        # regularization steps
        #number_iterations = len(self.regularisation_results)
        #results["Iterations"].append(number_iterations)

        # save result.xlsx
        if output_folder:
            df = pd.DataFrame.from_dict(results)
            df.to_excel(os.path.join(output_folder,name))  
        return results    



    def load(self, filename: str):
        data = np.load(filename, allow_pickle=True)

        if "R" in data:
            self.setNodes(data["R"])
        if "T" in data:
            self.setTetrahedra(data["T"])
        if "U_fixed" in data:
            print(data["U_fixed"].shape)
            print(data["f_target"].shape)
            self.setBoundaryCondition(data["f_target"], data["f_target"])

        for param in data:
            setattr(self, param, data[param])

        #if self.U_fixed is not None:
        #    self.var = np.any(np.isnan(self.U_fixed), axis=1)
        #if self.U_target_mask is not None:
        #    self.U_target_mask = np.any(~np.isnan(self.U_target_mask), axis=1)

    def vtk(self):
        import pyvista as pv
        point_cloud = pv.PolyData(self.R)
        #point_cloud.point_arrays["f"] = self.f
        #point_cloud.point_arrays["U"] = self.U
        point_cloud["U_target"] = self.U_target
        return point_cloud

    def plot(self, name="U", scale=None, vmin=None, vmax=None, cmap="turbo", export=None, camera_position=None, window_size=None, shape=None, pyvista_theme = "document"):
        #if getattr(self, "point_cloud", None) is None:
        #    self.point_cloud = self.vtk()

        import pyvista as pv
        import matplotlib.pyplot as plt
        if export is not None:
            pv.set_plot_theme(pyvista_theme)
        if isinstance(name, str):
            name = [name]
        if shape is None:
            shape = (1, len(name))
        plotter = pv.Plotter(off_screen=export is not None, shape=shape, window_size=window_size)
        for i, n in enumerate(name):
            plotter.subplot(i // shape[1], i % shape[1])
            # if the scale is not defined use default values
            if scale is None:
                if n == "f" or n == "f_target":
                    s = 3e4
                else:
                    s = 5
            else:
                s = scale

            point_cloud = pv.PolyData(self.R)
            point_cloud.point_arrays[n] = getattr(self, n)
            point_cloud.point_arrays[n+"_mag"] = np.linalg.norm(getattr(self, n), axis=1)
           
                
            # generate the arrows
            arrows = point_cloud.glyph(orient=n, scale=n+"_mag", factor=s)
            print(n, s)
            # select the colormap, if "turbo" should be used but is not defined, use "jet" instead
            if cmap == "turbo":
                try:
                    cmap = plt.get_cmap(cmap)
                except ValueError:
                    cmap = "jet"
            # add the mesh
            plotter.add_mesh(point_cloud, colormap=cmap, scalars=n+"_mag")
            # colorrange if specified
            if vmin is not None and vmax is not None:
                plotter.add_mesh(arrows, colormap=cmap, name="arrows",clim=[vmin, vmax])
                plotter.update_scalar_bar_range([vmin, vmax])
            else:
                plotter.add_mesh(arrows, colormap=cmap, name="arrows")
           

            plotter.show_grid()

        plotter.link_views()
        if camera_position is not None:
            plotter.camera_position = camera_position
        campos = plotter.show(screenshot=export)
        return plotter, campos


class Result(Saveable):
    __save_parameters__ = ['stack', 'time_delta', 'piv_parameter', 'mesh_piv',
                           'interpolate_parameter', 'solve_parameter', 'solver',
                           '___save_name__', '___save_version__']
    ___save_name__ = "Result"
    ___save_version__ = "1.0"
    output: str = None
    state: False

    stack: List[Stack] = None

    piv_parameter: dict = None
    mesh_piv: List[Mesh] = None

    interpolate_parameter: dict = None
    solve_parameter: dict = None
    solver: List[Solver] = None

    def __init__(self, output=None, stack=None, time_delta=None, **kwargs):
        self.output = str(output)

        self.stack = stack
        if stack == None:
            self.stack = []
        
        self.state = False
        self.time_delta = time_delta

        self.mesh_piv = [None] * (len(self.stack) - 1)
        self.solver = [None] * (len(self.mesh_piv))

        super().__init__(**kwargs)

    def save(self):
        Path(self.output).parent.mkdir(exist_ok=True, parents=True)
        super().save(self.output)

    def on_load(self, filename):
        self.output = str(Path(filename))

    def __repr__(self):
        def filename_to_string(filename):
            if isinstance(filename, list):
                return str(Path(common_start(filename) + "{z}" + common_end(filename)))
            return str(Path(filename))
        folders = [filename_to_string(stack.filename) for stack in self.stack]
        base_folder = common_start(folders)
        base_folder = os.sep.join(base_folder.split(os.sep)[:-1])
        indent = "    "
        text = "Result(" + "\n"
        text += indent + "output = " + self.output + "\n"
        text += indent + "stacks = [" + "\n"
        text += indent + indent + "base_folder = " + base_folder + "\n"
        for stack, filename in zip(self.stack, folders):
            text += indent + indent + filename[len(base_folder):] + " " + str(stack.voxel_size) + "\n"
        text += indent + "]" + "\n"
        if self.piv_parameter:
            text += indent + "piv_parameter = " + str(self.piv_parameter) + "\n"
        if self.interpolate_parameter:
            text += indent + "interpolate_parameter = " + str(self.interpolate_parameter) + "\n"
        if self.solve_parameter:
            text += indent + "solve_parameter = " + str(self.solve_parameter) + "\n"
        text += ")" + "\n"
        return text



def save(filename: str, M: Solver):
    M.save(filename)


def load(filename: str) -> Solver:
    M = Solver()
    M.load(filename)
    return M




from pathlib import Path
import os
def getStacks(filename, output_path, voxel_size, time_delta=None, exist_overwrite_callback=None):
    results = []
    if isinstance(filename, (list, tuple)):
        results1, output_base = format_glob(filename[0])
        results2, _ = format_glob(filename[1])

        for (r1, d1), (r2, d2) in zip(results1.groupby("template").max().iterrows(),
                                      results2.groupby("template").max().iterrows()):
            output = Path(output_path) / os.path.relpath(r1, output_base)
            output = output.parent / output.stem
            output = Path(str(output).replace("*", "") + ".npz")

            r1 = r1.format(z="*")
            r2 = r2.format(z="*")

            if output.exists() and exist_overwrite_callback is not None:
                mode = exist_overwrite_callback(output)
                if mode == 0:
                    break
                if mode == "read":
                    print('exists', output)
                    data = Result.load(output)
                    results.append(data)
                    continue

            data = Result(
                output=output,
                stack=[Stack(r1, voxel_size), Stack(r2, voxel_size)],
            )
            data.save()
            results.append(data)
    else:
        results1, output_base = format_glob(filename)
        if time_delta is None:
            raise ValueError("A time series needs a time_delta, None was given.")

        for template, d in results1.groupby("template"):
            output = Path(output_path) / os.path.relpath(d.iloc[0].template, output_base)
            output = output.parent / output.stem
            output = Path(str(output).replace("*", "") + ".npz")

            if output.exists() and exist_overwrite_callback is not None:
                mode = exist_overwrite_callback(output)
                if mode == 0:
                    break
                if mode == "read":
                    print('exists', output)
                    data = Result.load(output)
                    results.append(data)
                    continue

            stacks = []
            for t, d0 in d.groupby("t"):
                d0 = d0.sort_values(by='z', key=natsort.natsort_keygen())
                stacks.append(Stack(d0.filename, voxel_size))

            data = Result(
                output=output,
                stack=stacks,
                time_delta=time_delta,
            )
            data.save()
            results.append(data)
    return results

def common_start(values):
    if len(values) != 0:
        start = values[0]
        while start:
            if all(value.startswith(start) for value in values):
                return start
            start = start[:-1]
    return ""

def common_end(values):
    if len(values) != 0:
        end = values[0]
        while end:
            if all(value.endswith(end) for value in values):
                return end
            end = end[1:]
    return ""


def substract_reference_state(mesh_piv, mode):
    U = [M.getNodeVar("U_measured") for M in mesh_piv]
    # correct for the median position
    if len(U) > 2:
        xpos2 = np.cumsum(U, axis=0)  # mittlere position
        if mode == "first":
            xpos2 -= xpos2[0]
        elif mode == "median":
            xpos2 -= np.nanmedian(xpos2, axis=0)  # aktuelle abweichung von
        elif mode == "last":
            xpos2 -= xpos2[-1]
    else:
        xpos2 = U
    return xpos2


def interpolate_mesh(M, xpos2, params):
    import saenopy
    from saenopy.multigridHelper import getScaledMesh, getNodesWithOneFace
    points, cells = saenopy.multigridHelper.getScaledMesh(params["element_size"] * 1e-6,
                                                          params["inner_region"] * 1e-6,
                                                          np.array([params["mesh_size_x"], params["mesh_size_y"],
                                                                    params["mesh_size_z"]]) * 1e-6 / 2,
                                                          [0, 0, 0], params["thinning_factor"])

    R = (M.R - np.min(M.R, axis=0)) - (np.max(M.R, axis=0) - np.min(M.R, axis=0)) / 2
    U_target = saenopy.getDeformations.interpolate_different_mesh(R, xpos2, points)

    border_idx = getNodesWithOneFace(cells)
    inside_mask = np.ones(points.shape[0], dtype=bool)
    inside_mask[border_idx] = False

    M = saenopy.Solver()
    M.setNodes(points)
    M.setTetrahedra(cells)
    M.setTargetDisplacements(U_target, inside_mask)

    return M
