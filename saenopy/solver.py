import os
import sys
import time

import numpy as np
import scipy.sparse as ssp

from numba import njit
from typing import Union
from nptyping import NDArray, Shape, Float, Int, Bool

from saenopy.multigridHelper import getLinesTetrahedra2
from saenopy.buildBeams import buildBeams
from saenopy.materials import Material, SemiAffineFiberMaterial
from saenopy.conjugateGradient import cg
from saenopy.saveable import Saveable
from typing import List


class Mesh(Saveable):
    __save_parameters__ = ['R', 'T', 'node_vars']
    R: NDArray[Shape["N_c, 3"], Float] = None  # the 3D positions of the vertices, dimension: N_c x 3
    T: NDArray[Shape["N_t, 4"], Int] = None  # the tetrahedra' 4 corner vertices (defined by index), dimensions: N_T x 4

    node_vars: dict = None

    def __init__(self,
                 R: NDArray[Shape["N_c, 3"], Float] = None,
                 T: NDArray[Shape["N_t, 4"], Int] = None,
                 node_vars: dict = None):
        if R is not None:
            self.setNodes(R)
        if T is not None:
            self.setTetrahedra(T)
        self.node_vars = {}
        if node_vars is not None:
            for name, data in node_vars.items():
                self.setNodeVar(name, data)

    def setNodes(self, data: NDArray[Shape["N_c, 3"], Float]):
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

    def setTetrahedra(self, data: NDArray[Shape["N_t, 4"], Int]):
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
        self.T = data.astype(int)

    def setNodeVar(self, name: str, data: Union[NDArray[Shape["N_c"], Float], NDArray[Shape["N_c, 3"], Float]]):
        data = np.asarray(data)
        assert len(data.shape) == 1 or len(data.shape) == 2, "Node var needs to be scalar or vectorial"
        if len(data.shape) == 2:
            assert data.shape[1] == 3, "Vectrial node var needs to have 3 dimensionts"
        assert data.shape[0] == self.R.shape[0], "Node var has to have the same number of nodes than the mesh"
        self.node_vars[name] = data

    def getNodeVar(self, name: str) -> Union[NDArray[Shape["N_c"], Float], NDArray[Shape["N_c, 3"], Float]]:
        return self.node_vars[name]

    def hasNodeVar(self, name: str):
        return name in self.node_vars


class Solver(Saveable):
    __save_parameters__ = ["R", "T", "U", "f", "U_fixed", "U_target", "U_target_mask", "f_target", "E_glo", "var", "regularisation_results",
                      "reg_mask", "regularisation_parameters", "material_model"]

    R: NDArray[Shape["N_c, 3"], Float] = None  # the 3D positions of the vertices, dimension: N_c x 3
    T: NDArray[Shape["N_t, 4"], Int] = None  # the tetrahedra' 4 corner vertices (defined by index), dimensions: N_T x 4
    E: NDArray[Shape["N_t"], Float] = None  # the energy stored in each tetrahedron, dimensions: N_T
    V: NDArray[Shape["N_t"], Float] = None  # the volume of each tetrahedron, dimensions: N_T
    var: NDArray[Shape["N_c"], Bool] = None  # a bool if a node is movable

    Phi: NDArray[Shape["N_t, 4, 3"], Float] = None  # the shape tensor of each tetrahedron, dimensions: N_T x 4 x 3
    Phi_valid = False
    U: NDArray[Shape["N_c, 3"], Float] = None  # the displacements of each node, dimensions: N_c x 3
    U_fixed: NDArray[Shape["N_c, 3"], Float] = None
    U_target: NDArray[Shape["N_c, 3"], Float] = None
    U_target_mask: NDArray[Shape["N_c"], Bool] = None
    reg_mask: NDArray[Shape["N_c"], Bool] = None

    f: NDArray[Shape["N_c, 3"], Float] = None  # the global forces on each node, dimensions: N_c x 3
    f_target: NDArray[Shape["N_c, 3"], Float] = None  # the external forces on each node, dimensions: N_c x 3
    K_glo: NDArray[Shape["N_c, N_c, 3, 3"], Float] = None  # the global stiffness tensor, dimensions: N_c x N_c x 3 x 3

    Laplace = None

    E_glo: float = 0  # the global energy

    # a list of all vertices are connected via a tetrahedron, stored as pairs: dimensions: N_connections x 2
    connections: NDArray[Shape["N_con, 2"], Int] = None
    connections_valid = False

    N_T = 0  # the number of tetrahedra
    N_c = 0  # the number of vertices

    s: NDArray[Shape["N_b, 3"], Float] = None  # the beams, dimensions N_b x 3
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

    def setNodes(self, data: NDArray[Shape["N_c, 3"], Float]):
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

        self.var = np.ones(self.N_c, dtype=bool)
        self.U = np.zeros((self.N_c, 3))
        self.f = np.zeros((self.N_c, 3))
        self.f_target = np.zeros((self.N_c, 3))

    def setBoundaryCondition(self, displacements: NDArray[Shape["N_c, 3"], Float] = None,
                             forces: NDArray[Shape["N_c, 3"], Float] = None):
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
            # if not, check if the fixed displacements have no force
            elif np.all(np.isnan(self.f_target[~self.var])) is False:
                print("WARNING: Forces for fixed vertices were specified. These boundary conditions cannot be"
                      "fulfilled", file=sys.stderr)

    def setInitialDisplacements(self, displacements: NDArray[Shape["N_c, 3"], Float]):
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

    def _setExternalForces(self, forces: NDArray[Shape["N_c, 3"], Float]):
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

    def setTetrahedra(self, data: NDArray[Shape["N_t, 4"], Int]):
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
        self.T = data.astype(int)

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

    def setBeams(self, beams: Union[int, NDArray[Shape["N_b, 3"], Float]] = 300):
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
        # current scipy versions do not have the sputils anymore
        try:
            from scipy.sparse._sputils import get_index_dtype
        except ImportError:
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

    def getMaxTetStiffness(self) -> NDArray[Shape["N_t"], Float]:
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

    def _get_s_bar(self, t: slice) -> NDArray[Shape["N_t, 3, N_b"], Float]:
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

    def solve_boundarycondition(self, stepper: float = 0.066, i_max: int = 300, i_min: int = 12, rel_conv_crit: float = 0.01, relrecname: str = None, verbose: bool = False, callback: callable = None):
        """
        Solve the displacement of the free nodes constraint to the boundary conditions.

        Parameters
        ----------
        stepper : float, optional
            How much of the displacement of each conjugate gradient step to apply. Default 0.066
        i_max : int, optional
            The maximal number of iterations for the relaxation. Default 300
        i_min : int, optional
            The minimal number of iterations for the relaxation. Minimum value is 6. Default is 12
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
                
            # if we have passed i_min iterations we calculate average and std of the last 6 iteration
            if i > i_min:
                # calculate the average energy over the last 6 iterations
                last_Es = np.array(relrec)[:-6:-1, 1]
                Emean = np.mean(last_Es)
                Estd = np.std(last_Es)

                # if the iterations converge, stop the iteration
                # coefficient of variation below "rel_conv_crit"
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

    def setTargetDisplacements(self, displacement: NDArray[Shape["N_c, 3"], Float], reg_mask: NDArray[Shape["N_c"], Bool] = None):
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

    def solve_regularized(self, stepper: float = 0.33, solver_precision: float = 1e-18, i_max: int = 300,
                          i_min: int = 12, rel_conv_crit: float = 0.01, alpha: float = 1e10, method: str = "huber",
                          relrecname: str = None, verbose: bool = False, callback: callable = None):
        """
        Fit the provided displacements. Displacements can be provided with
        :py:meth:`~.Solver.setTargetDisplacements`.

        Parameters
        ----------
        stepper : float, optional
             How much of the displacement of each conjugate gradient step to apply. Default 0.33
        solver_precision : float, optional
            The tolerance for the conjugate gradient step. Will be multiplied by the number of nodes. Default 1e-18.
        i_max : int, optional
            The maximal number of iterations for the regularisation. Default 300
        i_min : int, optional
            The minimal number of iterations for the relaxation. Minimum value is 6. Default is 12.
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
            callback(self, relrec, 0, i_max)
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
                callback(self, relrec, i, i_max)

            # if we have passed i_min iterations we calculate average and std of the last 6 iteration
            if i > i_min:
                # calculate the average energy over the last 6 iterations
                last_Ls = np.array(relrec)[:-6:-1, 1]
                Lmean = np.mean(last_Ls)
                Lstd = np.std(last_Ls)       #  Use Coefficient of Variation; in saeno there was the additional factor  "/ np.sqrt(5)" behind 

                # if the iterations converge, stop the iteration
                if Lstd / Lmean < rel_conv_crit:
                    break

        self.regularisation_results = relrec
        return relrec

    def _solve_regularization_CG(self, stepper: float = 0.33, solver_precision: float = 1e-18):
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

    def getPolarity(self) -> float:

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

    def getCenter(self, mode="force", border=None) -> NDArray[Shape["3"], Float]:
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

    def getContractility(self, center_mode="force", r_max=None, width_outer=None) -> str:
        f = self.f
        R = self.R

        if self.reg_mask is not None:
            f = self.f * self.reg_mask[:, None]

        if width_outer is not None:
            # mask the data points in the outer layer (closer to the outer border than
            # width_outer and for the inner shell the remaining inner volume
            outer_layer = (((R[:, 0]) > (R[:, 0].max() - width_outer)) | ((R[:, 0]) < (R[:, 0].min() + width_outer))) \
                          | (((R[:, 1]) > (R[:, 1].max() - width_outer)) | ((R[:, 1]) < (R[:, 1].min() + width_outer))) \
                          | (((R[:, 2]) > (R[:, 2].max() - width_outer)) | ((R[:, 2]) < (R[:, 2].min() + width_outer)))
            # now ignore the forces in the border region
            f[outer_layer] = np.nan

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

    def plot(self, name="U", scale=None, vmin=None, vmax=None, cmap="turbo", export=None, camera_position=None, window_size=None, shape=None, pyvista_theme="document"):
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
            # color range if specified
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

# needs to stay here instead of top to prevent circular import
from saenopy.result_file import Result


def load_solver(filename: str) -> Solver:
    return Solver.load(filename)


def load(filename: str) -> Result:
    return Result.load(filename)


def load_results(filename: str) -> List[Result]:
    import glob
    return [Result.load(file) for file in glob.glob(filename, recursive=True)]


def subtract_reference_state(mesh_piv, mode):
    U = [M.getNodeVar("U_measured") for M in mesh_piv]
    # correct for the median position
    if len(U) > 2:
        xpos2 = np.cumsum(U, axis=0)
        if mode == "first":
            xpos2 -= xpos2[0]
        elif mode == "median":
            xpos2 -= np.nanmedian(xpos2, axis=0)
        elif mode == "last":
            xpos2 -= xpos2[-1]
        elif mode == "next":
            xpos2 = U
    else:
        xpos2 = U
    return xpos2


def interpolate_mesh(M: Solver, xpos2: np.ndarray, params: dict) -> Solver:
    import saenopy
    from saenopy.multigridHelper import getScaledMesh, getNodesWithOneFace

    x, y, z = (M.R.max(axis=0) - M.R.min(axis=0)) * 1e6
    if params["mesh_size_same"]:
        params["mesh_size_x"] = x
        params["mesh_size_y"] = y
        params["mesh_size_z"] = z
    if params["mesh_size_x"] < params["element_size"]*2 or \
       params["mesh_size_y"] < params["element_size"]*2 or \
       params["mesh_size_z"] < params["element_size"]*2:
        raise ValueError("Mesh size needs to be at least twice the element size.")

    # no thinning if no values are set
    if "inner_region" not in params:
        params["inner_region"] = x
    if "thinning_factor" not in params:    
        params["thinning_factor"] = 0
    
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
