import os
import sys
import time

import numpy as np
import scipy.sparse as ssp

from numba import jit, njit

from .multigridHelper import getLinesTetrahedra
from .buildBeams import buildBeams
from .materials import SemiAffineFiberMaterial
from .conjugateGradient import cg


class FiniteBodyForces:
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

    material_model = None  # the function specifying the material model

    def setNodes(self, data):
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

    def setBoundaryCondition(self, displacements=None, forces=None):
        """
        Provide the boundary condition for the mesh, to be used with :py:meth:`~.FiniteBodyForces.relax`.

        Parameters
        ----------
        displacements : ndarray, optional
            If the displacement of a node is not nan, it is treated as a Dirichlet boundary condition and the
            displacement of this node is kept fix during solving. Dimensions Nx3
        forces : ndarray, optional
            If the force of a node is not nan, it is treated as a von Neumann boundary condition and the solver tries to
            match the force on the node with the here given force. In contrast to the displacement conditions the force
            boundary conditions cannot be strictly enforced. Dimensions Nx3
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
            self.setExternalForces(forces)
            # if no displacements where given, take the variable nodes from the nans in the force list
            if displacements is None:
                self.var = ~np.any(np.isnan(forces), axis=1)
            # if not, check if the the fixed displacements have no force
            elif np.all(np.isnan(self.f_target[~self.var])) is False:
                print("WARNING: Forces for non-variable vertices were specified. These boundary conditions cannot be"
                      "fulfilled", file=sys.stderr)

    def setDisplacements(self, displacements):
        """
        Provide initial displacements of the vertices. For non-variable vertices these displacements stay during the
        relaxation. The displacements can also be set with :py:meth:`~.FiniteBodyForces.setNodes` directly with
        the vertices.

        Parameters
        ----------
        displacements : ndarray
            The list of displacements. Dimensions Nx3
        """
        # check the input
        displacements = np.asarray(displacements)
        assert displacements.shape == (self.N_c, 3)
        self.U = displacements.astype(np.float64)

    def setVariable(self, var):
        """
        Specifies whether the vertices can be moved or are fixed. The variable state can also be set with
        :py:meth:`~.FiniteBodyForces.setNodes` directly with the vertices.

        Parameters
        ----------
        var : ndarray
            A list of boolean values which states whether the node can be moved. Dimensions N
        """

        # check the input
        var = np.asarray(var)
        assert var.shape == (self.N_c, )
        self.var = var.astype(bool)
        # schedule to recalculate the connections
        self.connections_valid = False

    def setExternalForces(self, forces):
        """
        Provide external forces that act on the vertices. The forces can also be set with
        :py:meth:`~.FiniteBodyForces.setNodes` directly with the vertices.

        Parameters
        ----------
        forces : ndarray
            The list of forces. Dimensions Nx3
        """
        # check the input
        forces = np.asarray(forces)
        assert forces.shape == (self.N_c, 3)
        self.f_target = forces.astype(np.float64)

    def setTetrahedra(self, data):
        """
        Provide mesh tetrahedra. Each tetrahedron consts of the indices of the 4 vertices which it connects.

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

    def setMaterialModel(self, material):
        """
        Provides the material model for the mesh.

        Parameters
        ----------
        material : :py:class:`~.materials.Material`
             The material, must be of a subclass of Material which implements the method :py:func:`generate_look_up_table`
        """
        self.material_model = material
        self.material_model_look_up = self.material_model.generate_look_up_table()

    def setBeams(self, beams=300):
        """
        Sets the beams for the calculation over the whole body angle.

        Parameters
        ----------
        beams : int, ndarray
            Either an integer which defines in how many beams to discretize the whole body angle or an ndarray providing
            the beams, dimensions Nx3, default 300
        """
        if isinstance(beams, int):
            beams = buildBeams(beams)
        self.s = beams
        self.N_b = beams.shape[0]

    def _computeConnections(self):
        # calculate the indices for "update_f_glo"
        y, x = np.meshgrid(np.arange(3), self.T.ravel())
        self.force_distribute_coordinates = (x.ravel(), y.ravel())

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
        batchsize = 10000

        self.E_glo = 0
        f_glo = np.zeros((self.N_T, 4, 3))
        K_glo = np.zeros((self.N_T, 4, 4, 3, 3))

        for i in range(int(np.ceil(self.T.shape[0]/batchsize))):
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
        print("updating forces and stiffness matrix finished %.2fs" % (time.time() - t_start))

    def _get_s_bar(self, t):
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
    @jit(nopython=True, cache=True)
    def _get_applied_epsilon(s_bar, lookUpEpsilon, _V_over_Nb):
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

    def _update_energy(self, epsilon_b, t):
        # sum the energy of this tetrahedron
        # E_t = eps_tb * V_t
        self.E[t] = np.mean(epsilon_b, axis=1) * self.V[t]

        # only count the energy of the tetrahedron to the global energy if the tetrahedron has at least one
        # variable node
        self.E_glo += np.sum(self.E[t][self._countEnergy[t]])

    def _update_f_glo(self, s_star, s_bar, dEdsbar, out):
        # f_tmi = s*_tmb * s'_tib * dEds'_tb  (t in [0, N_T], i in {x,y,z}, m in {1,2,3,4}, b in [0, N_b])
        np.einsum("tmb,tib,tb->tmi", s_star, s_bar, dEdsbar, out=out)

    def _update_K_glo(self, s_star, s_bar, dEdsbar, dEdsbarbar, out):
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
        if self.N_c == 0:
            raise ValueError("No nodes have yet been set. Call setNodes first.")

        # check if we have tetrahedra
        if self.N_T == 0:
            raise ValueError("No tetrahedra have yet been set. Call setTetrahedra first.")

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

    def relax(self, stepper=0.066, i_max=300, rel_conv_crit=0.01, relrecname=None):
        """
        Calculate the displacement of the nodes for the given external forces.

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
        """

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
            print("Newton ", i, ": du=", du, "  Energy=", self.E_glo, "  Residuum=", ff)

            # log and store values (if a target file was provided)
            relrec.append([du, self.E_glo, ff])
            if relrecname is not None:
                np.savetxt(relrecname, relrec)

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
        print("| time for relaxation was", finish - start)

    def _solve_CG(self, stepper):
        """
        Solve the displacements from the current stiffness tensor using conjugate gradient.
        """
        # calculate the difference between the current forces on the nodes and the desired forces
        ff = self.f - self.f_target

        # ignore the force deviations on fixed nodes
        ff[~self.var, :] = 0

        # solve the conjugate gradient which solves the equation A x = b for x
        # where A is the stiffness matrix K_glo and b is the vector of the target forces
        uu = cg(self.K_glo, ff.ravel(), maxiter=3 * self.N_c, tol=0.00001).reshape(ff.shape)

        # add the new displacements to the stored displacements
        self.U[self.var] += uu[self.var] * stepper
        # sum the applied displacements
        du = np.sum(uu[self.var] ** 2) * stepper * stepper

        # return the total applied displacement
        return du

    """ regularization """

    def setTargetDisplacements(self, displacement):
        """
        Provide the displacements that should be fitted by the regularization.

        Parameters
        ----------
        displacement : ndarray
            If the displacement of a node is not nan, it is
            The displacements for each node. Dimensions N_n x 3
        """
        displacement = np.asarray(displacement)
        assert displacement.shape == (self.N_c, 3)
        self.U_target = displacement
        # only use displacements that are not nan
        self.U_target_mask = np.any(~np.isnan(displacement), axis=1)

    def _updateLocalRegularizationWeigth(self, method):

        self.localweight[:] = 1

        Fvalues = np.linalg.norm(self.f, axis=1)
        Fmedian = np.median(Fvalues[self.var])

        if method == "singlepoint":
            self.localweight[int(self.CFG["REG_FORCEPOINT"])] = 1.0e-10

        if method == "bisquare":
            k = 4.685

            index = Fvalues < k * Fmedian
            self.localweight[index * self.var] *= (1 - (Fvalues / k / Fmedian) * (Fvalues / k / Fmedian)) * (
                    1 - (Fvalues / k / Fmedian) * (Fvalues / k / Fmedian))
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

        index = self.localweight < 1e-10
        self.localweight[index & self.var] = 1e-10

        counter = np.sum(1.0 - self.localweight[self.var])
        counterall = np.sum(self.var)

        print("total weight: ", counter, "/", counterall)

    def _computeRegularizationAAndb(self, alpha):
        KA = self.K_glo.multiply(np.repeat(self.localweight * alpha, 3)[None, :])
        self.KAK = KA @ self.K_glo
        self.A = self.I + self.KAK

        self.b = (KA @ self.f.ravel()).reshape(self.f.shape)

        index = self.var & self.U_target_mask
        self.b[index] += self.U_target[index] - self.U[index]

    def _recordRegularizationStatus(self, relrec, alpha, relrecname=None):
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

        print("|u-uf|^2 =", uuf2, "\t\tperbead=", suuf/bcount)
        print("|w*f|^2  =", ff, "\t\t|u|^2 =", u2)
        print("L = |u-uf|^2 + lambda*|w*f|^2 = ", L)

        relrec.append((L, uuf2, ff))

        if relrecname is not None:
            np.savetxt(relrecname, relrec)

    def regularize(self, stepper=0.33, solver_precision=1e-18, i_max=100, rel_conv_crit=0.01, alpha=3e9, method="huber", relrecname=None):
        """
        Fit the provided displacements. Displacements can be provided with
        :py:meth:`~.FiniteBodyForces.setFoundDisplacements`.

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
            The file where to store the output. Default is to not store the output, just to return it.
        """
        self.I = ssp.lil_matrix((self.U_target_mask.shape[0] * 3, self.U_target_mask.shape[0] * 3))
        self.I.setdiag(np.repeat(self.U_target_mask, 3))

        # check if everything is prepared
        self._check_relax_ready()

        self._prepare_temporary_quantities()

        self.localweight = np.ones(self.N_c)

        # update the forces on each tetrahedron and the global stiffness tensor
        print("going to update glo f and K")
        self._updateGloFAndK()

        # log and store values (if a target file was provided)
        relrec = []
        self._recordRegularizationStatus(relrec, alpha, relrecname)

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

            print("Round", i+1, " |du|=", uu)

            # log and store values (if a target file was provided)
            self._recordRegularizationStatus(relrec, alpha, relrecname)

            # if we have passed 6 iterations calculate average and std
            if i > 6:
                # calculate the average energy over the last 6 iterations
                last_Ls = np.array(relrec)[:-6:-1, 1]
                Lmean = np.mean(last_Ls)
                Lstd = np.std(last_Ls) / np.sqrt(5)  # the original formula just had /N instead of /sqrt(N)

                # if the iterations converge, stop the iteration
                if Lstd / Lmean < rel_conv_crit:
                    break

        return relrec

    def _solve_regularization_CG(self, stepper=0.33, solver_precision=1e-18):
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

    def computeForceMoments(self, rmax):
        results = {}

        inner = np.linalg.norm(self.R, axis=1) < rmax
        f = self.f[inner]
        R = self.R[inner]

        fsum = np.sum(f, axis=0)

        # B1 += self.R[c] * np.sum(f**2)
        B1 = np.einsum("kj,ki->j", R, f**2)
        # B2 += f * (self.R[c] @ f)
        B2 = np.einsum("kj,ki,ki->j", f, R, f)

        # A += I * np.sum(f**2) - np.outer(f, f)
        A = np.sum(np.einsum("ij,kl,kl->kij", np.eye(3), f, f) - np.einsum("ki,kj->kij", f, f), axis=0)

        B = B1 - B2

        Rcms = np.linalg.inv(A) @ B

        results["FSUM_X"] = fsum[0]
        results["FSUM_Y"] = fsum[1]
        results["FSUM_Z"] = fsum[2]
        results["FSUMABS"] = np.linalg.norm(fsum)

        results["CMS_X"] = Rcms[0]
        results["CMS_Y"] = Rcms[1]
        results["CMS_Z"] = Rcms[2]

        RR = R - Rcms
        contractility = np.sum(np.einsum("ki,ki->k", RR, f) / np.linalg.norm(RR, axis=1))

        results["CONTRACTILITY"] = contractility

        vecs = buildBeams(150)

        eR = RR / np.linalg.norm(RR, axis=1)[:, None]
        f = self.f[inner]

        # (eR @ vecs[b]) * (vecs[b] @ self.f_glo[c])
        ff = np.sum(np.einsum("ni,bi->nb", eR, vecs) * np.einsum("bi,ni->nb", vecs, f), axis=0)
        # (RR @ vecs[b]) * (vecs[b] @ self.f_glo[c])
        mm = np.sum(np.einsum("ni,bi->nb", RR, vecs) * np.einsum("bi,ni->nb", vecs, f), axis=0)

        bmax = np.argmax(mm)
        fmax = ff[bmax]
        mmax = mm[bmax]

        bmin = np.argmin(mm)
        fmin = ff[bmin]
        mmin = mm[bmin]

        vmid = np.cross(vecs[bmax], vecs[bmin])
        vmid = vmid / np.linalg.norm(vmid)

        # (eR @ vmid) * (vmid @ self.f_glo[c])
        fmid = np.sum(np.einsum("ni,i->n", eR, vmid) * np.einsum("i,ni->n", vmid, f), axis=0)
        # (RR @ vmid) * (vmid @ self.f_glo[c])
        mmid = np.sum(np.einsum("ni,i->n", RR, vmid) * np.einsum("i,ni->n", vmid, f), axis=0)

        results["FMAX"] = fmax
        results["MMAX"] = mmax
        results["VMAX_X"] = vecs[bmax][0]
        results["VMAX_Y"] = vecs[bmax][1]
        results["VMAX_Z"] = vecs[bmax][2]

        results["FMID"] = fmid
        results["MMID"] = mmid
        results["VMID_X"] = vmid[0]
        results["VMID_Y"] = vmid[1]
        results["VMID_Z"] = vmid[2]

        results["FMIN"] = fmin
        results["MMIN"] = mmin
        results["VMIN_X"] = vecs[bmin][0]
        results["VMIN_Y"] = vecs[bmin][1]
        results["VMIN_Z"] = vecs[bmin][2]

        results["POLARITY"] = fmax / contractility

        return results

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

    def storeRAndU(self, Rname, Uname):
        Rrec = []
        Urec = []

        for c in range(self.N_c):
            Rrec.append(self.R[c])
            Urec.append(self.U[c])

        np.savetxt(Rname, Rrec)
        print(Rname, "stored.")
        np.savetxt(Uname, Urec)
        print(Uname, "stored.")

    def storeF(self, Fname):
        Frec = []

        for c in range(self.N_c):
            Frec.append(self.f[c])

        np.savetxt(Fname, Frec)
        print(Fname, "stored.")

    def storeFden(self, Fdenname):
        Vr = np.zeros(self.N_c)

        for tt in range(self.N_T):
            for t in range(4):
                Vr[self.T[tt][t]] += self.V[tt] * 0.25

        Frec = []
        for c in range(self.N_c):
            Frec.append(self.f[c] / Vr[c])

        np.savetxt(Fdenname, Frec)
        print(Fdenname, "stored.")

    def storeEandV(self, Rname, EVname):
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

    def plotMesh(self, use_displacement=True, edge_color=None, alpha=0.2):
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

    def viewMesh(self, f1, f2):
        from .meshViewer import MeshViewer

        L, Li = getLinesTetrahedra(self.T)

        return MeshViewer(self.R, L, self.f, self.U, f1, f2)

    def save(self, filename):
        parameters = ["R", "T", "U", "f", "U_fixed", "U_target", "f_target"]
        data = {}
        for param in parameters:
            data[param] = getattr(self, param)
        data["type"] = self.__class__.__name__

        np.savez(filename, **data)

    def load(self, filename):
        data = np.load(filename, allow_pickle=True)

        if "R" in data:
            self.setNodes(data["R"])
        if "T" in data:
            self.setTetrahedra(data["T"])
        if "U_fixed" in data:
            self.setBoundaryCondition(data["U_fixed"], data["f_target"])

        for param in data:
            setattr(self, param, data[param])

        #if self.U_fixed is not None:
        #    self.var = np.any(np.isnan(self.U_fixed), axis=1)
        #if self.U_target_mask is not None:
        #    self.U_target_mask = np.any(~np.isnan(self.U_target_mask), axis=1)


def save(filename, M):
    M.save(filename)


def load(filename):
    M = FiniteBodyForces()
    M.load(filename)
    return M
