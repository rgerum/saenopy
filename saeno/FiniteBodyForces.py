import os
import time

import numpy as np
import scipy.sparse as ssp

from numba import jit, njit

from .buildBeams import buildBeams
from .buildEpsilon import buildEpsilon
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

    f_glo = None  # the global forces on each node, dimensions: N_c x 3
    f_ext = None  # the external forces on each node, dimensions: N_c x 3
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

    def setNodes(self, data, var=None, displacements=None, forces=None):
        """
        Provide mesh coordinates, optional with a flag if they can be moved, a displacement and a force. Displacements,
        variable state, and forces can also be set afterwards with :py:meth:`~.FiniteBodyForces.setDisplacements`,
        :py:meth:`~.FiniteBodyForces.setVariable`, and :py:meth:`~.FiniteBodyForces.setExternalForces`.

        Parameters
        ----------
        data : ndarray
            The coordinates of the vertices. Dimensions Nx3
        var : ndarray, optional
            A boolean value whether the node is allowed to move. By default all vertices can be moved. Dimensions N
        displacements : ndarray, optional
            The initial displacement of the node. Dimensions Nx3
        forces : ndarray, optional
            The forces on the node. Dimensions Nx3
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

        # initialize 0 displacement for each node
        if displacements is None:
            self.U = np.zeros((self.N_c, 3))
        else:
            self.setDisplacements(displacements)

        # start with every node being variable (non-fixed)
        if var is None:
            self.var = np.ones(self.N_c, dtype=np.bool)
        else:
            self.setVariable(var)

        # initialize global and external forces
        self.f_glo = np.zeros((self.N_c, 3))
        if forces is None:
            self.f_ext = np.zeros((self.N_c, 3))
        else:
            self.setExternalForces(forces)

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
        self.f_ext = forces.astype(np.float64)

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

    def setMaterialModel(self, material_model_function):
        """
        Provides the material model for the mesh. The function needs to take a deformation and return the potential (w),
        the energy (w') and the stiffness (w'').

        Parameters
        ----------
        material_model_function : func
             The function needs to be able to take an ndarray and return w, w', and w'' with the same shape also as an
             ndarray.
        """
        self.material_model = material_model_function

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
            filter_in = np.zeros(T.shape[0]*4*4*3*3) == 1
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

                    for t2 in range(4):
                        # get two vertices of the tetrahedron
                        c2 = tet[t2]

                        for i in range(3):
                            for j in range(3):
                                if var[c1]:
                                    filter_in[t*4*4*3*3 + t1*4*3*3 + t2*3*3 + i*3 + j] = True
                                    # add the connection to the set
                                    stiffness_distribute_coordinates2.append((c1*3+i, c2*3+j))
            stiffness_distribute_coordinates2 = np.array(stiffness_distribute_coordinates2)
            return filter_in, (stiffness_distribute_coordinates2[:, 0], stiffness_distribute_coordinates2[:, 1])

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

    def _computeLaplace(self):
        self.Laplace = []
        for i in range(self.N_c):
            self.Laplace.append({})

        Idmat = np.eye(3)

        # iterate over all connected vertices
        for c1, c2 in self.connections:
            if c1 != c2:
                # get the inverse distance between the two vertices
                r_inv = 1 / np.linalg.norm(self.R[c1] - self.R[c2])

                # and store the in inverse distance (on the diagonal of a matrix)
                self.Laplace[c1][c2] += Idmat * -r_inv
                self.Laplace[c1][c1] += Idmat * r_inv

    """ relaxation """

    def _updateGloFAndK(self):
        t_start = time.time()
        batchsize = 10000

        self.E_glo = 0
        f_glo = np.zeros((self.N_T, 4, 3))
        K_glo = np.zeros((self.N_T, 4, 4, 3, 3))

        for i in range(int(np.ceil(self.T.shape[0]/batchsize))):
            print("updating forces and stiffness matrix %d%%" % (i/int(np.ceil(self.T.shape[0]/batchsize))*100), end="\r")
            t = slice(i*batchsize, (i+1)*batchsize)

            s_bar, s_star = self._get_s_star_s_bar(t)

            epsilon_b, dEdsbar, dEdsbarbar = self._get_applied_epsilon(s_bar, self.material_model, self.V[t])

            self._update_energy(epsilon_b, t)
            self._update_f_glo(s_star, s_bar, dEdsbar, out=f_glo[t])
            self._update_K_glo(s_star, s_bar, dEdsbar, dEdsbarbar, out=K_glo[t])

        # store the global forces in self.f_glo
        # transform from N_T x 4 x 3 -> N_v x 3
        ssp.coo_matrix((f_glo.ravel(), self.force_distribute_coordinates), shape=self.f_glo.shape).toarray(out=self.f_glo)

        # store the stiffness matrix K in self.K_glo
        # transform from N_T x 4 x 4 x 3 x 3 -> N_v * 3 x N_v * 3
        self.K_glo = ssp.coo_matrix((K_glo.ravel()[self.filter_in], self.stiffness_distribute_coordinates2),
                                shape=(self.N_c*3, self.N_c*3)).tocsr()
        print("updating forces and stiffness matrix finished %.2fs" % (time.time() - t_start))

    def _get_s_star_s_bar(self, s):
        # get the displacements of all corners of the tetrahedron (N_Tx3x4)
        # u_tim  (t in [0, N_T], i in {x,y,z}, m in {1,2,3,4})
        u_T = self.U[self.T[s]].transpose(0, 2, 1)

        # F is the linear map from T (the undeformed tetrahedron) to T' (the deformed tetrahedron)
        # F_tij = d_ij + u_tim * Phi_tmj  (t in [0, N_T], i,j in {x,y,z}, m in {1,2,3,4})
        F = np.eye(3) + u_T @ self.Phi[s]

        # iterate over the beams
        # for the elastic strain energy we would need to integrate over the whole solid angle Gamma, but to make this
        # numerically accessible, we iterate over a finite set of directions (=beams) (c.f. page 53)

        # multiply the F tensor with the beam
        # s'_tib = F_tij * s_jb  (t in [0, N_T], i,j in {x,y,z}, b in [0, N_b])
        s_bar = F @ self.s.T

        # and the shape tensor with the beam
        # s*_tmb = Phi_tmj * s_jb  (t in [0, N_T], i,j in {x,y,z}, m in {1,2,3,4}), b in [0, N_b])
        s_star = self.Phi[s] @ self.s.T

        return s_bar, s_star

    @staticmethod
    @jit(nopython=True, cache=True)
    def _get_applied_epsilon(s_bar, lookUpEpsilon, V):
        N_b = s_bar.shape[-1]

        # test if one node of the tetrahedron is variable
        # only count the energy if not the whole tetrahedron is fixed
        # countEnergy = np.any(var[T], axis=1)

        # the "deformation" amount # p 54 equ 2 part in the parentheses
        # s_tb = |s'_tib|  (t in [0, N_T], i in {x,y,z}, b in [0, N_b])
        s = np.linalg.norm(s_bar, axis=1)

        epsilon_b, epsbar_b, epsbarbar_b = lookUpEpsilon(s - 1)

        #                eps'_tb    1
        # dEdsbar_tb = - ------- * --- * V_t
        #                 s_tb     N_b
        dEdsbar = - (epsbar_b / s) / N_b * np.expand_dims(V, axis=1)

        #                  s_tb * eps''_tb - eps'_tb     1
        # dEdsbarbar_tb = --------------------------- * --- * V_t
        #                         s_tb**3               N_b
        dEdsbarbar = ((s * epsbarbar_b - epsbar_b) / (s ** 3)) / N_b * np.expand_dims(V, axis=1)

        return epsilon_b, dEdsbar, dEdsbarbar

    def _update_energy(self, epsilon_b, t):
        # test if one node of the tetrahedron is variable
        # only count the energy if not the whole tetrahedron is fixed
        countEnergy = np.any(self.var[self.T[t]], axis=1)

        # sum the energy of this tetrahedron
        # E_t = eps_tb * V_t
        self.E[t] = np.mean(epsilon_b, axis=1) * self.V[t]

        # only count the energy of the tetrahedron to the global energy if the tetrahedron has at least one
        # variable node
        self.E_glo += np.sum(self.E[t][countEnergy])

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
            How much of the displacement of each conjugate gradient step to apply. Defulat 0.066
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

        # update the forces and stiffness matrix
        self._updateGloFAndK()

        relrec = [[0, self.E_glo, np.sum(self.f_glo[self.var]**2)]]

        start = time.time()
        # start the iteration
        for i in range(i_max):
            # move the displacements in the direction of the forces one step
            # but while moving the stiffness tensor is kept constant
            du = self._solve_CG(stepper)

            # update the forces on each tetrahedron and the global stiffness tensor
            self._updateGloFAndK()

            # sum all squared forces of non fixed nodes
            ff = np.sum(self.f_glo[self.var]**2)

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
        ff = self.f_glo - self.f_ext

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

    def smoothen(self):
        ddu = 0
        for c in range(self.N_c):
            if self.var[c]:
                A = self.K_glo[c][c]

                f = self.f_glo[c]

                du = np.linalg.inv(A) * f

                self.U[c] += du

                ddu += np.linalg.norm(du)

    def computeStiffening(self, results):

        uu = self.U.copy()

        Ku = self._mulK(uu)

        kWithStiffening = np.sum(uu * Ku)
        k1 = self.CFG["K_0"]

        ds0 = self.CFG["D_0"]

        self.epsilon, self.epsbar, self.epsbarbar = buildEpsilon(k1, ds0, 0, 0, self.CFG)

        self._updateGloFAndK()

        uu = self.U.copy()

        Ku = self._mulK(uu)

        kWithoutStiffening = np.sum(uu, Ku)

        results["STIFFENING"] = kWithStiffening / kWithoutStiffening

        self.computeEpsilon()

    def computeForceMoments(self, results, rmax):
        inner = np.linalg.norm(self.R, axis=1) < rmax
        f = self.f_glo[inner]
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
        f = self.f_glo[inner]

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
            Frec.append(self.f_glo[c])

        np.savetxt(Fname, Frec)
        print(Fname, "stored.")

    def storeFden(self, Fdenname):
        Vr = np.zeros(self.N_c)

        for tt in range(self.N_T):
            for t in range(4):
                Vr[self.T[tt][t]] += self.V[tt] * 0.25

        Frec = []
        for c in range(self.N_c):
            Frec.append(self.f_glo[c] / Vr[c])

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
