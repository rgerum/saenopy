import os
import time

import numpy as np
from scipy.sparse import coo_matrix

from numba import jit

from .buildBeams import buildBeams
from .buildEpsilon import buildEpsilon
from .conjugateGradient import cg


class FiniteBodyForces:
    R = None  # the 3D positions of the vertices, dimension: N_c x 3
    T = None  # the tetrahedrons' 4 corner vertices (defined by index), dimensions: N_T x 4
    E = None  # the energy stored in each tetrahedron, dimensions: N_T
    V = None  # the volume of each tetrahedron, dimensions: N_T
    var = None  # a bool if a vertex is movable

    Phi = None  # the shape tensor of each tetrahedron, dimensions: N_T x 4 x 3
    U = None  # the displacements of each vertex, dimensions: N_c x 3

    f_glo = None  # the global forces on each vertex, dimensions: N_c x 3
    f_ext = None  # the external forces on each vertex, dimensions: N_c x 3
    K_glo = None  # the global stiffness tensor, dimensions: N_c x N_c x 3 x 3

    Laplace = None

    E_glo = 0  # the global energy

    # a list of all vertices are connected via a tetrahedron, stored as pairs: dimensions: N_connections x 2
    connections = None

    N_T = 0  # the number of tetrahedrons
    N_c = 0  # the number of vertices

    s = None  # the beams, dimensions N_b x 3
    N_b = 0  # the number of beams

    material_model = None  # the function specifying the material model

    def setMeshCoords(self, data, var=None, displacements=None, forces=None):
        """
        Provide mesh coordinates, optional with a flag if they can be moved, a displacement and a force.

        Parameters
        ----------
        data : ndarray
            The coordinates of the vertices. Dimensions Nx3
        var : ndarray, optional
            A boolean value wether the vertex is allowed to move. By default all vertices can be moved. Dimensions N
        displacements : ndarray, optional
            The initial displacement of the vertex. Dimensions Nx3
        forces : ndarray, optional
            The forces on the vertex. Dimensions Nx3
        """
        # check the input
        data = np.asarray(data)
        assert len(data.shape) == 2, "Mesh vertex data needs to be Nx3."
        assert data.shape[1] == 3, "Mesh vertices need to have 3 spacial coordinate."

        # store the loaded vertex coordinates
        self.R = data.astype(np.float64)

        # store the number of vertices
        self.N_c = data.shape[0]

        # initialize 0 displacement for each vertex
        if displacements is None:
            self.U = np.zeros((self.N_c, 3))
        else:
            # check the input
            displacements = np.asarray(displacements)
            assert displacements.shape == (self.N_c, 3)
            self.U = displacements.astype(np.float64)

        # start with every vertex being variable (non-fixed)
        if var is None:
            self.var = np.ones(self.N_c, dtype=np.bool)
        else:
            # check the input
            var = np.asarray(var)
            assert var.shape == (self.N_c, )
            self.var = var.astype(bool)

        # initialize global and external forces
        self.f_glo = np.zeros((self.N_c, 3))
        if forces is None:
            self.f_ext = np.zeros((self.N_c, 3))
        else:
            # check the input
            forces = np.asarray(forces)
            assert forces.shape == (self.N_c, 3)
            self.f_ext = forces.astype(np.float64)

    def setMeshTets(self, data):
        """
        Provide mesh tetrahedrons. Each tetrahedron consts of the indices of the 4 vertices which it connects.

        Parameters
        ----------
        data : ndarray
            The vertex indices of the 4 cornders. Dimensions Nx4
        """
        # check the input
        data = np.asarray(data)
        assert len(data.shape) == 2, "Mesh tetrahedrons needs to be Nx4."
        assert data.shape[1] == 4, "Mesh tetrahedrons need to have 4 corners."
        assert 0 <= data.min(), "Mesh tetrahedron vertex indices are not allowed to be negativ."
        assert data.max() < self.N_c, "Mesh tetrahedron vertex indices cannot be bigger than the number of vertices."

        # store the tetrahedron data (needs to be int indices)
        self.T = data.astype(np.int)

        # the number of tetrahedrons
        self.N_T = data.shape[0]

        # Phi is a 4x3 tensor for every tetrahedron
        self.Phi = np.zeros((self.N_T, 4, 3))

        # initialize the volume and energy of each tetrahedron
        self.V = np.zeros(self.N_T)
        self.E = np.zeros(self.N_T)

        self._computePhi()
        self._computeConnections()

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
        # initialize the connections as a set (to prevent double entries)
        connections = set()

        # iterate over all tetrahedrons
        for tet in self.T:
            # over all corners
            for t1 in range(4):
                c1 = tet[t1]

                # only for non fixed vertices
                if not self.var[c1]:
                    continue

                for t2 in range(4):
                    # get two vertices of the tetrahedron
                    c2 = tet[t2]

                    # add the connection to the set
                    connections.add((c1, c2))

        # convert the list of sets to an array N_connections x 2
        self.connections = np.array(list(connections), dtype=int)

        # initialize the stiffness matrix premultiplied with the connections
        self.K_glo_conn = np.zeros((self.connections.shape[0], 3, 3))

        # calculate the indices for "mulK" for multiplying the displacements to the stiffnes matrix
        x, y = np.meshgrid(np.arange(3), self.connections[:, 0])
        self.connections_sparse_indices = (y.flatten(), x.flatten())

        # calculate the indices for "update_f_glo"
        y, x = np.meshgrid(np.arange(3), self.T.flatten())
        self.force_distribute_coordinates = (x.flatten(), y.flatten())

        # calculate the indices for "update_K_glo"
        pairs = np.array(np.meshgrid(np.arange(4), np.arange(4))).reshape(2, -1)
        tensor_pairs = self.T[:, pairs.T]  # T x 16 x 2
        tensor_index = tensor_pairs[:, :, 0] + tensor_pairs[:, :, 1] * self.N_c
        y, x = np.meshgrid(np.arange(9), tensor_index.flatten())
        self.stiffness_distribute_coordinates = (x.flatten(), y.flatten())

        # calculate the indices for "update_K_glo"
        self.stiffness_distribute_coordinates2 = []
        self.stiffness_distribute_var = []
        self.filter_in = []
        # iterate over all tetrahedrons
        for tet in self.T:
            # over all corners
            for t1 in range(4):
                c1 = tet[t1]

                for t2 in range(4):
                    # get two vertices of the tetrahedron
                    c2 = tet[t2]

                    for i in range(3):
                        for j in range(3):
                            # add the connection to the set
                            self.filter_in.append(self.var[c1])
                            if self.var[c1]:
                                self.stiffness_distribute_coordinates2.append((c1*3+i, c2*3+j))

        self.stiffness_distribute_var = np.outer(self.var, np.ones(3, dtype=bool)).flatten()
        self.filter_in = np.array(self.filter_in, dtype=bool)
        self.stiffness_distribute_coordinates2 = np.array(self.stiffness_distribute_coordinates2)
        self.stiffness_distribute_coordinates2 = (self.stiffness_distribute_coordinates2[:, 0], self.stiffness_distribute_coordinates2[:, 1])

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

    def _updateGloFAndK(self):
        t_start = time.time()
        batchsize = 10000

        self.E_glo = 0
        f_glo = np.zeros((self.N_T, 4, 3))
        K_glo = np.zeros((self.N_T, 4, 4, 3, 3))

        for i in range(int(np.ceil(self.T.shape[0]/batchsize))):
            t = slice(i*batchsize, (i+1)*batchsize)

            s_bar, s_star = self._get_s_star_s_bar(t)

            epsilon_b, dEdsbar, dEdsbarbar = self._get_applied_epsilon(s_bar, self.material_model, self.V[t])

            self._update_energy(epsilon_b, t)
            self._update_f_glo(s_star, s_bar, dEdsbar, out=f_glo[t])
            self._update_K_glo(s_star, s_bar, dEdsbar, dEdsbarbar, out=K_glo[t])

        coo_matrix((f_glo.ravel(), self.force_distribute_coordinates), shape=self.f_glo.shape).toarray(out=self.f_glo)

        self.K_glo = coo_matrix((K_glo.ravel()[self.filter_in], self.stiffness_distribute_coordinates2),
                                shape=(self.N_c*3, self.N_c*3)).tocsr()
        print("updateGloFAndK time", time.time() - t_start, "s")


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

        # test if one vertex of the tetrahedron is variable
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
        # test if one vertex of the tetrahedron is variable
        # only count the energy if not the whole tetrahedron is fixed
        countEnergy = np.any(self.var[self.T[t]], axis=1)

        # sum the energy of this tetrahedron
        # E_t = eps_tb * V_t
        self.E[t] = np.mean(epsilon_b, axis=1) * self.V[t]

        # only count the energy of the tetrahedron to the global energy if the tetrahedron has at least one
        # variable vertex
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

    def relax(self, stepper=0.066, i_max=300, rel_conv_crit=0.01, relrecname=None):
        """
        Calculate the displacement of the vertices for the given external forces.

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

        # check if we have vertices
        if self.N_c == 0:
            raise ValueError("No vertices have yet been set. Call setMeshCoords first.")

        # check if we have tetrahedrons
        if self.N_T == 0:
            raise ValueError("No tetrahedrons have yet been set. Call setMeshTets first.")

        # check if we have a material model
        if self.material_model is None:
            raise ValueError("No material model has been set. Call setMaterialModel first.")

        # if the beams have not been set yet, initialize them with the default configuration
        if self.s is None:
            self.setBeams()

        # update the forces and stiffness matrix
        self._updateGloFAndK()

        if relrecname is not None:
            relrec = [[0, self.E_glo, np.sum(self.f_glo[self.var]**2)]]

        start = time.time()
        # start the iteration
        for i in range(i_max):
            # move the displacements in the direction of the forces one step
            # but while moving the stiffness tensor is kept constant
            du = self._solve_CG(stepper)

            # update the forces on each tetrahedron and the global stiffness tensor
            self._updateGloFAndK()

            # sum all squared forces of non fixed vertices
            ff = np.sum(self.f_glo[self.var]**2)

            # print and store status
            print("Newton ", i, ": du=", du, "  Energy=", self.E_glo, "  Residuum=", ff)

            # log and store values (if a target file was provided)
            if relrecname is not None:
                relrec.append([du, self.E_glo, ff])
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
        # calculate the difference between the current forces on the vertices and the desired forces
        ff = self.f_glo - self.f_ext

        # ignore the force deviations on fixed vertices
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

    def computeForceMoments(self, results):
        rmax = self.CFG["FM_RMAX"]

        in_ = np.zeros(self.N_c, dtype=bool)

        Rcms = np.zeros(3)
        fsum = np.zeros(3)
        B = np.zeros(3)
        B1 = np.zeros(3)
        B2 = np.zeros(3)
        A = np.zeros((3, 3))

        I = np.eye(3)

        for c in range(self.N_c):
            if abs(self.R[c]) < rmax:
                in_[c] = True

                fsum += f

                f = self.f_glo[c]

                B1 += self.R[c] * np.linalg.norm(f)
                B2 += f @ self.R[c] @ f  # TODO ensure matrix multiplication is the right thing to do here

                A += I * np.linalg.norm(f) - f[:, None] * f[None, :]

        B = B1 - B2

        Rcms = A.inv() @ B

        results["FSUM_X"] = fsum[0]
        results["FSUM_Y"] = fsum[1]
        results["FSUM_Z"] = fsum[2]
        results["FSUMABS"] = abs(fsum)

        results["CMS_X"] = Rcms[0]
        results["CMS_Y"] = Rcms[1]
        results["CMS_Z"] = Rcms[2]

        M = np.zeros((3, 3))

        contractility = 0.0

        for c in range(self.N_c):
            if in_[c]:
                RR = self.R[c] - Rcms

                contractility += (RR @ self.f_glo[c]) / abs(RR)

        results["CONTRACTILITY"] = contractility

        vecs = buildBeams(150)

        fmax = 0.0
        fmin = 0.0
        mmax = 0.0
        mmin = 0.0
        bmax = 0
        bmin = 0

        for b in range(len(vecs)):
            ff = 0
            mm = 0

            for c in range(self.N_c):
                if in_[c]:
                    RR = self.R[c] - Rcms
                    eR = RR / abs(RR)

                    ff += (eR @ vecs[b]) * (vecs[b] @ self.f_glo[c])
                    mm += (RR @ vecs[b]) * (vecs[b] @ self.f_glo[c])

            if mm > mmax or b == 0:
                bmax = b
                fmax = ff
                mmax = mm

            if mm < mmin or b == 0:
                bmin = b
                fmin = ff
                mmin = mm

        vmid = np.cross(vecs[bmax], vecs[bmin])
        vmid = vmid / abs(vmid)

        fmid = 0
        mmid = 0

        for c in range(self.N_c):
            if in_[c]:
                RR = self.R[c] - Rcms
                eR = RR / abs(RR)

                fmid += (eR @ vmid) * (vmid @ self.f_glo[c])
                mmid += (RR @ vmid) * (vmid @ self.f_glo[c])

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
            Frec.apppend(self.f_glo[c] / Vr[c])

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
