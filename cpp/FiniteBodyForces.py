import os
import time

import numpy as np
from scipy.sparse import coo_matrix

from numba import jit

from .buildBeams import buildBeams
from .buildEpsilon import buildEpsilon


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

    currentgrain = 0
    N_T = 0  # the number of tetrahedrons
    N_c = 0  # the number of vertices

    s = None  # the beams, dimensions N_b x 3
    N_b = 0  # the number of beams

    epsilon = None  # the lookup table for the material model
    epsbar = None  # the lookup table for the material model
    epsbarbar = None  # the lookup table for the material model

    # the min, max and step of the discretisation of the material model
    dlmin = 0
    dlmax = 0
    dlstep = 0

    def __init__(self):
        pass

    def setMeshCoords(self, data, var=None, displacements=None, forces=None):
        # check the input
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
            displacements = np.array(displacements)
            assert displacements.shape == (self.N_c, 3)
            self.U = displacements.astype(np.float64)

        # start with every vertex being variable (non-fixed)
        if var is None:
            self.var = np.ones(self.N_c, dtype=np.int8) == 1  # type bool!
        else:
            # check the input
            var = np.array(var)
            assert var.shape == (self.N_c, )
            self.var = var.astype(bool)

        # initialize global and external forces
        self.f_glo = np.zeros((self.N_c, 3))
        if forces is None:
            self.f_ext = np.zeros((self.N_c, 3))
        else:
            # check the input
            forces = np.array(forces)
            assert forces.shape == (self.N_c, 3)
            self.f_ext = forces.astype(np.float64)

        # initialize the list of the global stiffness
        self.K_glo = np.zeros((self.N_c, self.N_c, 3, 3))

    def setMeshTets(self, data):
        # check the input
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

        self.computePhi()
        self.computeConnections()

    def computeBeams(self, N):
        beams = buildBeams(N)
        self.setBeams(beams)
        print(beams.shape[0], "beams were generated")

    def setBeams(self, beams):
        self.s = beams
        self.N_b = beams.shape[0]

    def setMaterialModel(self, material_model_function):
        self.lookUpEpsilon = material_model_function

    def computeConnections(self):
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

    def computePhi(self):
        """
        Calculate the shape tensors of the tetrahedra (see page 49)
        """
        # define the helper matrix chi
        Chi = np.zeros((4, 3))
        Chi[0, :] = [-1, -1, -1]
        Chi[1, :] = [1, 0, 0]
        Chi[2, :] = [0, 1, 0]
        Chi[3, :] = [0, 0, 1]

        B = np.zeros((3, 3))

        # for t, tet in enumerate(self.T):
        for t in range(self.N_T):
            tet = self.T[t]
            # tetrahedron matrix B (linear map of the undeformed tetrahedron T onto the primitive tetrahedron P)
            B[:, 0] = self.R[tet[1]] - self.R[tet[0]]
            B[:, 1] = self.R[tet[2]] - self.R[tet[0]]
            B[:, 2] = self.R[tet[3]] - self.R[tet[0]]

            # calculate the volume of the tetrahedron
            self.V[t] = abs(np.linalg.det(B)) / 6.0

            # if the tetrahedron has a volume
            if self.V[t] != 0.0:
                # the shape tensor of the tetrahedron is defined as Chi * B^-1
                self.Phi[t] = Chi @ np.linalg.inv(B)

    def computeLaplace(self):
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

    def updateGloFAndK(self):
        t_start = time.time()

        s_bar, s_star = self.get_s_star_s_bar()

        epsilon_b, dEdsbar, dEdsbarbar = self.get_applied_epsilon(s_bar, self.lookUpEpsilon, self.V)

        self.update_energy(epsilon_b)

        self.update_f_glo(s_star, s_bar, dEdsbar)

        self.update_K_glo(s_star, s_bar, dEdsbar, dEdsbarbar)

        print("updateGloFAndK time", time.time()-t_start, "s")

    def get_s_star_s_bar(self):
        # get the displacements of all corners of the tetrahedron (N_Tx3x4)
        # u_tim  (t in [0, N_T], i in {x,y,z}, m in {1,2,3,4})
        u_T = self.U[self.T].transpose(0, 2, 1)

        # F is the linear map from T (the undeformed tetrahedron) to T' (the deformed tetrahedron)
        # F_tij = d_ij + u_tim * Phi_tmj  (t in [0, N_T], i,j in {x,y,z}, m in {1,2,3,4})
        F = np.eye(3) + u_T @ self.Phi

        # iterate over the beams
        # for the elastic strain energy we would need to integrate over the whole solid angle Gamma, but to make this
        # numerically accessible, we iterate over a finite set of directions (=beams) (c.f. page 53)

        # multiply the F tensor with the beam
        # s'_tib = F_tij * s_jb  (t in [0, N_T], i,j in {x,y,z}, b in [0, N_b])
        s_bar = F @ self.s.T

        # and the shape tensor with the beam
        # s*_tmb = Phi_tmj * s_jb  (t in [0, N_T], i,j in {x,y,z}, m in {1,2,3,4}), b in [0, N_b])
        s_star = self.Phi @ self.s.T

        return s_bar, s_star

    @staticmethod
    @jit(nopython=True, cache=True)
    def get_applied_epsilon(s_bar, lookUpEpsilon, V):
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

    def update_energy(self, epsilon_b):
        # test if one vertex of the tetrahedron is variable
        # only count the energy if not the whole tetrahedron is fixed
        countEnergy = np.any(self.var[self.T], axis=1)

        # sum the energy of this tetrahedron
        # E_t = eps_tb * V_t
        self.E[:] = np.mean(epsilon_b, axis=1) * self.V

        # only count the energy of the tetrahedron to the global energy if the tetrahedron has at least one
        # variable vertex
        self.E_glo = np.sum(self.E[countEnergy])

    def update_f_glo(self, s_star, s_bar, dEdsbar):
        # f_tmi = s*_tmb * s'_tib * dEds'_tb  (t in [0, N_T], i in {x,y,z}, m in {1,2,3,4}, b in [0, N_b])
        f = np.einsum("tmb,tib,tb->tmi", s_star, s_bar, dEdsbar)

        coo_matrix((f.flatten(), self.force_distribute_coordinates), shape=self.f_glo.shape).toarray(out=self.f_glo)

    def update_K_glo(self, s_star, s_bar, dEdsbar, dEdsbarbar):
        #                              / |  |     \      / |  |     \                   / |    |     \
        #     ___             /  s'  w"| |s'| - 1 | - w' | |s'| - 1 |                w' | | s' | - 1 |             \
        # 1   \   *     *     |   b    \ | b|     /      \ | b|     /                   \ |  b |     /             |
        # -    > s   * s    * | ------------------------------------ * s' * s'  + ---------------------- * delta   |
        # N   /   bm    br    |                  |s'|³                  ib   lb             |s'  |              li |
        #  b  ---             \                  | b|                                       |  b |                 /
        #
        # (t in [0, N_T], i,l in {x,y,z}, m,r in {1,2,3,4}, b in [0, N_b])
        sstarsstar = np.einsum("tmb,trb->tmrb", s_star, s_star)
        s_bar_s_bar = 0.5 * (np.einsum("tb,tib,tlb->tilb", dEdsbarbar, s_bar, s_bar)
                             - np.einsum("il,tb->tilb", np.eye(3), dEdsbar))

        stiffness = np.einsum("tmrb,tilb->tmril", sstarsstar, s_bar_s_bar)

        self.K_glo = coo_matrix((stiffness.flatten(), self.stiffness_distribute_coordinates), shape=(self.N_c * self.N_c, 9)).toarray().reshape(self.N_c, self.N_c, 3, 3)

        self.K_glo_conn[:] = self.K_glo[self.connections[:, 0], self.connections[:, 1], :, :]

    def relax(self, stepper=0.066, i_max=300, rel_conv_crit=0.01, relrecname=None):
        self.updateGloFAndK()

        if relrecname is not None:
            relrec = [[np.mean(self.K_glo), self.E_glo, np.sum(self.U[self.var]**2)]]

        start = time.time()
        # start the iteration
        for i in range(i_max):
            # move the displacements in the direction of the forces one step
            # but while moving the stiffness tensor is kept constant
            du = self.solve_CG(stepper)

            # update the forces on each tetrahedron and the global stiffness tensor
            self.updateGloFAndK()

            # sum all squared forces of non fixed vertices
            ff = np.sum(self.f_glo[self.var]**2)

            # print and store status
            print("Newton ", i, ": du=", du, "  Energy=", self.E_glo, "  Residuum=", ff)

            # log and store values (if a target file was provided)
            if relrecname is not None:
                relrec.append([np.mean(self.K_glo), self.E_glo, ff])
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

    def solve_CG(self, stepper):
        """
        Solve the displacements from the current stiffness tensor using conjugate gradient.
        """
        tol = 0.00001

        maxiter = 3 * self.N_c

        uu = np.zeros((self.N_c, 3))

        # calculate the difference between the current forces on the vertices and the desired forces
        ff = self.f_glo - self.f_ext

        # ignore the force deviations on fixed vertices
        ff[~self.var, :] = 0

        # calculate the total force "amplitude"
        normb = np.sum(ff * ff)

        kk = np.zeros((self.N_c, 3))
        Ap = np.zeros((self.N_c, 3))

        # if it is not 0 (always has to be positive)
        if normb > 0:
            # calculate the force for the given displacements (we start with 0 displacements)
            # TODO are the fixed displacements from the boundary conditions somehow included here? Probably in the stiffness tensor K
            self.mulK(kk, uu)

            # the difference between the desired force deviations and the current force deviations
            rr = ff - kk

            # and store it also in pp
            pp = rr

            # calculate the total force deviation "amplitude"
            resid = np.sum(pp * pp)

            # iterate maxiter iterations
            for i in range(1, maxiter + 1):
                self.mulK(Ap, pp)

                # calculate a good step size
                alpha = resid / np.sum(pp * Ap)

                # move the displacements by the stepsize in the directions of the forces
                uu = uu + alpha * pp
                # and decrease the forces by the stepsize
                rr = rr - alpha * Ap

                # calculate the current force deviation "amplitude"
                rsnew = np.sum(rr * rr)

                # check if we are already below the convergence tolerance
                if rsnew < tol * normb:
                    break

                # update pp and resid
                pp = rr + rsnew / resid * pp
                resid = rsnew

                # print status every 100 frames
                if i % 100 == 0:
                    print(i, ":", resid, "alpha=", alpha, "du=", np.sum(uu[self.var]**2))#, end="\r")

            # now we want to apply the obtained guessed displacements to the vertex displacements "permanently"
            # therefore we add the guessed displacements times a stepper parameter to the vertex displacements

            # add the new displacements to the stored displacements
            self.U[self.var] += uu[self.var] * stepper
            # sum the applied displacements
            du = np.sum(uu[self.var]**2) * stepper * stepper

            print(i, ":", du, np.sum(self.U), stepper)

            # return the total applied displacement
            return du
        # if the deviation is already 0 we are already at our goal
        else:
            return 0

    def mulK(self, f, u):
        """
        Multiply the displacement u with the global stiffness tensor K. Or in other words, calculate the forces on all
        vertices.
        """
        dout = np.einsum("nij,nj->ni", self.K_glo_conn, u[self.connections[:, 1]])
        coo_matrix((dout.flatten(), self.connections_sparse_indices), u.shape).toarray(out=f)

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

        Ku = self.mulK(uu)

        kWithStiffening = np.sum(uu * Ku)
        k1 = self.CFG["K_0"]

        ds0 = self.CFG["D_0"]

        self.epsilon, self.epsbar, self.epsbarbar = buildEpsilon(k1, ds0, 0, 0, self.CFG)

        self.updateGloFAndK()

        uu = self.U.copy()

        Ku = self.mulK(uu)

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
