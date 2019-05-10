import os
import time

import numpy as np

from .buildBeams import buildBeams
from .buildEpsilon import buildEpsilon
from .multigridHelper import makeBoxmeshCoords, makeBoxmeshTets, setActiveFields


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
    connections = None  # a nested list which vertices are connected via a tetrahedron

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

    def __init__(self, CFG):
        self.CFG = CFG

    def makeBoxmesh(self):

        self.currentgrain = 1

        nx = self.CFG["BM_N"]
        dx = self.CFG["BM_GRAIN"]

        rin = self.CFG["BM_RIN"]
        mulout = self.CFG["BM_MULOUT"]
        rout = nx * dx * 0.5

        if rout < rin:
            print("WARNING in makeBoxmesh: Mesh BM_RIN should be smaller than BM_MULOUT*BM_GRAIN*0.5")

        self.R = makeBoxmeshCoords(dx, nx, rin, mulout)

        self.N_c = len(self.R)

        self.U = []
        for i in range(self.N_c):
            self.U.append([0.0, 0.0, 0.0])

        self.T = makeBoxmeshTets(nx, self.currentgrain)
        self.N_T = len(self.T)

        self.var = setActiveFields(nx, self.currentgrain, True)

        self.Phi = np.zeros((self.N_T, 4, 3))

        self.V = np.zeros(self.N_T)
        self.E = np.zeros(self.N_T)

    def loadMeshCoords(self, fcoordsname):
        """
        Load the vertices. Each line represents a vertex and has 3 float entries for the x, y, and z coordinates of the
        vertex.
        """

        # load the vertex file
        data = np.loadtxt(fcoordsname, dtype=float)

        # check the data
        assert data.shape[1] == 3, "coordinates in "+fcoordsname+" need to have 3 columns for the XYZ"
        print("%s read (%d entries)" % (fcoordsname, data.shape[0]))

        # store the loaded vertex coordinates
        self.R = data

        # store the number of vertices
        self.N_c = data.shape[0]

        # initialize 0 displacement for each vertex
        self.U = np.zeros((self.N_c, 3))

        # start with every vertex being variable (non-fixed)
        self.var = np.ones(self.N_c, dtype=bool)

        # initialize global and external forces
        self.f_glo = np.zeros((self.N_c, 3))
        self.f_ext = np.zeros((self.N_c, 3))

    def loadMeshTets(self, ftetsname):
        """
        Load the tetrahedrons. Each line represents a tetrahedron. Each line has 4 integer values representing the vertex
        indices.
        """
        # load the data
        data = np.loadtxt(ftetsname, dtype=int)

        # check the data
        assert data.shape[1] == 4, "vertex indices in "+ftetsname+" need to have 4 columns, the indices of the vertices of the 4 corners fo the tetrahedron"
        print("%s read (%d entries)" % (ftetsname, data.shape[0]))

        # the loaded data are the vertex indices but they start with 1 instead of 0 therefore "-1"
        self.T = data - 1

        # the number of tetrahedrons
        self.N_T = data.shape[0]

        # Phi is a 4x3 tensor for every tetrahedron
        self.Phi = np.zeros((self.N_T, 4, 3))

        # initialize the volumne and energy of each tetrahedron
        self.V = np.zeros(self.N_T)
        self.E = np.zeros(self.N_T)

    def loadBeams(self, fbeamsname):
        self.s = np.loadtxt(fbeamsname)
        self.N_b = len(self.s)

    def loadBoundaryConditions(self, dbcondsname):
        """
        Loads a boundary condition file "bcond.dat".

        It has 4 values in each line.
        If the last value is 1, the other 3 define a force on a variable vertex
        If the last value is 0, the other 3 define a displacement on a fixed vertex
        """
        # load the data in the file
        temp = np.loadtxt(dbcondsname)
        assert temp.shape[1] == 4, "the boundary conditions need 4 columns"
        assert temp.shape[0] == self.N_c, "the boundary conditions need to have the same count as the number of vertices"
        print("%s read (%d x %d entries)" % (dbcondsname, temp.shape[0], temp.shape[1]))

        # iterate over all lines
        for i in range(temp.shape[0]):
            # the last column is a bool whether the vertex is fixed or not
            self.var[i] = temp[i][3] > 0.5

            # if it is fixed, the other coordinates define the displacement
            if not self.var[i]:
                # store the vector as the displacement of the vertex
                self.U[i] = temp[i][:3]
            else:
                # if it is fixed, the given vector is the force on the vertex
                self.f_ext[i] = temp[i][:3]

    def loadConfiguration(self, Uname):
        """
        Load the displacements for the vertices. The file has to have 3 columns for the displacement in XYZ and one
        line for each vertex.
        """
        data = np.loadtxt(Uname)
        assert data.shape[1] == 3, "the displacement file needs to have 3 columnds"
        assert data.shape[0] == self.N_c, "there needs to be a displacement for each vertex"
        print("%s read (%d entries)" % (Uname, data.shape[0]))

        # store the displacement
        self.U[:, :] = data

    def computeConnections(self):
        # initialize the connections list
        self.connections = []
        for i in range(self.N_c):
            self.connections.append([])

        # iterate over all tetrahedrons
        for tt in range(self.N_T):
            # over all corners
            for t1 in range(4):
                for t2 in range(4):
                    # get two vertices of the tetrahedron
                    c1 = self.T[tt][t1]
                    c2 = self.T[tt][t2]

                    # check if the connection is already in the list, if not, add it
                    if not c2 in self.connections[c1]:
                        self.connections[c1].append(c2)

        # sort all lists (TODO why is this necessary?)
        for c1 in range(self.N_c):
            self.connections[c1] = sorted(self.connections[c1])

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

        for t in range(self.N_T):
            tet = self.T[t]
            # tetrahedron matrix B (linear map of the undeformed tetrahedron T onto the primitive tetrahedron P)
            B = np.array([self.R[tet[1]] - self.R[tet[0]],
                          self.R[tet[2]] - self.R[tet[0]],
                          self.R[tet[3]] - self.R[tet[0]]])

            # calculate the volume of the tetrahedron
            self.V[t] = abs(np.linalg.det(B)) / 6.0

            # if the tetrahedron has a volume
            if self.V[t] != 0.0:
                # calculate the inverse of the tetrahedron matrix
                Binv = np.linalg.inv(B)

                # the shape tensor of the tetrahedron is defined as Chi * B^-1
                self.Phi[t] = Chi @ Binv

    def computeLaplace(self):
        self.Laplace = []
        for i in range(self.N_c):
            self.Laplace.append({})

        Idmat = np.eye(3)

        # iterate over all connected vertices
        for c in range(self.N_c):
            for it in self.connections[c]:
                if it != c:
                    # get the distance between the two vertices
                    r = np.linalg.norm(self.R[c] - self.R[it])
                    r_inv = 1.0 / r

                    # and store the in inverse distance (on the diagonal of a matrix)
                    self.Laplace[c][it] += Idmat * -r_inv
                    self.Laplace[c][c] += Idmat * r_inv

    def computeEpsilon(self, k1=None, ds0=None, s1=None, ds1=None):

        if k1 is None:
            k1 = self.CFG["K_0"]

        if ds0 is None:
            ds0 = self.CFG["D_0"]

        if s1 is None:
            s1 = self.CFG["L_S"]

        if ds1 is None:
            ds1 = self.CFG["D_S"]

        self.epsilon, self.epsbar, self.epsbarbar = buildEpsilon(k1, ds0, s1, ds1, self.CFG)

        self.dlmin = -1.0
        self.dlmax = self.CFG["EPSMAX"]
        self.dlstep = self.CFG["EPSSTEP"]

    def updateGloFAndK(self):
        # reset the global energy
        self.E_glo = 0.0

        # initialize the list of the global stiffness
        self.K_glo = np.zeros((self.N_c, self.N_c, 3, 3))
        #for i in range(self.N_c):
        #    self.K_glo.append([np.zeros((3, 3))])

        # reset the global forces acting on the tetrahedrons
        self.f_glo[:] = 0

        # iterate over all tetrahedrons
        for tt in range(self.N_T):
            # set the energy of the tetrahedron to zero
            self.E[tt] = 0

            # print status every 100 iterations
            if tt % 100 == 0:
                print("Updating f and K", (np.floor((tt / (self.N_T + 0.0)) * 1000) + 0.0) / 10.0, "                ", end="\r")

            # test if one vertex of the tetrahedron is variable
            if self.var[self.T[tt][0]] or \
               self.var[self.T[tt][1]] or \
               self.var[self.T[tt][2]] or \
               self.var[self.T[tt][3]]:
                # only count the energy if not the whole tetrahedron is fixed
                countEnergy = True
            else:
                countEnergy = False

            # get the displacements of all corners of the tetrahedron
            u_T = np.zeros((3, 4))
            for t in range(4):
                for i in range(3):
                    u_T[i][t] = self.U[self.T[tt][t]][i]

            # the force is the displacement multipied with the shape tensor plus 1 one the diagonal (p 49)
            F = u_T @ self.Phi[tt] + np.eye(3)

            # iterate over the beams
            # for the elastic strain energy we would need to integrate over the whole solid angle Gamma, but to make this
            # numerically accessible, we iterate over a finite set of directions (=beams) (c.f. page 53)
            for b in range(self.N_b):
                # multiply the force tensor with the beam
                s_bar = F @ self.s[b]
                # and the shape tensor with the beam
                s_star = self.Phi[tt] @ self.s[b]

                # the "deformation" amount # p 54 equ 2 part in the parentheses
                deltal = np.linalg.norm(s_bar) - 1

                # we now have to pass this though the non-linearity function w (material model)
                # this function has been discretized and we interpolate between these discretisation steps

                # the discretisation step
                li = int(np.floor((deltal - self.dlmin) / self.dlstep))
                # the part between the two steps
                dli = (deltal - self.dlmin) / self.dlstep - li

                # if we are at the border of the discretisation, we stick to the end
                if li > ((self.dlmax - self.dlmin) / self.dlstep) - 2:
                    li = int(((self.dlmax - self.dlmin) / self.dlstep) - 2)
                    dli = 0

                # interpolate between the two discretisation steps
                epsilon_b = (1 - dli) * self.epsilon[li] + dli * self.epsilon[li + 1]
                epsbar_b = (1 - dli) * self.epsbar[li] + dli * self.epsbar[li + 1]
                epsbarbar_b = (1 - dli) * self.epsbarbar[li] + dli * self.epsbarbar[li + 1]

                # only count the energy of the tetrahedron to the global energy if the tetrahedron has at least one
                # variable vertex
                if countEnergy:
                    self.E_glo += epsilon_b * self.V[tt] / self.N_b

                # sum the energy of this tetrahedron
                self.E[tt] += epsilon_b * self.V[tt] / self.N_b  # p 54 (equ. 2)

                dEdsbar = -1.0 * (epsbar_b / (deltal + 1.0)) / (self.N_b + 0.0) * self.V[tt]

                dEdsbarbar = (((deltal + 1.0) * epsbarbar_b - epsbar_b) / (
                        (deltal + 1.0) * (deltal + 1.0) * (deltal + 1.0))) / (self.N_b + 0.0) * self.V[tt]

                # iterate over all 4 corners of the tetrahedron
                for t1 in range(4):
                    # get the corner index
                    c1 = self.T[tt][t1]

                    # calculate its contribution to the global force on this tetrahedron
                    # p 54 last equation (in the thesis V is missing in the equation)
                    # TODO explain the minus in dEdsbar, perhaps just the direction of the force?
                    self.f_glo[c1] += s_star[t1] * s_bar * dEdsbar

                    # iterate over all 4 corners
                    for t2 in range(4):
                        c2 = self.T[tt][t2]

                        sstarsstar = s_star[t1] * s_star[t2]

                        # calculate the global stiffness tensor
                        # p 55 second to last equation
                        # TODO why the 0.5? Because we treat each two times?
                        # TODO why the Volume? Is it missing in the thesis? Very probable
                        self.K_glo[c1][c2] += sstarsstar * 0.5 * (
                                    dEdsbarbar * s_bar[:, None] * s_bar[None, :] - np.eye(3) * dEdsbar)

    def relax(self):
        i_max = self.CFG["REL_ITERATIONS"]

        outdir = self.CFG["DATAOUT"]
        relrecname = os.path.join(outdir, self.CFG["REL_RELREC"])

        relrec = []

        start = time.time()
        # start the iteration
        for i in range(i_max):
            # move the displacements in the direction of the forces one step
            # but while moving the stiffness tensor is kept constant
            du = self.solve_CG()

            # update the forces on each tetrahedron and the global stiffness tensor
            self.updateGloFAndK()

            # sum the forces of all vertices
            ff = 0.0

            # iterate over all vertices
            for ii in range(self.N_c):
                # only if they are not fixed
                if self.var[ii]:
                    # sum the force
                    ff += np.linalg.norm(self.f_glo[3 * ii:3 * ii + 3])

            # print and store status
            print("Newton ", i, ": du=", du, "  Energy=", self.E_glo, "  Residuum=", ff)
            relrec.append([du, self.E_glo, ff])
            np.savetxt(relrecname, relrec)

            # if we have passed 6 iterations calculate average and std
            if i > 6:
                # calculate the average energy over the last 6 iterations
                s = len(relrec)

                Emean = 0.0
                cc = 0.0

                for ii in range(s - 1, s - 6, -1):
                    Emean += relrec[ii][1]
                    cc = cc + 1.0

                Emean = Emean / cc

                # calculate the standard deviation of the last 6 iterations
                Estd = 0.0
                cc = 0.0

                for ii in range(s - 1, s - 6, -1):
                    Estd += (Emean - relrec[ii][1]) * (Emean - relrec[ii][1])
                    cc = cc + 1.0

                Estd = np.sqrt(Estd) / cc

                # if the iterations converge, stop the iteration
                if (Estd / Emean) < self.CFG["REL_CONV_CRIT"]:
                    break

        # print the elapsed time
        finish = time.time()
        print("| time for relaxation was", finish - start)

    def solve_CG(self):
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

        # if it is not 0 (always has to be positive)
        if normb > 0:
            # calculate the force for the given displacements (we start with 0 displacements)
            # TODO are the fixed displacements from the boundary conditions somehow included here? Probably in the stiffness tensor K
            kk = self.mulK(uu)

            # the difference between the desired force deviations and the current force deviations
            rr = ff - kk

            # and store it also in pp? TODO ?
            pp = rr

            # calculate the total force deviation "amplitude"
            resid = np.sum(pp * pp)

            # iterate maxiter iterations
            for i in range(1, maxiter + 1):
                # calculate the current forces
                Ap = self.mulK(pp)

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
                if i % 1 == 0:
                    print(i, ":", resid, "alpha=", alpha, end="\r")

            # now we want to apply the obtained guessed displacements to the vertex displacements "permanently"
            # therefore we add the guessed displacements times a stepper parameter to the vertex displacements

            # the stepper
            stepper = self.CFG["REL_SOLVER_STEP"]

            # sum the total applied displacements
            du = 0

            # iterate over all non fixed vertices
            for c in range(self.N_c):
                if self.var[c]:
                    # get the displacement
                    dU = uu[3 * c:3 * c + 3]

                    # apply it
                    # TODO possible optimistation multiply the whole uu array
                    self.U[c] += dU * stepper

                    # sum the applied displacements. TODO but why stepper**2?
                    du += np.linalg.norm(dU) * stepper * stepper

            # return the total applied displacement
            return du
        # if the deviation is already 0 we are already at our goal
        else:
            return 0

    def smoothen(self):
        ddu = 0
        for c in range(self.N_c):
            if self.var[c]:
                A = self.K_glo[c][c]

                f = self.f_glo[c]

                du = np.linalg.inv(A) * f

                self.U[c] += du

                ddu += np.linalg.norm(du)

        print("du=", ddu)

    def mulK(self, u):
        """
        Multiply the displacement u with the global stiffness tensor K. Or in other words, calculate the forces on all
        vertices.
        """
        # start with an empty force array
        f = np.zeros((self.N_c,  3))

        # iterate over all vertices
        for c1 in range(self.N_c):
            # only for non fixed vertices
            if not self.var[c1]:
                continue

            ff = np.zeros(3)
            # sum over all connections
            for c2 in self.connections[c1]:
                # get the offset of the partner
                uu = u[c2]
                # get the stiffness matrix
                A = self.K_glo[c1, c2]

                # the force is the stiffness matrix times the displacement
                ff += A @ uu

            # store the force in the return value
            f[c1] = ff

        # return the obtained forces
        return f

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
