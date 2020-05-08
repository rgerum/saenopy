import time

import numpy as np
import scipy.sparse as ssp

from .solver import Solver
from .conjugateGradient import cg
from .stack3DHelper import crosscorrelateStacks, getSubstack, findLocalDisplacement


class timeit:

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        print("Start", self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("End", self.name, "%.3fs" % (time.time() - self.start))


class IA:
    def __init__(self, A, I):
        self.A = A
        self.I = I

    def __matmul__(self, other):
        return self.I * other + self.A @ other


def norm(x):
    return np.sum(x**2)


class VirtualBeads:
    def __init__(self, CFG, ssX=None, ssY=None, ssZ=None, ddX=None, ddY=None, ddZ=None):
        self.CFG = CFG
        self.sX = ssX
        self.sY = ssY
        self.sZ = ssZ
        self.dX = ddX
        self.dY = ddY
        self.dZ = ddZ

        self.S_0 = []

        self.U_found = []
        self.R_found = []
        self.U_guess = []

        self.I = None
        self.Itrans = None
        self.ItransI = None

        self.vbead = []
        self.outofstack = []

        self.u_x = []
        self.u_y = []
        self.u_z = []

        self.localweight = []
        self.conconnections = None
        self.oldconconnections = None
        self.b = []

        self.Drift = np.array([0, 0, 0])

    def allBeads(self, M):
        thresh = self.CFG["VB_SX"] * self.dX + 2.0 * self.CFG["DRIFT_STEP"]

        self.vbead = np.zeros(M.N_c, dtype=bool)

        Scale = np.eye(3) * np.array([self.dX, self.dY, self.dZ])

        scaledShift = np.array([self.sX / 2, self.sY / 2, self.sZ / 2])

        Trans = Scale

        for t in range(M.N_c):
            R = Trans @ M.R[t] + scaledShift

            if thresh < R[0] < (self.sX - thresh) and \
                    thresh < R[1] < (self.sY - thresh) and \
                    thresh < R[2] < (self.sZ - thresh):
                self.vbead[t] = True

    def computeOutOfStack(self, M):
        thresh = self.CFG["VB_SX"] * self.dX + 2.0 * self.CFG["DRIFT_STEP"]

        self.outofstack = np.array(M.N_c, dtype=bool)

        Scale = np.eye(3) * np.array([self.dX, self.dY, self.dZ])

        scaledShift = np.array([self.sX / 2, self.sY / 2, self.sZ / 2])

        Trans = Scale

        for t in range(M.N_c):
            R = Trans @ M.R[t] + scaledShift

            if thresh < R[0] < (self.sX - thresh) and \
                    thresh < R[1] < (self.sY - thresh) and \
                    thresh < R[2] < (self.sZ - thresh):
                self.outofstack[t] = False

    def loadVbeads(self, VBname):
        vbead_temp = np.loadtxt(VBname)

        self.vbead = vbead_temp[:, 0] > 0.5

    def loadGuess(self, M, ugname):
        from .loadHelpers import loadBoundaryConditions
        var, U, f_ext = loadBoundaryConditions(ugname)
        M.var = var
        M.f_ext = f_ext
        self.U_guess = U
        M._computeConnections()

    def updateLocalWeigth(self, M, method):

        self.localweight[:] = 1

        Fvalues = np.linalg.norm(M.f_glo, axis=1)
        Fmedian = np.median(Fvalues[M.var])

        if method == "singlepoint":
            self.localweight[int(self.CFG["REG_FORCEPOINT"])] = 1.0e-10

        if method == "bisquare":
            k = 4.685

            index = Fvalues < k * Fmedian
            self.localweight[index * M.var] *= (1 - (Fvalues / k / Fmedian) * (Fvalues / k / Fmedian)) * (
                    1 - (Fvalues / k / Fmedian) * (Fvalues / k / Fmedian))
            self.localweight[~index * M.var] *= 1e-10

        if method == "cauchy":
            k = 2.385

            if Fmedian > 0:
                self.localweight[M.var] *= 1.0 / (1.0 + np.power((Fvalues / k / Fmedian), 2.0))
            else:
                self.localweight *= 1.0

        if method == "huber":
            k = 1.345

            index = (Fvalues > (k * Fmedian)) * M.var
            self.localweight[index] = k * Fmedian / Fvalues[index]

        index = self.localweight < 1e-10
        self.localweight[index * M.var] = 1e-10

        counter = np.sum(1.0 - self.localweight[M.var])
        counterall = np.sum(M.var)

        print("total weight: ", counter, "/", counterall)

    def loadUfound(self, Uname, Sname):
        self.U_found = np.loadtxt(Uname)
        self.S_0 = np.loadtxt(Sname)
        self.vbead = self.S_0 > 0.7

    def computeConconnections(self, M):
        """
        conconections seem to be the indirect connections e.g. A->B->C
        this means a has an indirect connection (or 2. order connection) to c
        """
        self.conconnections = []
        for c1 in range(M.N_c):
            self.conconnections.append(set())

        for c1 in range(M.N_c):
            for c2 in M.connections[c1]:
                for c3 in M.connections[c2]:
                    if c3 not in self.conconnections[c1]:
                        self.conconnections[c1].add(c3)

    def computeConconnections_Laplace(self, M):
        """
        seems to adanve the order of which connections to look at
        """
        self.conconnections = []
        self.oldconconnections = []
        for c1 in range(M.N_c):
            self.oldconconnections[c1] = self.conconnections[c1]
            self.conconnections.append(set())

        for c1 in range(M.N_c):
            for c2 in self.oldconconnections[c1]:
                for c3 in self.oldconconnections[c2]:
                    if c3 not in self.conconnections[c1]:
                        self.conconnections[c1].add(c3)

    def substractMedianDisplacements(self):
        index = np.abs(self.U_found) > 0.01 * self.CFG["VOXELSIZEX"]
        u_median = np.median(self.U_found[index], axis=0)
        print("Median displacement calculated to", u_median)

        self.U_found -= u_median

    def findDriftCoarse(self, stackr: np.ndarray, stacka: np.ndarray, abs_range: float, step: float) -> np.ndarray:

        Smax = -1.0
        best_shift = np.array([0, 0, 0])

        # iterate over all shifts in the rage and step and find the best correlation
        for dx in np.arange(-abs_range, abs_range, step):
            for dy in np.arange(-abs_range, abs_range, step):
                for dz in np.arange(-abs_range, abs_range, step):
                    shift = np.array([dx, dy, dz])

                    # get the correlation for this shift
                    Stemp = self.testDrift(stackr, stacka, shift)

                    # if it is better than the previous best, store it
                    if Stemp > Smax:
                        best_shift = shift
                        Smax = Stemp

        # return the best shift
        return best_shift

    def findDrift(self, stackr: np.ndarray, stacka: np.ndarray) -> np.ndarray:

        lambd = 0.0

        hinit = float(self.CFG["DRIFT_STEP"])
        subpixel = float(self.CFG["SUBPIXEL"])

        vsx = float(self.CFG["VOXELSIZEX"])
        vsy = float(self.CFG["VOXELSIZEY"])
        vsz = float(self.CFG["VOXELSIZEZ"])

        P = np.array([
            [1, 1, 1],
            [-1, -1, 1],
            [-1, 1, -1],
            [1, -1, -1],
        ]) * hinit

        S = [
            self.testDrift(stackr, stacka, P[0] + self.Drift) - lambd * norm(P[0]),
            self.testDrift(stackr, stacka, P[1] + self.Drift) - lambd * norm(P[1]),
            self.testDrift(stackr, stacka, P[2] + self.Drift) - lambd * norm(P[2]),
            self.testDrift(stackr, stacka, P[3] + self.Drift) - lambd * norm(P[3]),
        ]

        # std::cout<<S[0]<<" "<<S[1]<<" "<<S[2]<<" "<<S[3]<<" \n";
        done = False
        for ii in range(100000):
            if done:
                break

            mini = np.argmin(S)
            maxi = np.argmax(S)

            # std::cout<<"| new cycle mini = "<<mini<<std::endl;
            # reflect
            P_plane = np.array([0.0, 0.0, 0.0])
            for i in range(4):
                if i != mini:
                    P_plane += P[i]
            P_plane /= 3

            P_ref = P[mini] + (P_plane - P[mini]) * 2.0

            S_ref = self.testDrift(stackr, stacka, P_ref + self.Drift) - lambd * norm(P_ref)

            if S_ref > S[maxi]:
                # expand
                # std::cout<<"| expanding "<<std::endl;
                P_exp = P[mini] + (P_plane - P[mini]) * 3.0
                S_exp = self.testDrift(stackr, stacka, P_exp + self.Drift) - lambd * norm(P_exp)

                if S_exp > S_ref:
                    # std::cout<<"| took expanded "<<std::endl;
                    P[mini] = P_exp
                    S[mini] = S_exp

                else:
                    # std::cout<<"| took reflected (expanded was worse)"<<std::endl;
                    P[mini] = P_ref
                    S[mini] = S_ref
            else:
                bsw = False
                for i in range(4):
                    if i != mini:
                        if S_ref > S[i]:
                            bsw = True

                if bsw:
                    # std::cout<<"| took reflected (better than second worst)"<<std::endl;
                    P[mini] = P_ref
                    S[mini] = S_ref

                else:
                    if S_ref > S[maxi]:
                        # std::cout<<"| took reflected (not better than second worst)"<<std::endl;
                        P[mini] = P_ref
                        S[mini] = S_ref
                    else:
                        P_con = P[mini] + (P_plane - P[mini]) * 0.5
                        S_con = self.testDrift(stackr, stacka, P_con + self.Drift) - lambd * norm(P_con)

                        if S_con > S[mini]:
                            # std::cout<<"| took contracted"<<std::endl;
                            P[mini] = P_con
                            S[mini] = S_con
                        else:
                            # std::cout<<"| contracting myself"<<std::endl;
                            for i in range(4):
                                if i != maxi:
                                    P[i] = (P[maxi] + P[i]) * 0.5
                                    S[i] = self.testDrift(stackr, stacka, P[i] + self.Drift) - lambd * norm(P[i])

            # std::cout<<" S_ref = "<<S_ref<<std::endl;
            mx = np.mean(P[:, 0])
            my = np.mean(P[:, 1])
            mz = np.mean(P[:, 2])

            stdx = np.sqrt(np.sum((P[:, 0] - mx)**2)) / 4
            stdy = np.sqrt(np.sum((P[:, 1] - my)**2)) / 4
            stdz = np.sqrt(np.sum((P[:, 2] - mz)**2)) / 4

            # std::cout<<"stdx = "<<stdx<<" ; stdy = "<<stdy<<" ; stdz = ";
            if stdx < subpixel * vsx and stdy < subpixel * vsy and stdz < subpixel * vsz:
                done = True

        mini = np.argmin(S)
        maxi = np.argmax(S)

        return P[maxi] + self.Drift

    def testDrift(self, stack1: np.ndarray, stack2: np.ndarray, D: np.ndarray) -> float:

        Scale = np.array([[1.0 / self.dX, 0.0, 0.0], [0.0, 1.0 / self.dY], [0.0, 0.0, 0.0, 1.0 / self.dZ]])

        U = Scale @ D

        return crosscorrelateStacks(stack1, stack2, U)

    def findDisplacements(self, stack_r: np.ndarray, stack_a: np.ndarray, M: Solver, lambd: float):

        Srec = []

        R = []
        U = []

        self.U_found = np.zeros((M.N_c, 4))
        self.S_0 = np.zeros(M.N_c)

        Scale = np.array([[1.0 / self.dX, 0.0, 0.0], [0.0, 1.0 / self.dY, 0.0], [0.0, 0.0, 1.0 / self.dZ]])

        scaledShift = np.array([self.sX / 2, self.sY / 2, self.sZ / 2])

        Trans = Scale
        Transinv = np.linalg.inv(Trans)

        tend = M.R.shape[0]

        U_new = []
        Umean = []

        Stemp = 0.0

        n = 0

        Sd = []
        Ud = []

        """ NEW CHRISTOPH """

        sgX = int(self.CFG["VB_SX"])
        sgY = int(self.CFG["VB_SY"])
        sgZ = int(self.CFG["VB_SZ"])

        weight = np.zeros([sgX, sgY, sgZ])

        for ii in range(sgX):
            for jj in range(sgY):
                for kk in range(sgZ):
                    xxx = (ii - sgX / 2) * self.dX
                    yyy = (jj - sgY / 2) * self.dY
                    zzz = (kk - sgZ / 2) * self.dZ

                    width = sgX * self.dX * 0.25

                    weight[ii][jj][kk] = np.exp(-(xxx * xxx + yyy * yyy + zzz * zzz) / (2 * width))

        weight /= np.mean(weight)

        # sumweight=0.0;

        # for(int ii=0; ii<sgX; ii++) for(int jj=0; jj<sgY; jj++) for(int kk=0; kk<sgZ; kk++) sumweight+=weight[ii][jj][kk];

        # std::cout<<"sumweight = "<<sumweight<<"\n\n";

        """ END NEW CHRISTOPH """

        # iterate over all nodes
        for t in range(tend):
            # only if the node is selected as a "bead"
            if self.vbead[t]:
                # which method to use
                if int(self.CFG["VB_N"]) == 1:
                    print("finding Displacements", (np.floor((t / (tend + 0.0)) * 1000) + 0.0) / 10.0, "% S=", Stemp,
                          "        \r", end="")

                    R = Trans @ M.R[t] + scaledShift

                    # get the mean from the overall drift
                    Umean = self.Drift

                    # maybe add the intial guess
                    if bool(self.CFG["INITIALGUESS"]):
                        Umean += self.U_guess[t]

                    # transform with the scaling (µm -> pixel)
                    Umean = Trans * Umean

                    # get a substack
                    substackr = getSubstack(stack_r, R, sgX, sgY, sgZ)

                    # NEW CHRSITOPH
                    substackr *= weight

                    # get the displacement
                    U_new = findLocalDisplacement(substackr, stack_a, R, Umean, Srec, lambd, self.CFG["subpixel"])
                    # store the correlation
                    self.S_0[t] = Srec[-1]
                    Stemp = Srec[-1]

                    # rescale (pixel -> µm)
                    U_new = Transinv @ U_new

                    # and store
                    self.U_found[t] = U_new

                else:

                    print("finding Displacements", (np.floor((t / (tend + 0.0)) * 1000) + 0.0) / 10.0, "% S=", Stemp,
                          "n=", n, "      \r", end="")

                    Ud = []
                    Sd = []

                    imax = int(self.CFG["VB_N"]) - 1

                    for i in range(imax + 1):
                        for j in range(imax + 1):
                            for k in range(imax + 1):

                                dx = (i - (imax * 0.5)) / (imax + 1.0) * float(self.CFG["VB_SX"])
                                dy = (j - (imax * 0.5)) / (imax + 1.0) * float(self.CFG["VB_SY"])
                                dz = (k - (imax * 0.5)) / (imax + 1.0) * float(self.CFG["VB_SZ"])

                                R = Trans * M.R[t] + scaledShift + np.array([dx, dy, dz])

                                Umean = self.Drift

                                Umean = Trans @ Umean

                                substackr = self.getSubstack(stack_r, R)
                                U_new = self.findLocalDisplacement(substackr, stack_a, R, Umean, Srec, lambd)
                                Stemp = Srec[-1]

                                U_new = Transinv @ U_new

                                if Stemp > float(self.CFG["VB_MINMATCH"]):
                                    Ud.append(U_new)
                                    Sd.append(Stemp)

                    if Sd.size() > 0:

                        U_new = np.array([0.0, 0.0, 0.0])
                        Stemp = 0.0

                        for it in Sd:
                            Stemp += it
                        for it in Ud:
                            U_new += it

                        U_new = U_new * (1.0 / len(Sd))
                        Stemp = Stemp * (1.0 / len(Sd))

                        n = len(Sd)

                    else:

                        U_new = np.array([0.0, 0.0, 0.0])
                        Stemp = -1.0
                        n = 0

                    self.S_0[t] = Srec[-1]
                    self.U_found[t] = U_new

    def refineDisplacements(self, stack_r: np.ndarray, stack_a: np.ndarray, M: Solver, lambd: float):

        Srec = []

        Scale = np.array([[1.0 / self.dX, 0.0, 0.0], [0.0, 1.0 / self.dY, 0.0], [0.0, 0.0, 1.0 / self.dZ]])

        scaledShift = np.array([self.sX / 2, self.sY / 2, self.sZ / 2])

        Trans = Scale

        Transinv = np.linalg.inv(Trans)

        indices = []
        Svalues = []

        vbeadcount = 0

        for c in range(M.N_c):
            if self.vbead[c]:
                indices.append(c)
                Svalues.append(self.S_0[c])
                vbeadcount += 1

        # TODO BubbleSort_IVEC_using_associated_dVals(indices,Svalues)

        Nrenew = np.floor(vbeadcount)

        it = []

        Umean = []
        U_new = []
        R = []
        Stemp = 0.0

        for i in range(Nrenew):
            print("refining displacements:", (np.floor((i / (Nrenew + 0.0)) * 1000) + 0.0) / 10.0, "%     S=", Stemp,
                  "           \r", end="")

            c = indices[i]
            cccount = 0

            Umean = np.array([0.0, 0.0, 0.0])

            for it in M.connections:
                cc = it

                if self.vbead[cc]:
                    Umean += self.U_found[cc]
                    cccount += 1

            if cccount > 0:
                Umean = Umean / (cccount + 0.0)

                R = (Trans @ ((M.R[c])) + scaledShift)
                Umean = (Trans @ ((Umean)))

                substackr = getSubstack(stack_r, R)

                U_new = findLocalDisplacement(substackr, stack_a, R, Umean, Srec, lambd)
                self.S_0[c] = Srec[-1]
                substacka = getSubstack(stack_a, R)
                U_new -= findLocalDisplacement(substacka, stack_r, R + Umean, Umean * (-1.0), Srec, lambd)
                self.S_0[c] += Srec[-1]

                self.S_0[c] *= 0.5

                U_new = Transinv @ U_new * 0.5

                self.U_found[c] = U_new

                Stemp = self.S_0[c]

    def _computeRegularizationAAndb(self, M, alpha):
        KA = M.K_glo.multiply(np.repeat(self.localweight * alpha, 3)[None, :])
        self.KAK = KA @ M.K_glo
        self.A = self.I + self.KAK

        self.b = (KA @ M.f_glo.ravel()).reshape(M.f_glo.shape)

        index = M.var * self.vbead
        self.b[index] += self.U_found[index] - M.U[index]

    def _recordRegularizationStatus(self, relrecname, M, relrec):
        alpha = self.CFG["ALPHA"]

        indices = M.var & self.vbead
        btemp = self.U_found[indices] - M.U[indices]
        uuf2 = np.sum(btemp ** 2)
        suuf = np.sum(np.linalg.norm(btemp, axis=1))
        bcount = btemp.shape[0]

        u2 = np.sum(M.U[M.var] ** 2)

        f = np.zeros((M.N_c, 3))
        f[M.var] = M.f_glo[M.var]

        ff = np.sum(np.sum(f ** 2, axis=1) * self.localweight * M.var)

        L = alpha * ff + uuf2

        print("|u-uf|^2 =", uuf2, "\t\tperbead=", suuf / bcount)
        print("|w*f|^2  =", ff, "\t\t|u|^2 =", u2)
        print("L = |u-uf|^2 + lambda*|w*f|^2 = ", L)

        relrec.append((L, uuf2, ff))

        np.savetxt(relrecname, relrec)

    def regularize(self, M, stepper=0.33, REG_SOLVER_PRECISION=1e-18, i_max=100, rel_conv_crit=0.01, alpha=1.0,
                   method="huber", relrecname=None):
        self.I = ssp.lil_matrix((self.vbead.shape[0] * 3, self.vbead.shape[0] * 3))
        self.I.setdiag(np.repeat(self.vbead, 3))

        self.M = M

        # if the shape tensors are not valid, calculate them
        if self.M.Phi_valid is False:
            self.M._computePhi()

        # if the connections are not valid, calculate them
        if self.M.connections_valid is False:
            self.M._computeConnections()

        self.localweight = np.ones(M.N_c)

        # update the forces on each tetrahedron and the global stiffness tensor
        print("going to update glo f and K")
        M._updateGloFAndK()

        # log and store values (if a target file was provided)
        if relrecname is not None:
            relrec = []
            self._recordRegularizationStatus(relrecname, M, relrec)

        print("check before relax !")
        # start the iteration
        for i in range(i_max):
            # compute the weight matrix
            if method != "normal":
                self.updateLocalWeigth(M, method)

            # compute A and b for the linear equation that solves the regularisation problem
            self._computeRegularizationAAndb(M, alpha)

            # get and apply the displacements that solve the regularisation term
            uu = self._solve_regularization_CG(M, stepper, REG_SOLVER_PRECISION)

            # update the forces on each tetrahedron and the global stiffness tensor
            M._updateGloFAndK()

            print("Round", i + 1, " |du|=", uu)

            # log and store values (if a target file was provided)
            if relrecname is not None:
                self._recordRegularizationStatus(relrecname, M, relrec)

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

    def _solve_regularization_CG(self, M, stepper=0.33, REG_SOLVER_PRECISION=1e-18):
        """
        Solve the displacements from the current stiffness tensor using conjugate gradient.
        """

        # solve the conjugate gradient which solves the equation A x = b for x
        # where A is (I - KAK) (K: stiffness matrix, A: weight matrix) and b is (u_meas - u - KAf)
        uu = cg(self.A, self.b.flatten(), maxiter=25 * int(pow(M.N_c, 0.33333) + 0.5),
                tol=M.N_c * REG_SOLVER_PRECISION).reshape((M.N_c, 3))

        # add the new displacements to the stored displacements
        self.M.U += uu * stepper
        # sum the applied displacements
        du = np.sum(uu ** 2) * stepper * stepper

        # return the total applied displacement
        return np.sqrt(du / M.N_c)

    def storeUfound(self, Uname, Sname):
        np.savetxt(Uname, self.U_found)
        np.savetxt(Sname, self.S_0)

    def storeRfound(self, Rname):
        np.savetxt(Rname, self.R_found)

    def storeLocalweights(self, wname):
        np.savetxt(wname, self.localweight)
