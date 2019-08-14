import numpy as np
import time
import os

from numba import jit, njit
import scipy.sparse as ssp
from .conjugateGradient import cg


class timeit:

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        print("Start", self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("End", self.name, "%.3fs" % (time.time()-self.start))

class IA:
    def __init__(self, A, I):
        self.A = A
        self.I = I

    def __matmul__(self, other):
        return self.I * other + self.A @ other

class VirtualBeads:
    def __init__(self, CFG, ssX=None, ssY=None, ssZ=None, ddX=None, ddY=None, ddZ=None):
        self.CFG = CFG
        self.sX = ssX
        self.sY = ssY
        self.sZ = ssZ
        self.dX = ddX
        self.dY = ddY
        self.dZ = ddZ

    def allBeads(self, M):
        thresh = self.CFG["VB_SX"] * self.dX + 2.0 * self.CFG["DRIFT_STEP"]

        self.vbead = np.zeros(M.N_c, dtype=bool)

        Scale = np.eye(3) * np.array([self.dX, self.dY, self.dZ])

        scaledShift = np.array([self.sX / 2, self.sY / 2, self.sZ / 2])

        Trans = Scale

        for t in range(M.N_c):
            R = Trans @ M.R[t] + scaledShift

            if R[0] > thresh and R[0] < (self.sX - thresh) and \
               R[1] > thresh and R[1] < (self.sY - thresh) and \
               R[2] > thresh and R[2] < (self.sZ - thresh):
                self.vbead[t] = True

    def computeOutOfStack(self, M):
        thresh = self.CFG["VB_SX"] * self.dX + 2.0 * self.CFG["DRIFT_STEP"]

        self.outofstack = np.array(M.N_c, dtype=bool)

        Scale = np.eye(3) * np.array([self.dX, self.dY, self.dZ])

        scaledShift = np.array([self.sX / 2, self.sY / 2, self.sZ / 2])

        Trans = Scale

        for t in range(M.N_c):
            R = Trans @ M.R[t] + scaledShift

            if R[0] > thresh and R[0] < (self.sX - thresh) and \
               R[1] > thresh and R[1] < (self.sY - thresh) and \
               R[2] > thresh and R[2] < (self.sZ - thresh):
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

            index = (Fvalues > (k * Fmedian))*M.var
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
        pass

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
        index = np.abs(self.U_found) > 0.01*self.CFG["VOXELSIZEX"]
        u_median = np.median(self.U_found[index], axis=0)
        print("Median displacement calculated to", u_median)

        self.U_found -= u_median

    def findDriftCoarse(self):
        pass

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

        u2 = np.sum(M.U[M.var]**2)

        f = np.zeros((M.N_c, 3))
        f[M.var] = M.f_glo[M.var]

        ff = np.sum(np.sum(f**2, axis=1)*self.localweight*M.var)

        L = alpha*ff + uuf2

        print("|u-uf|^2 =", uuf2, "\t\tperbead=", suuf/bcount)
        print("|w*f|^2  =", ff, "\t\t|u|^2 =", u2)
        print("L = |u-uf|^2 + lambda*|w*f|^2 = ", L)

        relrec.append((L, uuf2, ff))

        np.savetxt(relrecname, relrec)

    def regularize(self, M, stepper=0.33, REG_SOLVER_PRECISION=1e-18, i_max=100, rel_conv_crit=0.01, alpha=1.0, method="huber", relrecname=None):
        self.I = ssp.lil_matrix((self.vbead.shape[0]*3, self.vbead.shape[0]*3))
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

            print("Round", i+1, " |du|=", uu)

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
        uu = cg(self.A, self.b.flatten(), maxiter=25*int(pow(M.N_c, 0.33333)+0.5), tol=M.N_c*REG_SOLVER_PRECISION).reshape((M.N_c, 3))

        # add the new displacements to the stored displacements
        self.M.U += uu * stepper
        # sum the applied displacements
        du = np.sum(uu ** 2) * stepper * stepper

        # return the total applied displacement
        return np.sqrt(du/M.N_c)

    def storeUfound(self, Uname, Sname):
        np.savetxt(Uname, self.U_found)
        np.savetxt(Sname, self.S_0)

    def storeRfound(self, Rname):
        np.savetxt(Rname, self.R_found)

    def storeLocalweights(self, wname):
        np.savetxt(wname, self.localweight)
