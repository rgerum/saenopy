import numpy as np
import time
import os

from numba import jit, njit
import scipy.sparse as ssp

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

    def updateLocalWeigth(self, M):
        k = 1.345
        if self.CFG["ROBUSTMETHOD"] == "bisquare":
            k = 4.685
        if self.CFG["ROBUSTMETHOD"] == "cauchy":
            k = 2.385

        self.localweight = np.ones(M.N_c)

        Fvalues = np.linalg.norm(M.f_glo, axis=1)
        Fmedian = np.median(Fvalues[M.var])

        if self.CFG["ROBUSTMETHOD"] == "singlepoint":
            self.localweight[int(self.CFG["REG_FORCEPOINT"])] = 1.0e-10

        if self.CFG["ROBUSTMETHOD"] == "bisquare":
            index = Fvalues < k * Fmedian
            self.localweight[index * M.var] *= (1 - (Fvalues / k / Fmedian) * (Fvalues / k / Fmedian)) * (
                    1 - (Fvalues / k / Fmedian) * (Fvalues / k / Fmedian))
            self.localweight[~index * M.var] *= 1e-10

        if self.CFG["ROBUSTMETHOD"] == "cauchy":
            if Fmedian > 0:
                self.localweight[M.var] *= 1.0 / (1.0 + np.power((Fvalues / k / Fmedian), 2.0))
            else:
                self.localweight *= 1.0

        if self.CFG["ROBUSTMETHOD"] == "huber":
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

    def computeAAndb(self, M, alpha):
        uselaplace = self.CFG["REGMETHOD"] == "laplace"
        print("uselaplace", uselaplace)
        lagrain = self.CFG["REG_LAPLACEGRAIN"] ** 2
        llambda_z = self.CFG["REG_SIGMAZ"]

        import time
        if 1:
            with timeit("computeAAndb"):
                #with timeit("multiply A"):
                #    with timeit("1"):
                AK = M.K_glo.multiply(np.repeat(self.localweight*alpha, 3)[:, None])
                KA = M.K_glo.multiply(np.repeat(self.localweight * alpha, 3)[None, :])
                self.KAK = M.K_glo @ AK
                self.A = self.I + self.KAK

                self.b = (KA @ M.f_glo.ravel()).reshape(M.f_glo.shape)

                index = M.var * self.vbead
                self.b[index] += self.U_found[index] - M.U[index]
            return

        if 0:
            K1 = M.K_glo[0:3].data.reshape(3, -1)
            l = lwa[[0, 1, 2, 1 + 0, 1 + 1, 1 + 2, 18 + 0, 18 + 1, 18 + 2, 324 + 0, 324 + 1, 324 + 2]]

        if 1:
            #return
            Aj = []
            Asp = []
            with open("outputAj.dat") as fp_Aj:
                with open("outputAsp.dat") as fp_Asp:
                    for l1, l2 in zip(fp_Aj, fp_Asp):
                        Aj.append([int(i) for i in l1.split()])
                        Asp.append(np.array([float(i) for i in l2.split()]).reshape(-1, 3, 3))

            self.A_sp = []
            self.A_j = []
            for i in range(M.N_c):
                self.A_sp.append([])
                self.A_j.append([])
            self.b = np.zeros(M.N_c)

            for i in range(M.N_c):
                if not M.var[i]:
                    continue

                if i % 100 == 0:
                    print("computing A", int(i/M.N_c*100), "%")

                self.A_j[i] = []
                self.A_sp[i] = []

                for j in self.conconnections[i]:
                    A_ijtemp = np.zeros((3, 3))

                    hasentry = False

                    if M.var[j]:
                        if uselaplace:
                            set_intersection = self.oldconconnections[i] & self.oldconconnections[j]
                        else:
                            set_intersection = M.connections[i] & M.connections[j]

                        for k in set_intersection:
                            if not M.var[k]:
                                continue

                            hasentry = True

                            if uselaplace:
                                KL_ik = M.K_glo[i] @ M.Laplace[k]
                                LK_kj = M.Laplace[k] @ M.K_glo[j]

                                if M.K_glo[i].find(k) != M.K_glo[i].end():
                                    KA_ik = M.K_glo[i][k]
                                else:
                                    KA_ik = np.zeros((3, 3))

                                if M.K_glo[k].find(j) != M.K_glo[k].end():
                                    AK_kj = M.K_glo[k][j]
                                else:
                                    AK_kj = np.zeros((3, 3))

                                A_ijtemp += KA_ik*AK_kj*alpha*self.localweight[k]

                                A_ijtemp += KL_ik*LK_kj*alpha*lagrain
                            else:
                                if 1:
                                    KA_ik = M.K_glo[i*3:i*3+3, k*3:k*3+3].reshape(3, 3)
                                    AK_kj = M.K_glo[k*3:k*3+3, j*3:j*3+3].reshape(3, 3)
                                else:
                                    KA_ik = M.K_glo_all[i, k]
                                    AK_kj = M.K_glo_all[k, j]

                                A_ijtemp += KA_ik @ AK_kj * alpha * self.localweight[k]
                                print("i", i, "j", j, "k", k)
                                print("KA_ik", KA_ik)
                                print("KA_kj", AK_kj)
                                print("alpha", alpha)
                                print("localweight[k]", self.localweight[k])

                    if i == j and self.vbead[i]:
                        hasentry = True

                        A_ijtemp[0, 0] += 1.0
                        A_ijtemp[1, 1] += 1.0
                        A_ijtemp[2, 2] += llambda_z

                    if hasentry:

                        self.A_j[i].append(j)
                        self.A_sp[i].append(A_ijtemp)
                    break
                break

            self.b = np.zeros((M.N_c, 3))
            f = M.f_glo.copy()

            if uselaplace:
                lf = M.Laplace @ f
                llf = M.Laplace @ lf

            for i in range(M.N_c):
                if M.var[i]:
                    if i % 100 == 0:
                        print("computing b", int(i/M.N_c*100), "%")

                        if self.vbead[i]:
                            btemp = self.U_found[i]-M.U[i]
                            btemp[2] *= llambda_z
                            self.b[i] += btemp

                        for j in M.connections[i]:
                            if M.var[j]:
                                self.b[i] += M.K_glo[i][j] * f[j] * alpha * self.localweight[j]
                                if uselaplace:
                                    self.b[i] += M.K_glo[i][j] * llf[j] * alpha * lagrain

    def getB(self):
        M = self.M
        llambda_z = 1
        alpha = self.CFG["ALPHA"]

        b3 = np.zeros((M.N_c, 3))
        f = M.f_glo.copy()

        for i in range(M.N_c):
            if M.var[i]:
                if i % 100 == 0:
                    print("computing b", int(i / M.N_c * 100), "%")

                if self.vbead[i]:
                    btemp = self.U_found[i] - M.U[i]
                    btemp[2] *= llambda_z
                    b3[i] += btemp

                for j in M.connections[i]:
                    if M.var[j]:
                        b2[i] += M.K_glo[i*3:i*3+3, j*3:j*3+3] @ f[j] * alpha * self.localweight[j]
        return b

    def mulA(self, u):
        return self.mulA_jit(u, self.M.N_c, self.A_j, self.A_sp)

    @staticmethod
    @jit()
    def mulA_jit(u, N_c, A_j, A_sp):
        #print("Mul A")
        #return (self.A @ u.flatten()).reshape(u.shape)
        f = np.zeros(u.shape)
        for c1 in range(N_c):
            ff = np.zeros(3)

            for j, c2 in enumerate(A_j[c1]):

                uu = u[3*c2:3*c2+3]
                ff += A_sp[c1][j] @ uu

            f[3*c1:3*c1+3] = ff
        return f

    def recordRelaxationStatus(self, M, relrec):
        ff = 0.0
        L = 0.0
        u2 = 0.0
        uuf2 = 0.0
        suuf = 0.0

        lagrain = self.CFG["REG_LAPLACEGRAIN"]**2

        bcount = 0

        alpha = self.CFG["ALPHA"]
        sigz = self.CFG["REG_SIGMAZ"]

        for b in range(M.N_c):
            if M.var[b] and self.vbead[b]:
                btemp = self.U_found[b] - M.U[b]
                btemp[2] *= sigz
                uuf2 += np.sum(btemp**2)
                suuf += np.linalg.norm(btemp)
                bcount += 1

        u2 = np.sum(M.U[M.var]**2)

        f = np.zeros((M.N_c, 3))
        f[M.var] = M.f_glo[M.var]

        if self.CFG["REGMETHOD"] == "laplace":
            lf = M.Laplace @ f

            for b in range(M.N_c):
                if M.var[b]:
                    f[b] += lf[b] * lagrain

        ff = np.sum(np.sum(f**2, axis=1)*self.localweight*M.var)

        L = alpha*ff + uuf2

        print("|u-uf|^2 =", uuf2, "\t\tperbead=", suuf/bcount)
        print("|w*f|^2  =", ff, "\t\t|u|^2 =", u2)
        print("L = |u-uf|^2 + lambda*|w*f|^2 = ", L)

        relrec.append((L, uuf2, ff))

        relrecname = os.path.join(self.CFG["DATAOUT"], self.CFG["REG_RELREC"])
        np.savetxt(relrecname, relrec)

    def relax(self, M):
        llambda_z = self.CFG["REG_SIGMAZ"]

        #self.I = ssp.lil_matrix((3, 3))
        #self.I.setdiag([1, 1, llambda_z])
        #self.I = ssp.kron(np.outer(self.vbead, np.ones(M.N_c)), self.I, format="csc")
        self.I_linear = np.repeat(self.vbead, 3)
        self.I = ssp.lil_matrix((self.vbead.shape[0]*3, self.vbead.shape[0]*3))
        self.I.setdiag(self.I_linear)

        relrec = []

        lagrain = self.CFG["REG_LAPLACEGRAIN"]**2

        alpha = self.CFG["ALPHA"]
        self.M = M

        self.localweight = np.ones(M.N_c)

        def loadK(name):
            return
            data = np.load(name+".npy")
            x = np.load(name+"X.npy")
            y = np.load(name+"Y.npy")
            K = ssp.coo_matrix((data, (x, y)), shape=(M.N_c*3, M.N_c*3))
            f = np.load(name+"F.npy")

            self.M.K_glo = K
            self.M.f_glo = f

        def loadAb(self, name, i=0):
            return
            if i == 0:
                return
            b = np.load(name.replace("Aj", "b")+".npy")

            name = name.replace("Aj", "Asp")
            data = np.load(name+"_data.npy")
            indices = np.load(name+"_indices.npy")
            indptr = np.load(name+"_indptr.npy")
            A = ssp.bsr_matrix((data, indices, indptr), shape=(b.shape[0] * 3, b.shape[0] * 3)).tocsr()

            self.A = A
            self.b = b

        print("going to update glo f and K")
        M._updateGloFAndK()
        loadK("output/K%d" % -1)

        self.recordRelaxationStatus(M, relrec)

        i_max = self.CFG["REG_ITERATIONS"]

        print("check before relax !")

        for i in range(i_max):
            if self.CFG["REGMETHOD"] != "normal":
                self.updateLocalWeigth(M)
                #self.localweight = np.loadtxt("output/lw%d.dat" % i)

            self.computeAAndb(M, alpha)
            #loadAb(self, "output/Aj%d.dat" % i, i)

            uu = self._solve_CG(M)
            #M.U = np.loadtxt("output/U%d.dat" % i)

            M._updateGloFAndK()
            #loadK("output/K%d" % i)

            print("Round", i+1, " |du|=", uu)

            self.recordRelaxationStatus(M, relrec)

            # if we have passed 6 iterations calculate average and std
            if i > 6:
                # calculate the average energy over the last 6 iterations
                last_Ls = np.array(relrec)[:-6:-1, 1]
                Lmean = np.mean(last_Ls)
                Lstd = np.std(last_Ls) / np.sqrt(5)  # the original formula just had /N instead of /sqrt(N)

                # if the iterations converge, stop the iteration
                if Lstd / Lmean < self.CFG["REG_CONV_CRIT"]:
                    break
            #break

        return relrec

    def _solve_CG(self, M):
        """
        Solve the displacements from the current stiffness tensor using conjugate gradient.
        """
        stepper = self.CFG["REG_SOLVER_STEP"]

        # solve the conjugate gradient which solves the equation A x = b for x
        # where A is the stiffness matrix K_glo and b is the vector of the target forces
        self.M = M
        A = IA(self.KAK, self.I_linear)
        #A.__matmul__ = self.mulA
        uu = self.cg(self.A, self.b.flatten(), maxiter=25*int(pow(M.N_c, 0.33333)+0.5), tol=M.N_c*self.CFG["REG_SOLVER_PRECISION"]).reshape((M.N_c, 3))

        # add the new displacements to the stored displacements
        self.M.U += uu * stepper
        # sum the applied displacements
        du = np.sum(uu ** 2) * stepper * stepper

        # return the total applied displacement
        return np.sqrt(du/M.N_c)

    def cg(self, A, b, maxiter=1000, tol=0.00001):
        def norm(x):
            return np.inner(x.flatten(), x.flatten())

        # calculate the total force "amplitude"
        normb = norm(b)

        # if it is not 0 (always has to be positive)
        if normb == 0:
            return 0

        x = np.zeros_like(b)

        # the difference between the desired force deviations and the current force deviations
        r = b - A @ x

        # and store it also in pp
        p = r

        # calculate the total force deviation "amplitude"
        resid = norm(p)

        # iterate maxiter iterations
        for i in range(1, maxiter + 1):
            Ap = A @ p

            alpha = resid / np.sum(p * Ap)

            x = x + alpha * p
            r = r - alpha * Ap

            rsnew = norm(r)

            # check if we are already below the convergence tolerance
            if rsnew < tol * normb:
                break

            beta = rsnew / resid

            # update pp and resid
            p = r + beta * p
            resid = rsnew

            # print status every 100 frames
            if i % 100 == 0:
                print(i, ":", resid, "alpha=", alpha, "du=", np.sum(x ** 2))  # , end="\r")

        return x

    def storeUfound(self, Uname, Sname):
        np.savetxt(Uname, self.U_found)
        np.savetxt(Sname, self.S_0)

    def storeRfound(self, Rname):
        np.savetxt(Rname, self.R_found)

    def storeLocalweights(self, wname):
        np.savetxt(wname, self.localweight)
