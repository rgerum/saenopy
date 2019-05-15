import os
import time

import numpy as np
from numba import jit, double

from .buildBeams import buildBeams
from .buildEpsilon import buildEpsilon
from .multigridHelper import makeBoxmeshCoords, makeBoxmeshTets, setActiveFields

pairs = []
for t1 in range(4):
    for t2 in range(4):
        pairs.append([t1, t2])
pairs = np.array(pairs).astype(int)
#p1 = pairs[:, 0]
#p2 = pairs[:, 1]
#print(repr(p1))
#print(repr(p2))
#exit()

@jit(nopython=True)
def mulK_static_jit(f, u, connections, K_glo_conn):
    f[:] = 0
    # iterate over all connected pairs (contains only connections from variable vertices)
    for i in range(connections.shape[0]):
        c1, c2 = connections[i]
        # the force is the stiffness matrix times the displacement
        f[c1] += K_glo_conn[i] @ u[c2]

from numba import types
from numba.extending import overload

#@overload(np.einsum)
#def einsum(string, array):
#   if isinstance(string, types.string) and isinstance(array, types.double[:]):
#       n = len(seq)
#       def len_impl(seq):
#           return n
#       return len_impl

@jit(double[:, :](double[:,:], double[:, :]), nopython=True)
def dot(a, b):
    c = np.zeros((a.shape[0], b.shape[1]), dtype=np.double)
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(a.shape[1]):
                c[i, j] += a[i, k] * b[k, j]
    return c

@jit(double[:](double[:,:]), nopython=True)
def abs0(a):
    c = np.zeros((a.shape[1]), dtype=np.double)
    for i in range(a.shape[0]):
        sum_ = 0
        for k in range(a.shape[1]):
            sum_ += a[i, k]*a[i, k]
        c[i] = np.sqrt(sum_)
    return c

@jit(double(double[:], double[:]), nopython=True)
def imul(a, b):
    c = 0
    for i in range(a.shape[0]):
        c += a[i]*b[i]
    return c

def det(a):
    xx = a[0, 0]
    xy = a[0, 1]
    xz = a[0, 2]
    yx = a[1, 0]
    yy = a[1, 1]
    yz = a[1, 2]
    zx = a[2, 0]
    zy = a[2, 1]
    zz = a[2, 2]
    return xx * yy * zz\
            +xy * yz * zx\
            +xz * zy * yx\
            -xx * yz * zy\
            -yy * xz * zx\
            -zz * xy * yx

def invert(a):
    deter = det(a)
    xx = a[0, 0]
    xy = a[0, 1]
    xz = a[0, 2]
    yx = a[1, 0]
    yy = a[1, 1]
    yz = a[1, 2]
    zx = a[2, 0]
    zy = a[2, 1]
    zz = a[2, 2]
    return np.array([
    [(yy * zz - yz * zy) / deter,
    (xz * zy - xy * zz) / deter,
    (xy * yz - yy * xz) / deter],
    [(yz * zx - yx * zz) / deter,
    (xx * zz - xz * zx) / deter,
    (xz * yx - xx * yz) / deter],
    [(yx * zy - yy * zx) / deter,
    (xy * zx - xx * zy) / deter,
    (xx * yy - xy * yx) / deter]])


path = None
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

        self.setMeshCoords(makeBoxmeshCoords(dx, nx, rin, mulout))

        self.setMeshTets(makeBoxmeshTets(nx, self.currentgrain))

        self.var = setActiveFields(nx, self.currentgrain, True)

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

        self.setMeshCoords(data.astype(np.float64))

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

        self.setMeshTets(data)

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
        data = np.loadtxt(dbcondsname)
        assert data.shape[1] == 4, "the boundary conditions need 4 columns"
        assert data.shape[0] == self.N_c, "the boundary conditions need to have the same count as the number of vertices"
        print("%s read (%d x %d entries)" % (dbcondsname, data.shape[0], data.shape[1]))

        # the last column is a bool whether the vertex is fixed or not
        self.var = data[:, 3] > 0.5
        # if it is fixed, the other coordinates define the displacement
        self.U[~self.var] = data[~self.var, :3]
        # if it is fixed, the given vector is the force on the vertex
        self.f_ext[self.var] = data[self.var, :3]

        # update the connections (as they only contain non-fixed vertices)
        self.computeConnections()

    def loadConfiguration(self, Uname):
        """
        Load the displacements for the vertices. The file has to have 3 columns for the displacement in XYZ and one
        line for each vertex.
        """
        data = np.loadtxt(Uname)
        assert data.shape[1] == 3, "the displacement file needs to have 3 columnds"
        assert data.shape[0] == self.N_c, "there needs to be a displacement for each vertex"
        print("%s read (%d entries)" % (Uname, data.shape[0]))

        #data = np.random.rand(data.shape[0], data.shape[1])-0.5
        #data *= 0#.0001
        #np.savetxt(Uname, data)

        # store the displacement
        self.U[:, :] = data

    def setMeshCoords(self, data):
        # store the loaded vertex coordinates
        self.R = data

        # store the number of vertices
        self.N_c = data.shape[0]

        # initialize 0 displacement for each vertex
        self.U = np.zeros((self.N_c, 3))

        # start with every vertex being variable (non-fixed)
        self.var = np.ones(self.N_c, dtype=np.int8) == 1  # type bool!

        # initialize global and external forces
        self.f_glo = np.zeros((self.N_c, 3))
        self.f_ext = np.zeros((self.N_c, 3))

        # initialize the list of the global stiffness
        self.K_glo = np.zeros((self.N_c, self.N_c, 3, 3))

    def setMeshTets(self, data):
        # the loaded data are the vertex indices but they start with 1 instead of 0 therefore "-1"
        self.T = data - 1

        # the number of tetrahedrons
        self.N_T = data.shape[0]

        # Phi is a 4x3 tensor for every tetrahedron
        self.Phi = np.zeros((self.N_T, 4, 3))

        # initialize the volume and energy of each tetrahedron
        self.V = np.zeros(self.N_T)
        self.E = np.zeros(self.N_T)

        self.total = np.zeros((self.N_T, 4, 4, 3, 3))

    def computeBeams(self, N):
        beams = buildBeams(N)
        self.setBeams(beams)
        print(beams.shape[0], "beams were generated")

    def setBeams(self, beams):
        self.s = beams
        self.N_b = beams.shape[0]

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
        self.connections = np.array(list(connections)).astype(int)

        # initialize the stiffness matrix premultiplied with the connections
        self.K_glo_conn = np.zeros((self.connections.shape[0], 3, 3))

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

    def computeEpsilon(self, k1=None, ds0=None, s1=None, ds1=None):

        if k1 is None:
            k1 = self.CFG["K_0"]

        if ds0 is None:
            ds0 = self.CFG["D_0"]

        if s1 is None:
            s1 = self.CFG["L_S"]

        if ds1 is None:
            ds1 = self.CFG["D_S"]

        self.epsilon, self.epsbar, self.epsbarbar, self.lookUpEpsilon, self.e0 = buildEpsilon(k1, ds0, s1, ds1, self.CFG)

        self.dlmin = -1.0
        self.dlmax = self.CFG["EPSMAX"]
        self.dlstep = self.CFG["EPSSTEP"]

    def updateGloFAndK(self):
        t_start = time.time()

        s_bar, s_star = self.get_star_and_bar()

        epsilon_b, dEdsbar, dEdsbarbar = self.zzEpsilons3(s_bar, self.lookUpEpsilon, self.V)

        self.update_energy(epsilon_b)

        self.update_f_glo(s_star, s_bar, dEdsbar)

        self.update_K_glo(s_star, s_bar, dEdsbar, dEdsbarbar)

        print("updateGloFAndK time", time.time()-t_start, "s")

    def update_f_glo(self, s_star, s_bar, dEdsbar):
        # f_tmi = s*_tmb * s'_tib * dEds'_tb  (t in [0, N_T], i in {x,y,z}, m in {1,2,3,4}, b in [0, N_b])
        f = np.einsum("tmb,tib,tb->tmi", s_star, s_bar, dEdsbar)

        self.reshape_f3(self.T, f, self.f_glo)

    @staticmethod
    @jit(nopython=True)
    def reshape_f3(T, f, f_glo):
        # f_vi = sum(f_tmi over all corner points of each tetrahedron)
        # (t in [0, N_T], i in {x,y,z}, m in {1,2,3,4}, v in [0, N_c])
        f_glo[:] = 0
        for tt in range(T.shape[0]):
            tet = T[tt]
            f_glo[tet] += f[tt]

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

    def get_star_and_bar(self):
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
        # s*_tmb = Phi_tmj * s_jb  (t in [0, N_T], i,j in {x,y,z}, b in [0, N_b])
        s_star = self.Phi @ self.s.T

        return s_bar, s_star

    @staticmethod
    @jit(nopython=True, cache=True)
    def zzEpsilons3(s_bar, lookUpEpsilon, V):
        N_b = s_bar.shape[-1]

        # test if one vertex of the tetrahedron is variable
        # only count the energy if not the whole tetrahedron is fixed
        # countEnergy = np.any(var[T], axis=1)

        def e0(x):
            epsilon_b, epsbar_b, epsbarbar_b = lookUpEpsilon(x.flatten())
            return epsilon_b.reshape(*x.shape), epsbar_b.reshape(*x.shape), epsbarbar_b.reshape(*x.shape)

        # the "deformation" amount # p 54 equ 2 part in the parentheses
        # s_tb = |s'_tib|  (t in [0, N_T], i in {x,y,z}, b in [0, N_b])
        # s = np.linalg.norm(s_bar, axis=1)
        s = np.sqrt(np.sum(s_bar ** 2, axis=1))

        # evaluate the material function (and its derivatives) at s - 1
        # eps_tb
        epsilon_b, epsbar_b, epsbarbar_b = e0(s - 1)

        #                eps'_tb    1
        # dEdsbar_tb = - ------- * --- * V_t
        #                 s_tb     N_b
        dEdsbar = - (epsbar_b / s) / N_b * np.expand_dims(V, axis=1)

        #                  s_tb * eps''_tb - eps'_tb     1
        # dEdsbarbar_tb = --------------------------- * --- * V_t
        #                         s_tb**3               N_b
        dEdsbarbar = ((s * epsbarbar_b - epsbar_b) / (s ** 3)) / N_b * np.expand_dims(V, axis=1)

        return epsilon_b, dEdsbar, dEdsbarbar

    @staticmethod
    def starbar(s_star, s_bar, dEdsbar, dEdsbarbar):
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

        return np.einsum("tmrb,tilb->tmril", sstarsstar, s_bar_s_bar)

    def update_K_glo(self, s_star, s_bar, dEdsbar, dEdsbarbar):
        self.reshape_stiffnes2(self.T, self.starbar(s_star, s_bar, dEdsbar, dEdsbarbar), self.K_glo)
        self.K_glo_conn[:] = self.K_glo[self.connections[:, 0], self.connections[:, 1], :, :]

    @staticmethod
    @jit(nopython=True)
    def reshape_stiffnes2(T, total, K_glo):
        K_glo[:] = 0
        for tt in range(T.shape[0]):
            tet = T[tt]
            for t1 in range(4):
                c1 = tet[t1]
                for t2 in range(4):
                    c2 = tet[t2]
                    K_glo[c1, c2] += total[tt, t1, t2]

    def relax(self):
        stepper = self.CFG["REL_SOLVER_STEP"]

        i_max = self.CFG["REL_ITERATIONS"]

        outdir = self.CFG["DATAOUT"]
        relrecname = os.path.join(outdir, self.CFG["REL_RELREC"])

        relrec = []

        self.updateGloFAndK()

        relrec.append([np.mean(self.K_glo), self.E_glo, np.sum(self.U[self.var]**2)])

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

            sum_u = np.sum(self.U[self.var]**2)

            # print and store status
            print("Newton ", i, ": du=", du, "  Energy=", self.E_glo, "  Residuum=", ff, "sum_u=", sum_u)
            relrec.append([np.mean(self.K_glo), self.E_glo, sum_u])
            np.savetxt(relrecname, relrec)

            # if we have passed 6 iterations calculate average and std
            if i > 6:
                # calculate the average energy over the last 6 iterations
                last_Es = np.array(relrec)[:-6:-1, 1]
                Emean = np.mean(last_Es)
                Estd = np.std(last_Es)/np.sqrt(5)  # the original formula just had /N instead of /sqrt(N)

                # if the iterations converge, stop the iteration
                if Estd / Emean < self.CFG["REL_CONV_CRIT"]:
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
        kk2 = np.zeros((uu[self.var].shape[0], 3))
        Ap = np.zeros((self.N_c, 3))

        # if it is not 0 (always has to be positive)
        if normb > 0:
            # calculate the force for the given displacements (we start with 0 displacements)
            # TODO are the fixed displacements from the boundary conditions somehow included here? Probably in the stiffness tensor K
            self.mulK(kk, uu)

            # the difference between the desired force deviations and the current force deviations
            rr = ff - kk

            # and store it also in pp? TODO ?
            pp = rr

            # calculate the total force deviation "amplitude"
            resid = np.sum(pp * pp)

            # iterate maxiter iterations
            for i in range(1, maxiter + 1):
                # calculate the current forces
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
            # sum the applied displacements TODO but why stepper**2?
            du = np.sum(uu[self.var]**2) * stepper * stepper

            print(i, ":", du, np.sum(self.U), stepper)

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

    def mulK(self, f, u):
        """
        Multiply the displacement u with the global stiffness tensor K. Or in other words, calculate the forces on all
        vertices.
        """
        f[:] = 0

        c1 = self.connections[:, 0]
        c2 = self.connections[:, 1]
        np.add.at(f, c1, np.einsum("nij,nj->ni", self.K_glo_conn, u[c2]))

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




if 0:
    import numba
    from numba import jitclass, types, jit
    from numba import int32, int64, float32, double, bool_    # import the types

    spec = [
        ('R', double[:, :]),
        ('T', int64[:, :]),
        ('E', double[:]),
        ('V', double[:]),
        ('var', bool_[:]),
        ('Phi', double[:, :, :]),
        ('U', double[:, :]),
        ('f_glo', double[:, :]),
        ('f_ext', double[:, :]),
        ('K_glo', double[:, :, :, :]),
        ('Laplace', double[:]),
        ('E_glo', double),
        ('connections', int64[:, :]),
        ('N_T', int64),
        ('N_c', int64),
        ('s', double[:, :]),
        ('N_b', int64),
        ('epsilon', double[:]),
        ('epsbar', double[:]),
        ('epsbarbar', double[:]),
        ('epsmin', double),
        ('epsmax', double),
        ('epsstep', double),
        ('stepper', double),

    ]

    class FiniteBodyForces:
        def __init__(self, CFG):
            self.CFG = CFG
            self.var = np.zeros(5, dtype=bool)
            self.core = FiniteBodyForcesCore()

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

            self.core.setMeshCoords(data.astype(np.float64))

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

            self.core.setMeshTets(data)

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
            data = np.loadtxt(dbcondsname)
            assert data.shape[1] == 4, "the boundary conditions need 4 columns"
            assert data.shape[0] == self.core.N_c, "the boundary conditions need to have the same count as the number of vertices"
            print("%s read (%d x %d entries)" % (dbcondsname, data.shape[0], data.shape[1]))

            # the last column is a bool whether the vertex is fixed or not
            self.core.var = data[:, 3] > 0.5
            # if it is fixed, the other coordinates define the displacement
            self.core.U[~self.core.var] = data[~self.core.var, :3]
            # if it is fixed, the given vector is the force on the vertex
            self.core.f_ext[self.core.var] = data[self.core.var, :3]

            # update the connections (as they only contain non-fixed vertices)
            self.computeConnections()

        def loadConfiguration(self, Uname):
            """
            Load the displacements for the vertices. The file has to have 3 columns for the displacement in XYZ and one
            line for each vertex.
            """
            data = np.loadtxt(Uname)
            assert data.shape[1] == 3, "the displacement file needs to have 3 columnds"
            assert data.shape[0] == self.core.N_c, "there needs to be a displacement for each vertex"
            print("%s read (%d entries)" % (Uname, data.shape[0]))

            # store the displacement
            self.core.U[:, :] = data

        def computeBeams(self, N):
            beams = buildBeams(N)
            self.core.setBeams(beams)
            print(beams.shape[0], "beams were generated")

        def computeConnections(self):
            # initialize the connections as a set (to prevent double entries)
            connections = set()

            # iterate over all tetrahedrons
            for tet in self.core.T:
                # over all corners
                for t1 in range(4):
                    c1 = tet[t1]

                    # only for non fixed vertices
                    if not self.core.var[c1]:
                        continue

                    for t2 in range(4):
                        # get two vertices of the tetrahedron
                        c2 = tet[t2]

                        # add the connection to the set
                        connections.add((c1, c2))

            # convert the list of sets to an array N_connections x 2
            self.core.connections = np.array(list(connections)).astype(np.int64)

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

            for c in range(self.core.N_c):
                Rrec.append(self.core.R[c])
                Urec.append(self.core.U[c])

            np.savetxt(Rname, Rrec)
            print(Rname, "stored.")
            np.savetxt(Uname, Urec)
            print(Uname, "stored.")

        def storeF(self, Fname):
            Frec = []

            for c in range(self.core.N_c):
                Frec.append(self.core.f_glo[c])

            np.savetxt(Fname, Frec)
            print(Fname, "stored.")

        def storeFden(self, Fdenname):
            Vr = np.zeros(self.core.N_c)

            for tt in range(self.core.N_T):
                for t in range(4):
                    Vr[self.core.T[tt][t]] += self.core.V[tt] * 0.25

            Frec = []
            for c in range(self.core.N_c):
                Frec.apppend(self.core.f_glo[c] / Vr[c])

            np.savetxt(Fdenname, Frec)
            print(Fdenname, "stored.")

        def storeEandV(self, Rname, EVname):
            Rrec = []
            EVrec = []

            for t in range(self.core.N_T):
                N = np.mean(np.array([self.core.R[self.core.T[t][i]] for i in range(4)]), axis=0)

                Rrec.append(N)

                EVrec.append([self.core.E[t], self.core.V[t]])

            np.savetxt(Rname, Rrec)
            print(Rname, "stored.")

            np.savetxt(EVname, EVrec)
            print(EVname, "stored.")

        def computeEpsilon(self, k1=None, ds0=None, s1=None, ds1=None):

            if k1 is None:
                k1 = self.CFG["K_0"]

            if ds0 is None:
                ds0 = self.CFG["D_0"]

            if s1 is None:
                s1 = self.CFG["L_S"]

            if ds1 is None:
                ds1 = self.CFG["D_S"]

            self.core.epsilon, self.core.epsbar, self.core.epsbarbar, self.e0 = buildEpsilon(k1, ds0, s1, ds1, self.CFG)

            #self.core.e0 = self.e0
            self.core.epsmin = -1.0
            self.core.epsmax = self.CFG["EPSMAX"]
            self.core.epsstep = self.CFG["EPSSTEP"]

        def computePhi(self):
            return self.core.computePhi()

        def updateGloFAndK(self):
            return self.core.updateGloFAndK()

        def relax(self):
            # the stepper
            self.core.stepper = self.CFG["REL_SOLVER_STEP"]
            start = time.time()
            self.core.relax(self.CFG["REL_ITERATIONS"], self.CFG["REL_CONV_CRIT"])
            # print the elapsed time
            finish = time.time()
            print("| time for relaxation was", finish - start)


    @jitclass(spec)
    class FiniteBodyForcesCore:
        def __init__(self):
            pass

        def e0(self, deltal):
            # we now have to pass this though the non-linearity function w (material model)
            # this function has been discretized and we interpolate between these discretisation steps

            # the discretisation step
            li = np.floor((deltal - self.epsmin) / self.epsstep)
            # the part between the two steps
            dli = (deltal - self.epsmin) / self.epsstep - li

            # if we are at the border of the discretisation, we stick to the end
            max_index = li > ((self.epsmax - self.epsmin) / self.epsstep) - 2
            li[max_index] = int(((self.epsmax - self.epsmin) / self.epsstep) - 2)
            dli[max_index] = 0

            # convert now to int after fixing the maximum
            li = li.astype(np.int64)

            # interpolate between the two discretisation steps
            epsilon_b = (1 - dli) * self.epsilon[li] + dli * self.epsilon[li + 1]
            epsbar_b = (1 - dli) * self.epsbar[li] + dli * self.epsbar[li + 1]
            epsbarbar_b = (1 - dli) * self.epsbarbar[li] + dli * self.epsbarbar[li + 1]

            return epsilon_b, epsbar_b, epsbarbar_b

        def setMeshCoords(self, data):
            # store the loaded vertex coordinates
            self.R = data

            # store the number of vertices
            self.N_c = data.shape[0]

            # initialize 0 displacement for each vertex
            self.U = np.zeros((self.N_c, 3))

            # start with every vertex being variable (non-fixed)
            self.var = np.ones(self.N_c, dtype=np.int8) == 1  # type bool!

            # initialize global and external forces
            self.f_glo = np.zeros((self.N_c, 3))
            self.f_ext = np.zeros((self.N_c, 3))

            # initialize the list of the global stiffness
            self.K_glo = np.zeros((self.N_c, self.N_c, 3, 3))

        def setMeshTets(self, data):
            # the loaded data are the vertex indices but they start with 1 instead of 0 therefore "-1"
            self.T = data - 1

            # the number of tetrahedrons
            self.N_T = data.shape[0]

            # Phi is a 4x3 tensor for every tetrahedron
            self.Phi = np.zeros((self.N_T, 4, 3))

            # initialize the volumne and energy of each tetrahedron
            self.V = np.zeros(self.N_T)
            self.E = np.zeros(self.N_T)

        def setBeams(self, beams):
            self.s = beams
            self.N_b = beams.shape[0]

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

            #for t, tet in enumerate(self.T):
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

        """
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
        """



        def updateGloFAndK(self):
            # reset the global energy
            self.E_glo = 0.0

            # initialize the list of the global stiffness
            self.K_glo[:] = 0

            # reset the global forces acting on the tetrahedrons
            self.f_glo[:] = 0

            # iterate over all tetrahedrons
            for tt in range(self.N_T):
                # print status every 100 iterations
                if tt % 100 == 0:
                    print("Updating f and K", (np.floor((tt / self.N_T) * 1000) + 0.0) / 10.0)#, "                ", end="\r")

                # test if one vertex of the tetrahedron is variable
                # only count the energy if not the whole tetrahedron is fixed
                countEnergy = np.any(self.var[self.T[tt]])

                # get the displacements of all corners of the tetrahedron (3x4)
                u_T = self.U[self.T[tt]].T

                # the force is the displacement multiplied with the shape tensor plus 1 one the diagonal (p 49)
                F = u_T @ self.Phi[tt] + np.eye(3)

                # iterate over the beams
                # for the elastic strain energy we would need to integrate over the whole solid angle Gamma, but to make this
                # numerically accessible, we iterate over a finite set of directions (=beams) (c.f. page 53)

                # multiply the force tensor with the beam
                s_bar = F @ self.s.T  # 3xN_b
                # and the shape tensor with the beam
                s_star = self.Phi[tt] @ self.s.T  # 4xN_b

                # the "deformation" amount # p 54 equ 2 part in the parentheses
                #s = np.linalg.norm(s_bar, axis=0)
                s = np.sqrt(np.sum(s_bar**2, axis=0))
                #s = s_bar[0, :]

                # evaluate the material function (and its derivatives) at s - 1
                epsilon_b, epsbar_b, epsbarbar_b = self.e0(s - 1)

                # sum the energy of this tetrahedron
                self.E[tt] = np.mean(epsilon_b) * self.V[tt]

                # only count the energy of the tetrahedron to the global energy if the tetrahedron has at least one
                # variable vertex
                if countEnergy:
                    self.E_glo += self.E[tt]

                dEdsbar = - (epsbar_b / s) / self.N_b * self.V[tt]

                dEdsbarbar = ((s * epsbarbar_b - epsbar_b) / (s**3)) / self.N_b * self.V[tt]

                # calculate its contribution to the global force on this tetrahedron
                # p 54 last equation (in the thesis V is missing in the equation)
                self.f_glo[self.T[tt]] += s_star @ (s_bar * dEdsbar).T

                if 1:
                    for t1 in range(4):
                        for t2 in range(4):
                            # dimensions 4x4xN_b
                            sstarsstar = s_star[t1, :] * s_star[t2, :]

                            # dimensions 4x4x3x3xN_b and summation over N_b
                            test = np.expand_dims(s_bar, axis=0) * np.expand_dims(s_bar, axis=1)
                            K_glo = np.sum(sstarsstar * 0.5 * (
                                    dEdsbarbar * np.expand_dims(s_bar, axis=0) * np.expand_dims(s_bar, axis=1) - np.expand_dims(np.eye(3), axis=2) * np.sum(
                                dEdsbar)), axis=-1)

                            # and for each corner pair 4x4 we assign the 3x3 stiffness matrix
                            self.K_glo[self.T[tt][t1], self.T[tt][t2]] += K_glo
                else:
                    # dimensions 4x4xN_b
                    sstarsstar = s_star[None, :, :] * s_star[:, None, :]

                    # dimensions 4x4x3x3xN_b and summation over N_b
                    K_glo = np.sum(sstarsstar[:, :, None, None, :] * 0.5 * (
                            dEdsbarbar * s_bar[None, :, :] * s_bar[:, None, :] - np.eye(3)[:, :, None] * np.sum(dEdsbar)), axis=-1)

                    # and for each corner pair 4x4 we assign the 3x3 stiffness matrix
                    self.K_glo[self.T[tt][:, None], self.T[tt][None, :]] += K_glo

        def relax(self, i_max, REL_CONV_CRIT):
            #i_max = self.CFG["REL_ITERATIONS"]

            #outdir = self.CFG["DATAOUT"]
            #relrecname = os.path.join(outdir, self.CFG["REL_RELREC"])

            relrec = []

            # start the iteration
            for i in range(i_max):
                # move the displacements in the direction of the forces one step
                # but while moving the stiffness tensor is kept constant
                du = self.solve_CG()

                # update the forces on each tetrahedron and the global stiffness tensor
                self.updateGloFAndK()

                # sum all squared forces of non fixed vertices
                ff = np.sum(self.f_glo[self.var] ** 2)

                # print and store status
                print("Newton ", i, ": du=", du, "  Energy=", self.E_glo, "  Residuum=", ff)
                relrec.append([du, self.E_glo, ff])
                #np.savetxt(relrecname, relrec)

                # if we have passed 6 iterations calculate average and std
                if i > 6:
                    # calculate the average energy over the last 6 iterations
                    last_Es = np.array(relrec)[:-6:-1, 1]
                    Emean = np.mean(last_Es)
                    Estd = np.std(last_Es) / np.sqrt(5)  # the original formula just had /N instead of /sqrt(N)

                    # if the iterations converge, stop the iteration
                    #if (Estd / Emean) < REL_CONV_CRIT:
                    if Estd < REL_CONV_CRIT * Emean:
                        break

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

                if 0:
                    uu = self.CG_static_iterations(maxiter, tol, normb, pp, uu, rr, resid, self.connections, self.K_glo)
                else:
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
                        if i % 100 == 0:
                            print(i, ":", resid, "alpha=", alpha)#, end="\r")

                # now we want to apply the obtained guessed displacements to the vertex displacements "permanently"
                # therefore we add the guessed displacements times a stepper parameter to the vertex displacements

                # sum the total applied displacements
                du = 0

                # iterate over all non fixed vertices
                for c in range(self.N_c):
                    if self.var[c]:
                        # get the displacement
                        dU = uu[c]

                        # apply it
                        # TODO possible optimistation multiply the whole uu array
                        self.U[c] += dU * self.stepper

                        # sum the applied displacements. TODO but why stepper**2?
                        du += np.sum(dU**2) * self.stepper * self.stepper

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

        def mulK(self, u):
            """
            Multiply the displacement u with the global stiffness tensor K. Or in other words, calculate the forces on all
            vertices.
            """
            #return self.mulK_static(u, self.connections, self.K_glo)

            # start with an empty force array
            f = np.zeros((self.K_glo.shape[0], 3))

            # iterate over all connected pairs (containes only connections from variable vertices)
            #for c1, c2 in self.connections:
            for _ in range(self.connections.shape[0]):
                c1, c2 = self.connections[_]
                # get the offset of the partner
                uu = u[c2]
                # get the stiffness matrix
                A = self.K_glo[c1, c2]

                # the force is the stiffness matrix times the displacement
                f[c1] += A @ uu

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

