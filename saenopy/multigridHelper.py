import numpy as np
import time


def createMesh(count=None, element_width=None, box_width=None):
    if isinstance(box_width, (int, float)):
        box_width = [box_width, box_width, box_width]

    if isinstance(element_width, (int, float)):
        element_width = [element_width, element_width, element_width]

    if isinstance(count, (int, float)):
        count = [count, count, count]

    if element_width is None:
        element_width = np.array(box_width) / np.array(count)
    if box_width is None:
        box_width = np.array(element_width) * np.array(count)
    if count is None:
        count = [int(np.round(box_width[i] / element_width[i])) for i in range(3)]

    # R, T = createBoxMesh(*[np.linspace(-box_width[i]/2, box_width[i]/2, count[i]) for i in range(3)])
    R, T = createBoxMesh(np.linspace(0, box_width[0], count[0]),
                         np.linspace(-box_width[1] / 2, box_width[1] / 2, count[1]),
                         np.linspace(-box_width[2] / 2, box_width[2] / 2, count[2])
                         )
    return R, T


def createSolverBoxMesh(count=None, element_width=None, box_width=None, material=None):
    from saenopy import Solver
    M = Solver()
    if material is not None:
        M.setMaterialModel(material)

    R, T = createMesh(count, element_width, box_width)

    M.setNodes(R)
    M.setTetrahedra(T)

    return M


def createBoxMesh(x, y=None, z=None):
    if y is None:
        y = x
    if z is None:
        z = x
    mesh = np.array(np.meshgrid(x, y, z, indexing="ij")).reshape(3, -1).T
    R = np.zeros(mesh.shape)
    R[:, :] = mesh
    T = makeBoxmeshTets(len(x), len(y), len(z))
    return R, T


def makeBoxmeshCoords(dx, nx, rin, mulout):
    ny = nx
    nz = nx
    dy = dx
    dz = dx

    rout = nx * dx * 0.5

    N_c = nx * nx * nx

    R = []
    for i in range(N_c):
        R.append([0.0, 0.0, 0.0])

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                i = x + nx * y + nx * ny * z

                X = x * dx - (nx - 1) * dx * 0.5
                Y = y * dy - (ny - 1) * dy * 0.5
                Z = z * dz - (nz - 1) * dz * 0.5

                f = max(abs(X), max(abs(Y), abs(Z)))

                mul = max(1.0, ((f - rin) / (rout - rin) + 1.0) * (mulout - 1.0) + 1.0)

                R[i] = [X * mul, Y * mul, Z * mul]

    return R

from numba import njit

@njit()
def makeBoxmeshTets(nx, ny=None, nz=None, grain=1):
    if ny is None:
        ny = nx
    if nz is None:
        nz = nx
    T = []
    
    def index_of(x, y, z):
        return ny * nz * x + nz * y + z

    for x in range(0, nx):
        for y in range(0, ny):
            for z in range(0, nz):
                i = index_of(x, y, z)

                if x > 0 and y > 0 and z > 0:
                    i1 = index_of(x-0, y-0, z-0)
                    i2 = index_of(x-0, y-1, z-0)#(x - 0) + nx * (y - grain) + nx * ny * (z - 0)
                    i3 = index_of(x-1, y-1, z-0)#(x - grain) + nx * (y - grain) + nx * ny * (z - 0)
                    i4 = index_of(x-1, y-0, z-0)#(x - grain) + nx * (y - 0) + nx * ny * (z - 0)
                    i5 = index_of(x-0, y-0, z-1)#(x - 0) + nx * (y - 0) + nx * ny * (z - grain)
                    i6 = index_of(x-1, y-0, z-1)#(x - grain) + nx * (y - 0) + nx * ny * (z - grain)
                    i7 = index_of(x-1, y-1, z-1)#(x - grain) + nx * (y - grain) + nx * ny * (z - grain)
                    i8 = index_of(x-0, y-1, z-1)#(x - 0) + nx * (y - grain) + nx * ny * (z - grain)

                    T.append([i1, i2, i3, i8])

                    T.append([i1, i3, i4, i6])

                    T.append([i1, i5, i8, i6])

                    T.append([i3, i6, i8, i7])

                    T.append([i1, i8, i3, i6])

    return np.array(T, dtype=np.int64)

def getTetrahedraFromHexahedra(hexs):
    T = []
    for h in hexs:
        i1, i2, i3, i4, i5, i6, i7, i8 = h
        T.append([i1, i2, i3, i8])

        T.append([i1, i3, i4, i6])

        T.append([i1, i5, i8, i6])

        T.append([i3, i6, i8, i7])

        T.append([i1, i8, i3, i6])
    return T

def getTetrahedraVolumnes(T, R):
    # define the helper matrix chi
    Chi = np.zeros((4, 3))
    Chi[0, :] = [-1, -1, -1]
    Chi[1, :] = [1, 0, 0]
    Chi[2, :] = [0, 1, 0]
    Chi[3, :] = [0, 0, 1]

    # tetrahedron matrix B (linear map of the undeformed tetrahedron T onto the primitive tetrahedron P)
    B = R[T[:, 1:4]] - R[T[:, 0]][:, None, :]
    B = B.transpose(0, 2, 1)

    # calculate the volume of the tetrahedron
    V = np.abs(np.linalg.det(B)) / 6.0
    return V


def setActiveFields(nx, grain, val):
    ny = nx
    nz = nx
    var = [0] * nx * ny * nz

    for x in range(0, nx, grain):
        for y in range(0, ny, grain):
            for z in range(0, nz, grain):
                i = x + nx * y + nx * ny * z
                var[i] = val

    return val


def getTetrahedraVolumnes(T, R):
    # define the helper matrix chi
    Chi = np.zeros((4, 3))
    Chi[0, :] = [-1, -1, -1]
    Chi[1, :] = [1, 0, 0]
    Chi[2, :] = [0, 1, 0]
    Chi[3, :] = [0, 0, 1]

    # tetrahedron matrix B (linear map of the undeformed tetrahedron T onto the primitive tetrahedron P)
    B = R[T[:, 1:4]] - R[T[:, 0]][:, None, :]
    B = B.transpose(0, 2, 1)

    # calculate the volume of the tetrahedron
    V = np.abs(np.linalg.det(B)) / 6.0
    return V


def subdivideTetrahedra(T, R):
    centroids = np.mean(R[T], axis=1)
    T2 = []
    for index, tet in enumerate(T):
        t1, t2, t3, t4 = tet
        c = R.shape[0] + index
        T2.append([t1, t2, t3, c])
        T2.append([t1, t2, c, t4])
        T2.append([t1, c, t3, t4])
        T2.append([c, t2, t3, t4])
    return np.array(T2), np.concatenate((R, centroids))


def tetsToHex(T, R):
    F, Fi = getFaces(T)
    L, Li = getLinesTetrahedra(T)

    centroids = np.mean(R[T], axis=1)
    face_centers = np.mean(R[F], axis=1)
    line_centers = np.mean(R[L], axis=1)

    H = []
    for index, tet in enumerate(T):
        t1, t2, t3, t4 = tet
        c = R.shape[0] + index
        f123, f124, f134, f234 = R.shape[0] + centroids.shape[0] + Fi[index]
        l12, l13, l14, l23, l24, l34 = R.shape[0] + centroids.shape[0] + face_centers.shape[0] + Li[index]

        H.append([t1, l12, f123, l13, l14, f124, c, f134])
        H.append([t2, l23, f123, l12, l24, f234, c, f124])
        H.append([t3, l13, f123, l23, l34, f134, c, f234])
        H.append([t4, l14, f124, l24, l34, f134, c, f234])

    return np.array(H), np.concatenate((R, centroids, face_centers, line_centers))

def subdivideHexahedra(T, R):
    F, Fi = getFacesHexahedra(T)
    L, Li = getLinesHexahedra(T)

    centroids = np.mean(R[T], axis=1)
    face_centers = np.mean(R[F], axis=1)
    line_centers = np.mean(R[L], axis=1)

    H = []
    for index, tet in enumerate(T):
        t1, t2, t3, t4, t5, t6, t7, t8 = tet
        c = R.shape[0] + index

        f1234, f5678, f1256, f2367, f3478, f1458 = R.shape[0] + centroids.shape[0] + Fi[index]

        l12, l23, l34, l14,\
        l56, l67, l78, l58,\
        l15, l26, l37, l48 = R.shape[0] + centroids.shape[0] + face_centers.shape[0] + Li[index]

        H.append([t1, l12, f1234, l14, l15, f1256, c, f1458])
        H.append([t2, l23, f1234, l12, l26, f2367, c, f1256])
        H.append([t3, l34, f1234, l23, l37, f3478, c, f2367])
        H.append([t4, l14, f1234, l34, l48, f1458, c, f3478])
        H.append([l15, f1256, c, f1458, t5, l56, f5678, l58])
        H.append([l26, f2367, c, f1256, t6, l67, f5678, l56])
        H.append([l37, f3478, c, f2367, t7, l78, f5678, l67])
        H.append([l48, f1458, c, f3478, t8, l58, f5678, l78])

    return np.array(H), np.concatenate((R, centroids, face_centers, line_centers))


def getTetrahedraFromHexahedra(hexs):
    T = []
    for h in hexs:
        i1, i2, i3, i4, i5, i6, i7, i8 = h
        T.append([i1, i2, i3, i8])

        T.append([i1, i3, i4, i6])

        T.append([i1, i5, i8, i6])

        T.append([i3, i6, i8, i7])

        T.append([i1, i8, i3, i6])
    return np.array(T)


def getTetrahedraVolumnes(T, R):
    # define the helper matrix chi
    Chi = np.zeros((4, 3))
    Chi[0, :] = [-1, -1, -1]
    Chi[1, :] = [1, 0, 0]
    Chi[2, :] = [0, 1, 0]
    Chi[3, :] = [0, 0, 1]

    # tetrahedron matrix B (linear map of the undeformed tetrahedron T onto the primitive tetrahedron P)
    B = R[T[:, 1:4]] - R[T[:, 0]][:, None, :]
    B = B.transpose(0, 2, 1)

    # calculate the volume of the tetrahedron
    V = np.abs(np.linalg.det(B)) / 6.0
    return V


def getHexahedraVolumnes(H, R):
    T = getTetrahedraFromHexahedra(H)
    V = getTetrahedraVolumnes(T, R).reshape(H.shape[0], -1)
    return np.sum(V, axis=1)


def getFaces(T):
    faces = []
    faces_of_T = []
    for tet in T:
        t1, t2, t3, t4 = tet
        tet_faces = [sorted([t1, t2, t3]), sorted([t1, t2, t4]), sorted([t1, t3, t4]), sorted([t2, t3, t4])]
        face_indices = []
        for face in tet_faces:
            i = 0
            for i in range(len(faces)):
                if faces[i] == face:
                    break
            else:
                faces.append(face)
                face_indices.append(len(faces) - 1)
                continue
            face_indices.append(i)
        faces_of_T.append(face_indices)
    return np.array(faces), np.array(faces_of_T)


@njit()
def getLinesTetrahedra(T):
    lines = []
    lines_of_T = []
    for i in range(T.shape[0]):
        tet = T[i]
    #for tet in T:
        t1, t2, t3, t4 = tet
        tet_lines = [sorted([t1, t2]), sorted([t1, t3]), sorted([t1, t4]), sorted([t2, t3]), sorted([t2, t4]),
                     sorted([t3, t4])]
        line_indices = []
        for j in range(len(tet_lines)):
            line = tet_lines[j]
        #for line in tet_lines:
            i = 0
            for i in range(len(lines)):
                if lines[i] == line:
                    break
            else:
                lines.append(line)
                line_indices.append(len(lines) - 1)
                continue
            line_indices.append(i)
        lines_of_T.append(line_indices)
    return np.array(lines), np.array(lines_of_T)



def getLinesTetrahedra2(T):
    indi = np.array([[i, j] for i in range(4) for j in range(i + 1, 4)], dtype=int)
    lines = set()
    for tet in T:
        lines |= {frozenset(i) for i in tet[indi]}
    return np.array([tuple(l) for l in lines])


def getLinesHexahedra(T):
    lines = []
    lines_of_T = []
    for tet in T:
        t1, t2, t3, t4, t5, t6, t7, t8 = tet
        tet_lines = [sorted([t1, t2]), sorted([t2, t3]), sorted([t3, t4]), sorted([t4, t1]),
                     sorted([t5, t6]), sorted([t6, t7]), sorted([t7, t8]), sorted([t8, t5]),
                     sorted([t1, t5]), sorted([t2, t6]), sorted([t3, t7]), sorted([t4, t8])]
        line_indices = []
        for line in tet_lines:
            i = 0
            for i in range(len(lines)):
                if lines[i] == line:
                    break
            else:
                lines.append(line)
                line_indices.append(len(lines) - 1)
                continue
            line_indices.append(i)
        lines_of_T.append(line_indices)
    return np.array(lines), np.array(lines_of_T)

def getFacesHexahedra(T):
    faces = []
    faces_of_T = []
    for tet in T:
        t1, t2, t3, t4, t5, t6, t7, t8 = tet
        tet_faces = [sorted([t1, t2, t3, t4]),
                     sorted([t5, t6, t7, t8]),
                     sorted([t1, t2, t6, t5]),
                     sorted([t2, t3, t7, t6]),
                     sorted([t3, t4, t8, t7]),
                     sorted([t4, t1, t5, t8]),
                     ]
        face_indices = []
        for face in tet_faces:
            i = 0
            for i in range(len(faces)):
                if faces[i] == face:
                    break
            else:
                faces.append(face)
                face_indices.append(len(faces) - 1)
                continue
            face_indices.append(i)
        faces_of_T.append(face_indices)
    return np.array(faces), np.array(faces_of_T)


def getStrain(M, stress, stepper=0.066, rel_conv_crit=0.01, verbose=False, callback=None):
    t = time.time()

    left = (M.R[:, 0] == np.min(M.R[:, 0]))
    right = (M.R[:, 0] == np.max(M.R[:, 0]))

    l, w, h = np.max(M.R, axis=0) - np.min(M.R, axis=0)

    A = w * h

    # omit two borders for the sum
    sum_region = (M.R[left, 1] < np.max(M.R[left, 1])) & \
                 (M.R[left, 2] < np.max(M.R[left, 2]))
    count = M.R[left, 0][sum_region].shape[0]
    f = stress * A / count

    # displacement boundary coundition is nan in the bluk and the border
    displacement = np.zeros(M.R.shape)
    displacement[:] = np.nan
    displacement[left, :] = 0
    # force boundary condition is 0 in the bulk
    # and f in the border
    force = np.zeros(M.R.shape)
    force[:] = 0
    force[left, :] = np.nan
    force[right, 0] = -f

    # initial displacement is a uniform strain field in x direction
    initial_displacement = np.zeros(M.R.shape)
    # initial_displacement[:, 0] = (lambd-1)*M.R[:, 0]

    # give the boundary conditions and initial displacement guess to the solver
    M.setBoundaryCondition(displacement, force)
    M.setInitialDisplacements(initial_displacement)

    M.solve_boundarycondition(stepper=stepper, verbose=verbose, rel_conv_crit=rel_conv_crit,
                              callback=callback)
    strain = np.mean(M.U[right, 0] / l) - 1

    print("stress", stress, "strain", strain, "duration", time.time() - t)
    return strain


def getStress(M, lambd, stepper=0.066, rel_conv_crit=0.01, verbose=False, callback=None):
    t = time.time()

    left = (M.R[:, 0] == np.min(M.R[:, 0]))
    right = (M.R[:, 0] == np.max(M.R[:, 0]))

    x_min = np.min(M.R[:, 0])

    l, w, h = np.max(M.R, axis=0) - np.min(M.R, axis=0)

    A = w * h

    # displacement boundary coundition is nan in the bluk
    # and ad the border lambda in x and 0 in yz
    displacement = np.zeros(M.R.shape)
    displacement[:] = np.nan

    displacement[left, :] = 0

    displacement[right, :] = 0
    displacement[right, 0] = lambd * (M.R[right, 0] - x_min)
    # force boundary condition is 0 in the bulk
    # and nan in the border
    force = np.zeros(M.R.shape)
    force[:] = 0
    force[right] = np.nan

    # initial displacement is a uniform strain field in x direction
    initial_displacement = np.zeros(M.R.shape)
    initial_displacement[:, 0] = lambd * (M.R[:, 0] - x_min)

    # give the boundary condutions and initial displacement guess to the solver
    M.setBoundaryCondition(displacement, force)
    M.setInitialDisplacements(initial_displacement)

    M.solve_boundarycondition(stepper=stepper, verbose=verbose, rel_conv_crit=rel_conv_crit,
                              callback=callback)

    # omit two borders for the sum
    sum_region = (M.R[right, 1] < np.max(M.R[right, 1])) & \
                 (M.R[right, 2] < np.max(M.R[right, 2]))

    stress = -np.sum(M.f[right, 0][sum_region], axis=0) / A
    print("strain", lambd, "stress", stress, "duration", time.time() - t)
    return stress
