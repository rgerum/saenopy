import numpy as np


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
def makeBoxmeshTets(nx, grain=1):
    ny = nx
    nz = nx
    T = []

    for x in range(0, nx, grain):
        for y in range(0, ny, grain):
            for z in range(0, nz, grain):
                i = x + nx * y + nx * ny * z

                if x > 0 and y > 0 and z > 0:
                    i1 = i
                    i2 = (x - 0) + nx * (y - grain) + nx * ny * (z - 0)
                    i3 = (x - grain) + nx * (y - grain) + nx * ny * (z - 0)
                    i4 = (x - grain) + nx * (y - 0) + nx * ny * (z - 0)
                    i5 = (x - 0) + nx * (y - 0) + nx * ny * (z - grain)
                    i6 = (x - grain) + nx * (y - 0) + nx * ny * (z - grain)
                    i7 = (x - grain) + nx * (y - grain) + nx * ny * (z - grain)
                    i8 = (x - 0) + nx * (y - grain) + nx * ny * (z - grain)

                    T.append([i1, i2, i3, i8])

                    T.append([i1, i3, i4, i6])

                    T.append([i1, i5, i8, i6])

                    T.append([i3, i6, i8, i7])

                    T.append([i1, i8, i3, i6])

    return np.array(T, dtype=np.int64)


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
