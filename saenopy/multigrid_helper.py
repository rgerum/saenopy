import numpy as np
import pandas as pd
import time
from numba import njit
from nptyping import NDArray, Shape, Float, Int, Bool


def create_box_mesh(x, y=None, z=None, tesselation_mode="6"):
    if y is None:
        y = x
    if z is None:
        z = x
    mesh = np.array(np.meshgrid(x, y, z, indexing="ij")).reshape(3, -1).T
    nodes = np.zeros(mesh.shape)
    nodes[:, :] = mesh
    tetrahedra = make_box_mesh_tets(len(x), len(y), len(z), tesselation_mode=tesselation_mode)
    return nodes, tetrahedra




@njit()
def make_box_mesh_tets(nx, ny=None, nz=None, grain=1, tesselation_mode="6"):  # pragma: no cover
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
                    if tesselation_mode == "5" or tesselation_mode == "5_old":
                        if tesselation_mode == "5":
                            fx = [0, 1] if x % 2 else [1, 0]
                            fy = [0, 1] if y % 2 else [1, 0]
                            fz = [0, 1] if z % 2 else [1, 0]
                        else:
                            fx = [0, 1]
                            fy = [0, 1]
                            fz = [0, 1]
                        i1 = index_of(x-fx[0], y-fy[0], z-fz[0])
                        i2 = index_of(x-fx[0], y-fy[1], z-fz[0])#(x - 0) + nx * (y - grain) + nx * ny * (z - 0)
                        i3 = index_of(x-fx[1], y-fy[1], z-fz[0])#(x - grain) + nx * (y - grain) + nx * ny * (z - 0)
                        i4 = index_of(x-fx[1], y-fy[0], z-fz[0])#(x - grain) + nx * (y - 0) + nx * ny * (z - 0)
                        i5 = index_of(x-fx[0], y-fy[0], z-fz[1])#(x - 0) + nx * (y - 0) + nx * ny * (z - grain)
                        i6 = index_of(x-fx[1], y-fy[0], z-fz[1])#(x - grain) + nx * (y - 0) + nx * ny * (z - grain)
                        i7 = index_of(x-fx[1], y-fy[1], z-fz[1])#(x - grain) + nx * (y - grain) + nx * ny * (z - grain)
                        i8 = index_of(x-fx[0], y-fy[1], z-fz[1])#(x - 0) + nx * (y - grain) + nx * ny * (z - grain)

                        T.append([i1, i2, i3, i8])

                        T.append([i1, i3, i4, i6])

                        T.append([i1, i5, i8, i6])

                        T.append([i3, i6, i8, i7])

                        T.append([i1, i8, i3, i6])
                    elif tesselation_mode == "6":
                        i2 = index_of(x - 0, y - 0, z - 0)
                        i3 = index_of(x - 0, y - 1, z - 0)
                        i4 = index_of(x - 1, y - 1, z - 0)
                        i1 = index_of(x - 1, y - 0, z - 0)

                        i7 = index_of(x - 0, y - 0, z - 1)
                        i6 = index_of(x - 1, y - 0, z - 1)
                        i5 = index_of(x - 1, y - 1, z - 1)
                        i8 = index_of(x - 0, y - 1, z - 1)


                        T.append([i6, i3, i2, i1])
                        T.append([i6, i3, i1, i4])
                        T.append([i6, i3, i4, i5])
                        T.append([i6, i3, i5, i8])
                        T.append([i6, i3, i8, i7])
                        T.append([i6, i3, i7, i2])
                    else:
                        raise ValueError("Wrong Tesselation mode")


    return np.array(T, dtype=np.int64)


def get_nodes_with_one_face(tetrahedra):
    # get the faces of the tetrahedrons
    faces = np.sort(np.array(tetrahedra[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]]).reshape(-1, 3), axis=1)
    # encode the faces as integers
    maxi = np.max(faces) + 1
    face_index = faces[:, 0] * maxi ** 2 + faces[:, 1] * maxi ** 1 + faces[:, 2]
    # count how often each face is present
    face_counts = pd.Series(face_index).value_counts()
    # filter only faces that occur once
    single_faces = face_counts[face_counts == 1].index
    # get the nodes that belong to these faces
    return np.unique(np.array([single_faces % maxi, (single_faces // maxi) % maxi, (single_faces // maxi ** 2) % maxi]))


def get_scaling(voxel_in, size_in, size_out, center, a):
    old_settings = np.seterr(all='ignore')  # seterr to known value

    np.seterr(all='ignore')

    n0 = size_in / voxel_in
    nplus = (voxel_in * (2 * a * n0 - 1) + np.sqrt(
        voxel_in * (-4 * a * voxel_in * n0 + 4 * a * (size_out - center) + voxel_in))) / (2 * a * voxel_in)
    nminus = (voxel_in * (2 * a * n0 - 1) + np.sqrt(
        voxel_in * (-4 * a * voxel_in * n0 + 4 * a * (size_out + center) + voxel_in))) / (2 * a * voxel_in)

    np.seterr(**old_settings)

    if np.isnan(nplus):
        nplus = (size_out - center) / voxel_in
    if np.isnan(nminus):
        nminus = (size_out + center) / voxel_in

    n = np.arange(-np.floor(nminus), np.floor(nplus)+1)
    y = voxel_in * n + voxel_in * a * np.clip(n - n0, 0, np.inf) ** 2 - voxel_in * a * np.clip(-n - n0, 0,
                                                                                               np.inf) ** 2 + center
    return y


def get_scaled_mesh(voxel_in, size_in, size_out, center, a, tesselation_mode="6"):
    if isinstance(size_out, (int, float)):
        size_out = [size_out]*3
    x = get_scaling(voxel_in, size_in, size_out[0], center[0], a)
    y = get_scaling(voxel_in, size_in, size_out[1], center[1], a)
    z = get_scaling(voxel_in, size_in, size_out[2], center[1], a)

    R, T = create_box_mesh(x, y, z, tesselation_mode=tesselation_mode)
    return R, T

