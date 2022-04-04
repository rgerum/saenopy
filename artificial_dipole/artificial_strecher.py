import saenopy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from saenopy.materials import SemiAffineFiberMaterial
import os
from saenopy.getDeformations import interpolate_different_mesh
from saenopy.multigridHelper import getScaledMesh, createMesh
import time

M = saenopy.Solver()
Wx = 2e-2*1.05
Wy = 2e-2
Wz = 700e-6*1e-3/(Wx*Wy)
R, T = createMesh(element_width=500e-6, box_width=(Wx, Wy, Wz))
R[:, 0] -= np.min(R[:, 0])
R[:, 0] -= np.max(R[:, 0])/2
R[:, 1] -= np.min(R[:, 1])
R[:, 1] -= np.max(R[:, 1])/2
R[:, 2] -= np.min(R[:, 2])
bcond_disp = np.zeros_like(R) * np.nan
bcond_force = np.zeros_like(R)
minR = np.min(R, axis=0)
maxR = np.max(R, axis=0)
width = 0.5e-6
wall_x0 = (R[:, 0] < minR[0] + width)
wall_x1 = (R[:, 0] > maxR[0] - width)
wall_y0 = (R[:, 1] < minR[1] + width)
wall_y1 = (R[:, 1] > maxR[1] - width)
wall_z0 = (R[:, 2] < minR[2] + width)
bcond_force[wall_x0 | wall_x1] = np.nan
bcond_force[wall_y0 | wall_y1] = np.nan
bcond_force[wall_z0] = np.nan
stretch = 0.05 * Wx
bcond_disp[wall_x0] = np.array([-stretch / 2, 0, 0])
bcond_disp[wall_x1] = np.array([stretch / 2, 0, 0])
bcond_disp[wall_y0] = R[wall_y0, 0:1] / maxR[1] * np.array([[stretch / 2, 0, 0]])
bcond_disp[wall_y1] = R[wall_y1, 0:1] / maxR[1] * np.array([[stretch / 2, 0, 0]])
bcond_disp[wall_z0] = R[wall_z0, 0:1] / maxR[1] * np.array([[stretch / 2, 0, 0]])
U = R[:, 0:1] / maxR[1] * np.array([[stretch / 2, 0, 0]])
M.setNodes(R)
M.setTetrahedra(T)
M.setBoundaryCondition(bcond_disp, bcond_force)
M.setInitialDisplacements(U)
M.setMaterialModel(SemiAffineFiberMaterial(1449, 0.00215, 0.032, 0.055))
#M.plot(["U_fixed", "f_target"])
M.solve_boundarycondition(verbose=True)

def getStretch(stretch):
    bcond_disp = np.zeros_like(R) * np.nan
    bcond_force = np.zeros_like(R)
    minR = np.min(R, axis=0)
    maxR = np.max(R, axis=0)
    width = 0.5e-6
    wall_x0 = (R[:, 0] < minR[0] + width)
    wall_x1 = (R[:, 0] > maxR[0] - width)
    wall_y0 = (R[:, 1] < minR[1] + width)
    wall_y1 = (R[:, 1] > maxR[1] - width)
    wall_z0 = (R[:, 2] < minR[2] + width)
    bcond_force[wall_x0 | wall_x1] = np.nan
    bcond_force[wall_y0 | wall_y1] = np.nan
    bcond_force[wall_z0] = np.nan
    bcond_disp[wall_x0] = np.array([-stretch / 2, 0, 0])
    bcond_disp[wall_x1] = np.array([stretch / 2, 0, 0])
    bcond_disp[wall_y0] = R[wall_y0, 0:1] / maxR[1] * np.array([[stretch / 2, 0, 0]])
    bcond_disp[wall_y1] = R[wall_y1, 0:1] / maxR[1] * np.array([[stretch / 2, 0, 0]])
    bcond_disp[wall_z0] = R[wall_z0, 0:1] / maxR[1] * np.array([[stretch / 2, 0, 0]])
    U = R[:, 0:1] / maxR[1] * np.array([[stretch / 2, 0, 0]])
    M.setBoundaryCondition(bcond_disp, bcond_force)
    M.setInitialDisplacements(U)
    M.solve_boundarycondition()
    return M.U[getNearestNode(R, [0, 0, maxR[2]]), 2] / maxR[2]

for i in np.arange(0, 0.1, 0.01):
    print(i, getStretch(i*Wx))

data = np.array([
[0.00, 9.639731946956672e-17],
[0.01, -0.0028876824941188994],
[0.02, -0.006793464310006459],
[0.03, -0.011653808339864761],
[0.04, -0.02191106208313836],
[0.05, -0.029733408435941897],
[0.06, -0.04325336411038499],
[0.07, -0.06080549279004046],
[0.08, -0.08136238513958438],
[0.09, -0.09579808047033185],
])
from saenopy import macro
lambda_h = np.arange(1-0.05, 1+0.07, 0.01)
lambda_v = np.arange(0, 1.1, 0.001)

x, y = macro.getStretchThinning(lambda_h, lambda_v, M.material_model)
plt.plot(x, y, lw=3, label="model")
plt.plot(data[:, 0]+1, data[:, 1]+1)

def getBorder(R):
    minR = np.min(R, axis=0)
    maxR = np.max(R, axis=0)
    width = 0.5e-6
    border = (R[:, 0] < minR[0] + width) | (R[:, 0] > maxR[0] - width) | \
             (R[:, 1] < minR[1] + width) | (R[:, 1] > maxR[1] - width) | \
             (R[:, 2] < minR[2] + width) | (R[:, 2] > maxR[2] - width)
    return border

def getNearestNode(R, point):
    return np.argmin(np.linalg.norm(R-np.array(point)[None, :], axis=1))

def calculateDipole(size):
    M = saenopy.load(
        r"\\131.188.117.96\biophysDS\dboehringer\Platte_4\Software\2-integrate-piv-saenopy\Eval\4-tumor-cell-piv\cell10\testdeformations_win20.npz")
    M.R -= np.mean(M.R, axis=0)
    print("get scaled mesh", (np.max(M.R, axis=0) - np.min(M.R, axis=0)) * 1e6)

    voxel_size = 7.5e-6
    amplitude = 100e-9
    radius = 50e-6

    M.setMaterialModel(SemiAffineFiberMaterial(1645, 0.0008, 0.0075, 0.033))

    xmax, ymax, zmax = np.max(M.R, axis=0)
    nx, ny, nz = (np.max(M.R, axis=0) / voxel_size).astype(np.int)

    points, cells = getScaledMesh(size * 1e-6, 100e-6, (np.max(M.R, axis=0) - np.min(M.R, axis=0)) / 2, [0, 0, 0],
                                  0.2)
    border = getBorder(points)
    M.setNodes(points)
    M.setTetrahedra(cells)
    bcond_disp = np.zeros_like(M.U) * np.nan
    bcond_disp[border] = 0
    bcond_force = np.zeros_like(M.U)
    bcond_force[border] = np.nan
    distance_from_point = np.linalg.norm(M.R, np.array([-radius, 0, 0]), axis=-1)
    bcond_force[getNearestNode(M.R, [-radius, 0, 0])] = [-amplitude, 0, 0]
    bcond_force[getNearestNode(M.R, [radius, 0, 0])] = [amplitude, 0, 0]
    M.setBoundaryCondition(bcond_disp, bcond_force)
    M.solve_boundarycondition(verbose=True)
    M.save(r"dipole_boundary.npz")

def addBlurredForces(point, force, blur, R):
    point = np.array(point)
    force = np.array(force)
    distance_from_point = np.linalg.norm(R - point[None, :], axis=-1)
    factor = np.exp(-(distance_from_point)**2/(2*blur**2))
    factor = factor/np.sum(factor)
    return factor[:, None] * force[None, :]

def dipole(radius, amplitude, blur, bcond_force, R):
    bcond_force += addBlurredForces([-radius, 0, 0], [-amplitude, 0, 0], blur, R)
    bcond_force += addBlurredForces([radius, 0, 0], [amplitude, 0, 0], blur, R)

def quadrupole(radius, amplitude, blur, bcond_force, R):
    points = [
        [0, 0, 3 / np.sqrt(6)],
        [0, 2 / np.sqrt(3), -1 / np.sqrt(6)],
        [+1, -1 / np.sqrt(3), -1 / np.sqrt(6)],
        [-1, -1 / np.sqrt(3), -1 / np.sqrt(6)],
    ]
    force_points = np.array(points) * np.sqrt(2 / 3) * radius
    for p in force_points:
        bcond_force += addBlurredForces(p, p / np.linalg.norm(p) * amplitude, blur, R)

def calculateArtificialCell(elementsize=4e-6, amplitude=20e-9, distance=10e-6, stacksize=150e-6, blur=3e-6, type="dipole", plot=False):
    radius = distance / 2
    M = saenopy.Solver()

    # biomatrix 2020
    M.setMaterialModel(SemiAffineFiberMaterial(1449, 0.00215, 0.032, 0.055))

    # points, cells = getScaledMesh(size * 1e-6, 100e-6, (np.max(M.R, axis=0) - np.min(M.R, axis=0)) / 2, [0, 0, 0], 0.2)
    points, cells = getScaledMesh(elementsize, 100e-6, stacksize / 2, [0, 0, 0], 0.2)

    border = getBorder(points)
    M.setNodes(points)
    M.setTetrahedra(cells)
    bcond_disp = np.zeros_like(M.U) * np.nan
    bcond_disp[border] = 0
    bcond_force = np.zeros_like(M.U)
    bcond_force[border] = np.nan
    # set the two forces
    if type == "dipole":
        dipole(radius, amplitude, blur, bcond_force, M.R)
    elif type == "quadrupole":
        quadrupole(radius, amplitude, blur, bcond_force, M.R)
    else:
        raise ValueError(f"Unknown type {type}, known types dipole, quadrupole.")

    M.setBoundaryCondition(bcond_disp, bcond_force)

    if plot is True:
        M.plot(["U_fixed", "f_target"])
    # solve
    M.solve_boundarycondition(verbose=True)
    # save
    M.save(rf"Amp{amplitude}_dist{distance}.npz")

times = []
for size in [9]:
    calculateArtificialCell(type="quadrupole", plot=True)
    break
    #for alpha in [9]:
    #for alpha in np.arange(6, 10, 1 / 3):
    for alpha in [np.arange(6, 10, 1 / 3)[-1]]:
        for noise in [0.5]:
            #for cut in np.arange(1, 0, -0.1):
            M0 = saenopy.load("dipole_boundary.npz")
            M = saenopy.Solver()
            border = getBorder(M0.R)
            M.setNodes(M0.R)
            M.setTetrahedra(M0.T)
            M.setTargetDisplacements(M0.U+np.random.normal(0, noise*1e-6, M0.U.shape))#, ~border)
            print("set material")
            M.setMaterialModel(SemiAffineFiberMaterial(1645, 0.0008, 0.0075, 0.033))
            print("regluarlize")
            t = time.time()
            x = M.solve_regularized(alpha=10**float(alpha), verbose=True, i_max=100)
            times.append([size, alpha, time.time()-t])
            print("save")
            M.save(f"regularized_noborder_noise{noise}_mesh{size}um_{alpha:5.2f}.npz")

np.savetxt("times.txt", times)
exit()
