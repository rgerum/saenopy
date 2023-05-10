#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import unittest
import numpy as np
from saenopy.multigrid_helper import create_box_mesh

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "package"))

import saenopy
from saenopy import Solver
from saenopy.materials import SemiAffineFiberMaterial


class Test_DataFile(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testStretch(self):
        R, T = create_box_mesh(np.linspace(-0.5, 0.5, 10))

        M = Solver()
        M.set_nodes(R)
        M.set_tetrahedra(T)

        # provide a material model
        material = SemiAffineFiberMaterial(2)
        M.set_material_model(material)

        def getForce(lambd, step_size=0.066, verbose=False):
            global convergence
            print("lambd", lambd)
            d = lambd - 1

            displacement = np.zeros(M.mesh.R.shape) * np.nan
            force = np.zeros(M.mesh.R.shape)

            initial_displacement = np.zeros(M.mesh.R.shape)
            left = M.mesh.R[:, 0] == -0.5
            right = M.mesh.R[:, 0] == 0.5

            displacement[left, :] = 0
            displacement[left, 0] = -d / 2
            displacement[right, :] = 0
            displacement[right, 0] = d / 2
            force[left] = np.nan
            force[right] = np.nan

            initial_displacement[:, 0] = M.mesh.R[:, 0] / 1 * d

            M.set_boundary_condition(displacement, force)
            M.set_initial_displacements(initial_displacement)
            convergence = M.solve_boundarycondition(step_size=step_size, verbose=verbose)
            return np.mean(np.concatenate((M.mesh.f[left, 0], -M.mesh.f[right, 0])))

        getForce(1.01)

        for i in range(3):
            np.testing.assert_almost_equal(-np.mean(M.mesh.U[R[:, i] < 0]), np.mean(M.mesh.U[R[:, i] > 0]))


if __name__ == '__main__':
    __path__ = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(__path__, 'log_'+__key__+'.txt')
    with open(log_file, "w") as f:
        runner = unittest.TextTestRunner(f)
        unittest.main(testRunner=runner)
