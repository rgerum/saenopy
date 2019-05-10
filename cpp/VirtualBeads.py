import numpy as np
import time
import os


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
        vbead_temp = readFromDatFile(VBname)

        self.vbead = vbead_temp[:, 0] > 0.5

    def loadGuess(self):
        pass