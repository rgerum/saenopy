#!/usr/bin/env python
# coding: utf-8


get_ipython().run_line_magic('matplotlib', 'inline')


import numpy as np
import matplotlib.pyplot as plt

from saenopy import macro
from saenopy.materials import SemiAffineFiberMaterial

material = SemiAffineFiberMaterial(900, 0.0004, 0.0075, 0.033)
print(material)

x, y = np.loadtxt("../macrorheology/data_exp/rheodata.dat").T
plt.plot(x, y, "o", label="data")

gamma = np.arange(0.005, 0.3, 0.0001)
x, y = macro.getShearRheometerStress(gamma, material)
plt.loglog(x, y, "-", lw=3, label="model")

plt.xlabel("strain")
plt.ylabel("shear stress [Pa]")

plt.legend()


import numpy as np
import matplotlib.pyplot as plt

from saenopy import macro
from saenopy.materials import SemiAffineFiberMaterial

material = SemiAffineFiberMaterial(900, 0.0004, 0.0075, 0.033)
print(material)

x, y = np.loadtxt("../macrorheology/data_exp/stretcherdata.dat").T
plt.plot(x, y, "o", label="data")

lambda_h = np.arange(1-0.05, 1+0.07, 0.01)
lambda_v = np.arange(0, 1.1, 0.001)

x, y = macro.getStretchThinning(lambda_h, lambda_v, material)
plt.plot(x, y, lw=3, label="model")

plt.xlabel("horizontal stretch")
plt.ylabel("vertical contraction")

plt.ylim(0, 1.2)
plt.xlim(0.9, 1.2)

plt.legend()


import numpy as np
import matplotlib.pyplot as plt

from saenopy import macro
from saenopy.materials import SemiAffineFiberMaterial, LinearMaterial

material = SemiAffineFiberMaterial(900, 0.0004, 0.0075, 0.033)
print(material)

epsilon = np.arange(1, 1.17, 0.0001)
x, y = macro.getExtensionalRheometerStress(epsilon, material)
plt.plot(x, y, lw=3, label="model")

plt.xlabel("strain")
plt.ylabel("stress [Pa]")

plt.legend()

