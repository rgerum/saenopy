import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from saenopy.materials import SemiAffineFiberMaterial
from saenopy.macro import get_shear_rheometer_stress
import saenopy
material = SemiAffineFiberMaterial(900, 0.0004, 0.0075, 0.033)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.40)

gamma = np.arange(-0.01, 0.03, 0.0001)
M = saenopy.Solver()
M.set_beams()
#x, y = getShearRheometerStress(gamma, material, M.s)
y = material.stiffness(gamma)

l, = plt.plot(gamma, y, lw=2)
ax.margins(x=0)
plt.ylabel("stiffness")
plt.xlabel("strain")

axcolor = 'lightgoldenrodyellow'
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
axlambdas = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
axds = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)

sfreq = Slider(axfreq, 'k', 0, 1200.0, valinit=900, valstep=100)
samp = Slider(axamp, '$d_0$', 0.0, 0.1, valinit=0.0004)
slambdas = Slider(axlambdas, '$\\lambda_s$', 0.0, 0.03, valinit=0.0075)
sds = Slider(axds, '$d_s$', 0.0, 0.1, valinit=0.033)



def update(val):
    d_0 = samp.val
    k = sfreq.val
    lamda_s = slambdas.val
    d_s = sds.val
    material = SemiAffineFiberMaterial(k, d_0, lamda_s, d_s)
    y = material.stiffness(gamma)
    l.set_ydata(y)
    fig.canvas.draw_idle()


sfreq.on_changed(update)
samp.on_changed(update)
slambdas.on_changed(update)
sds.on_changed(update)


plt.show()