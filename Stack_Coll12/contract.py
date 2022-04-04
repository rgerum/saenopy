import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage.transform import swirl


image = data.checkerboard()

def shift_left(xy):
    offset = np.array([100, 100])
    xy -= offset
    r = np.linalg.norm(xy, axis=1)
    angle = np.arctan2(xy[:, 1], xy[:, 0])
    a = 1000
    r_new = a * 1/r**3 + r
    xy = np.array([np.cos(angle)*r_new, np.sin(angle)*r_new]).T

    xy += offset
    return xy

def shift_left2(xy):
    offset = np.array([10, 10])
    xy -= offset
    xy *= 10
    r = np.linalg.norm(xy, axis=1)
    angle = np.arctan2(xy[:, 1], xy[:, 0])
    strength = 1/r**2 * 1
    deformation = np.array([np.cos(angle)*strength, np.sin(angle)*strength]).T
    xy += offset*10
    for i in range(0, xy.shape[0]):
        plt.plot([xy[i, 0], xy[i, 0]+deformation[i, 0]], [xy[i, 1], xy[i, 1]+deformation[i, 1]])
    #plt.show()
    xy -= deformation
    return xy


from skimage.transform import warp
print(image.shape)
swirled = warp(image[::1, ::1], shift_left)
#swirled = warp(image[0:1, ::1], shift_left)

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3),
                               sharex=True, sharey=True)

ax0.imshow(image, cmap=plt.cm.gray, interpolation="nearest")
ax0.axis('off')
ax1.imshow(swirled, cmap=plt.cm.gray, interpolation="nearest")
ax1.axis('off')

plt.sca(ax1)
swirled2 = warp(image[::10, ::10], shift_left2)

plt.show()