import saenopy
from pathlib import Path
import matplotlib.pyplot as plt

folder = Path("/home/richard/.local/share/saenopy/1_ClassicSingleCellTFM/example_output")

pos = "007"
res0 = saenopy.load(folder / f"Pos{pos}_S001_z{{z}}_ch{{c00}}_new14_boundary-False_saenooldbulk.saenopy")
res1 = saenopy.load(folder / f"Pos{pos}_S001_z{{z}}_ch{{c00}}_a-2.saenopy")

def get_cell_boundary(result: saenopy.Result, channel=1, thershold=20, smooth=2, element_size=14.00e-6, boundary=True, pos=None, label=None):
    from scipy.ndimage import gaussian_filter
    import matplotlib.pyplot as plt
    import numpy as np

    for i in range(len(result.stacks)):
        stack_deformed = result.stacks[i]
        voxel_size1 = stack_deformed.voxel_size

        im = stack_deformed[:, :, 0, :, channel]
        im = gaussian_filter(im, sigma=smooth, truncate=2.0)

        im_thresh = (im[:, :, :] > thershold).astype(np.uint8)


        from skimage.measure import label
        def largest_connected_component(segmentation):
            labels = label(segmentation)
            counts = [0]
            for i in range(1, np.max(labels)):
                counts.append(np.sum(i == labels))
            return labels == np.argmax(counts)

        im_thresh = largest_connected_component(im_thresh).astype(np.uint8)

        from skimage.morphology import erosion
        if boundary:
            im_thresh = (im_thresh - erosion(im_thresh)).astype(bool)
        else:
            im_thresh = im_thresh.astype(bool)

        du, dv, dw = voxel_size1

        u = im_thresh
        y, x, z = np.indices(u.shape)
        y, x, z = (y * stack_deformed.shape[0] * dv / u.shape[0] * 1e-6,
                   x * stack_deformed.shape[1] * du / u.shape[1] * 1e-6,
                   z * stack_deformed.shape[2] * dw / u.shape[2] * 1e-6)
        z -= np.max(z) / 2
        x -= np.max(x) / 2
        y -= np.max(y) / 2

        x = x[im_thresh]
        y = y[im_thresh]
        z = z[im_thresh]

        yxz = np.vstack([y, x, z])

        difference_vec = result.solvers[0].mesh.nodes[:, :, None] - yxz[None, :, :]
        difference_length = np.linalg.norm(difference_vec, axis=1)
        dist_to_cell = np.min(difference_length, axis=1)

        x = dist_to_cell.ravel()
        y = np.linalg.norm(result.solvers[0].mesh.forces * result.solvers[0].mesh.regularisation_mask[:, None], axis=1)
        i = np.argsort(x)
        x = x[i]
        y = y[i]

        y = np.cumsum(y[::-1])[::-1]

        y = y/y[0]
        l, = plt.plot(x*1e6, y*100, "-", label=label)
        index_max = np.where(y<0.05)[0][0]
        print(x[index_max]*1e6)
        plt.axvline(x[index_max]*1e6, color=l.get_color(), lw=0.8, linestyle='--')
        return l

l1 = get_cell_boundary(res0, label="saeno")
l2 = get_cell_boundary(res1, label="saenopy")

plt.axhline(5, color='k', linestyle='--', lw=.8)
plt.xlabel("Distance to Cell Surface (µm)")
plt.ylabel("Percentage of Forces (%)")
plt.legend([l1, l2], ["Saeno", "Saenopy"], frameon=False)
plt.savefig("surface_comparison.png")
plt.show()