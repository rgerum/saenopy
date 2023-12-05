import numpy as np

import saenopy
from pathlib import Path
import matplotlib.pyplot as plt

params = {'image': {'width': 768, 'height': 768, 'logo_size': 0, 'scale': 1.0, 'antialiase': True}, 'camera': {'elevation': 26.52, 'azimuth': 39.0, 'distance': 1221, 'offset_x': 13, 'offset_y': 77, 'roll': 0}, 'theme': 'document', 'show_grid': 2, 'use_nans': False, 'arrows': 'fitted forces', 'averaging_size': 10.0, 'deformation_arrows': {'autoscale': True, 'scale_max': 10.0, 'colormap': 'turbo', 'arrow_scale': 1.0, 'arrow_opacity': 1.0, 'skip': 1}, 'force_arrows': {'autoscale': False, 'scale_max': 1000.0, 'use_center': False, 'colormap': 'turbo', 'arrow_scale': 1.0, 'arrow_opacity': 1.0, 'skip': 1}, 'stack': {'image': 2, 'channel': '01', 'z_proj': 2, 'use_contrast_enhance': True, 'contrast_enhance': (4.0, 29.0), 'colormap': 'gray', 'z': 188, 'use_reference_stack': False, 'channel_B': '', 'colormap_B': 'gray'}, 'scalebar': {'length': 0.0, 'width': 5.0, 'xpos': 15.0, 'ypos': 10.0, 'fontsize': 18.0}, '2D_arrows': {'width': 2.0, 'headlength': 5.0, 'headheight': 5.0}, 'crop': {'x': (156, 356), 'y': (156, 356), 'z': (163, 213)}, 'channel0': {'show': False, 'skip': 1, 'sigma_sato': 2, 'sigma_gauss': 0, 'percentiles': (0, 1), 'range': (0, 1), 'alpha': (0.1, 0.5, 1), 'cmap': 'pink'}, 'channel1': {'show': False, 'skip': 1, 'sigma_sato': 0, 'sigma_gauss': 7, 'percentiles': (0, 1), 'range': (0, 1), 'alpha': (0.1, 0.5, 1), 'cmap': 'Greens', 'channel': 1}, 'channel_thresh': 1.0, 'time': {'t': 0, 'format': '%d:%H:%M', 'start': 0.0, 'display': True, 'fontsize': 18}}
params = {'image': {'width': 768, 'height': 768, 'logo_size': 0, 'scale': 1.0, 'antialiase': True},
          'camera': {'elevation': 16.7, 'azimuth': 31.58, 'distance': 1201, 'offset_x': 7, 'offset_y': 78, 'roll': 0},
          'theme': 'document', 'show_grid': 3, 'use_nans': False, 'arrows': 'fitted forces', 'averaging_size': 10.0,
          'deformation_arrows': {'autoscale': True, 'scale_max': 10.0, 'colormap': 'turbo', 'arrow_scale': 1.0, 'arrow_opacity': 1.0, 'skip': 1},
          'force_arrows': {'autoscale': False, 'use_log': True, 'scale_max': 80000.0, 'use_center': False, 'colormap': 'turbo', 'arrow_scale': 1.0, 'arrow_opacity': 1.0, 'skip': 1}, 'stack': {'image': 2, 'channel': '01', 'z_proj': 3, 'use_contrast_enhance': True, 'contrast_enhance': (4.0, 12.0), 'colormap': 'gray', 'z': 188, 'use_reference_stack': False, 'channel_B': '', 'colormap_B': 'gray'}, 'scalebar': {'length': 0.0, 'width': 5.0, 'xpos': 15.0, 'ypos': 10.0, 'fontsize': 18.0}, '2D_arrows': {'width': 2.0, 'headlength': 5.0, 'headheight': 5.0}, 'crop': {'x': (156, 356), 'y': (156, 356), 'z': (163, 213)}, 'channel0': {'show': False, 'skip': 1, 'sigma_sato': 2, 'sigma_gauss': 0, 'percentiles': (0, 1), 'range': (0, 1), 'alpha': (0.1, 0.5, 1), 'cmap': 'pink'}, 'channel1': {'show': False, 'skip': 1, 'sigma_sato': 0, 'sigma_gauss': 0, 'percentiles': (0, 1), 'range': (0, 1), 'alpha': (0.028954886793518153, 0.5444943820224719, 0.6402247191011237), 'cmap': 'Greens', 'channel': 1}, 'channel_thresh': 1.0, 'time': {'t': 0, 'format': '%d:%H:%M', 'start': 0.0, 'display': True, 'fontsize': 18}}
folder = Path("/home/richard/.local/share/saenopy/1_ClassicSingleCellTFM/example_output")

if 1:
    j = []
    pos = "007"
    for use_log in [False, True]: # "004", "007",
        params["force_arrows"]["use_log"] = use_log
        im = saenopy.render_image(params, saenopy.load(folder / f"Pos{pos}_S001_z{{z}}_ch{{c00}}_a-2.saenopy"))

        #im2 = render_image(params, saenopy.load(folder / f"Pos{pos}_S001_z{{z}}_ch{{c00}}_new14_boundary-False_saenonew.saenopy"))

        im0 = saenopy.render_image(params, saenopy.load(folder / f"Pos{pos}_S001_z{{z}}_ch{{c00}}_new14_boundary-False_saenooldbulk.saenopy"))

        #im00 = render_image(params, saenopy.load(folder / f"Pos{pos}_S001_z{{z}}_ch{{c00}}_new14_boundary-True_saenonew_strong.saenopy"))

        #j.append(np.hstack([im0, im, im2, im00]))
        j.append(np.hstack([im, im0]))
    plt.imsave("test2.png", np.vstack(j))
    exit()
if 1:
    pos = "004"
    import sys
    sys.path.insert(0, "/home/richard/PycharmProjects/utils")
    from rgerum_utils.plot.plot_group_new import PlotGroup

    group = PlotGroup()
    for pos in group.row(["007"]): #["004", "007", "008"]
        res0 = saenopy.load(folder / f"Pos{pos}_S001_z{{z}}_ch{{c00}}_new14_boundary-False_saenooldbulk.saenopy")
        res1 = saenopy.load(folder / f"Pos{pos}_S001_z{{z}}_ch{{c00}}_a-2.saenopy")
        #res2 = saenopy.load(folder / f"Pos{pos}_S001_z{{z}}_ch{{c00}}_new14_boundary-False_saenonew.saenopy")
        #res3 = saenopy.load(folder / f"Pos{pos}_S001_z{{z}}_ch{{c00}}_new14_boundary-True_saenonew_strong2.saenopy")

        def get_cell_boundary(result: saenopy.Result, channel=1, thershold=20, smooth=2, element_size=14.00e-6, boundary=True, pos=None, label=None):
            from scipy.ndimage import gaussian_filter
            import matplotlib.pyplot as plt
            import numpy as np

            for i in range(len(result.stacks)):
                if 0:
                    mesh = result.solvers[i].mesh
                    index = np.argsort(mesh.nodes[:, 1])
                    x = mesh.nodes[index, 1]
                    f = np.linalg.norm(mesh.forces[index], axis=1)*mesh.regularisation_mask
                    xl = []
                    fl = []
                    for xx in np.unique(x):
                        xl.append(xx)
                        fl.append(np.max(f[x == xx]))
                    count = np.sum(x == x[0])
                    print(count)
                    plt.plot(xl, fl, "-", label=label)
                    return

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
                    largest_cc = labels == np.argmax(np.bincount(labels[segmentation]))
                    return largest_cc

                print(im_thresh.shape, im_thresh.dtype)
                im_thresh = largest_connected_component(im_thresh).astype(np.uint8)
                print(im_thresh.shape,im_thresh.dtype)

                from skimage.morphology import erosion
                if boundary:
                    im_thresh = (im_thresh - erosion(im_thresh)).astype(bool)
                else:
                    im_thresh = im_thresh.astype(bool)
                #if pos == "004":
                #    im_thresh[:, :, :112] = False
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
                index = np.argmin(difference_length, axis=1)
                print(difference_vec.shape)
                print(difference_length.shape)
                print(result.solvers[0].mesh.nodes.shape)
                print(index.shape, index.dtype)
                #difference_vec = difference_vec[index]
                difference_vec = np.array([difference_vec[i, :, x] for i, x in enumerate(index)])
                print(difference_vec.shape)
                dist_to_cell = np.min(difference_length, axis=1)

                difference_vec_normalized = difference_vec / dist_to_cell[:, None]

                print(dist_to_cell.shape)
                print(result.solvers[0].mesh.forces.shape)
                x = dist_to_cell.ravel()
                y = np.linalg.norm(result.solvers[0].mesh.forces * result.solvers[0].mesh.regularisation_mask[:, None], axis=1)
                y_proj = np.sum(result.solvers[0].mesh.forces * result.solvers[0].mesh.regularisation_mask[:, None] * difference_vec_normalized, axis=1)
                i = np.argsort(x)
                x = x[i]
                y = y[i]
                y_proj = y_proj[i]
                y = np.cumsum(y[::-1])[::-1]
                #y_proj = np.cumsum(y_proj[::-1])[::-1]

                #group.select_ax(col=0)
                y = y/y[0]
                l, = plt.plot(x*1e6, y*100, "-", label=label)
                index_max = np.where(y<0.05)[0][0]
                print(x[index_max]*1e6)
                plt.axvline(x[index_max]*1e6, color=l.get_color(), lw=0.8, linestyle='--')
                return l
                #group.select_ax(col=1)
                #y_proj = y_proj/y_proj[0]
                #plt.plot(x*1e6, y_proj*100, "--", label=label, color=l.get_color())

        l1 = get_cell_boundary(res0, label="saeno")
        l2 = get_cell_boundary(res1, label="saenopy")
        #get_cell_boundary(res2, label="surface")
        #get_cell_boundary(res3, label="surface2")
    plt.axhline(5, color='k', linestyle='--', lw=.8)
    plt.xlabel("Distance to Cell Surface (µm)")
    plt.ylabel("Percentage of Forces (%)")
    plt.legend([l1, l2], ["Saeno", "Saenopy"], frameon=False)
    plt.savefig("surface_comparison.png")
    plt.show()